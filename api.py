
import glob
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from collections import deque
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from flask import Flask, request, jsonify
from flask_cors import CORS

# FLASK APP 

app = Flask(__name__)
CORS(app)  
USE_LLM = False
def _safe_llm_setup():
    """
    Try to set up an Ollama-backed LangChain pipeline without using deprecated LLMChain.
    If anything fails, we fall back to a stub.
    """
    try:
        from langchain_ollama import OllamaLLM
        from langchain.prompts import PromptTemplate  
        llm = OllamaLLM(model="llama2", streaming=False, callbacks=[])
        template = (
            "You are an expert financial advisor speaking to an adult learner.\n"
            "Facts:\n"
            "- Current closing price: {close}\n"
            "- Predicted next-day price: {pred}\n"
            "- User chose to: {ua} with ${amt}\n"
            "- The AI agent recommended: {ba}\n\n"
            "Write 3–4 short paragraphs:\n"
            "1) What RSI/SMA suggest. 2) Why the recommendation makes sense (or not). "
            "3) Possible next steps and momentum. 4) One actionable takeaway in **bold**.\n"
            "Keep tone neutral, non-sensual, and direct."
        )
        prompt = PromptTemplate(
            input_variables=["close", "pred", "ua", "ba", "amt"],
            template=template
        )
        # Runnable pipeline (no LLMChain)
        chain = prompt | llm
        return True, chain
    except Exception as e:
        logging.warning(f"LLM disabled (reason: {e})")
        return False, None

USE_LLM, LLM_CHAIN = _safe_llm_setup()

def explain_with_llm(close, pred, ua, ba, amt):
    if USE_LLM and LLM_CHAIN is not None:
        try:
            return LLM_CHAIN.invoke({"close": close, "pred": pred, "ua": ua, "ba": ba, "amt": amt})
        except Exception as e:
            logging.warning(f"LLM explanation failed at runtime: {e}")
    # Fallback: concise, safe explanation
    return (
        f"RSI/SMA context suggests trend awareness is useful. "
        f"Price now {close}, forecast {pred}. You chose to {ua} with ${amt}, "
        f"while the agent suggested {ba}. Consider momentum and risk buffer. "
        f"**Takeaway:** align size with conviction; avoid overreacting to tiny moves."
    )


# 1) TRADING ENVIRONMENT + AGENT + LSTM


class TradingEnv:
    def __init__(self, data, window_size=10, initial_balance=10000):
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.current_step = None
        self.balance = None
        self.shares_held = None
        self.total_profit = None

    def reset(self):
        if len(self.data) <= self.window_size:
            raise ValueError(f"Need more data ({len(self.data)}) than window_size ({self.window_size})")
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0.0
        return self._get_state()

    def _get_state(self):
        window = self.data.iloc[self.current_step - self.window_size : self.current_step]
        features = ['open', 'high', 'low', 'close', 'volume']
        state = [window[feat].iat[i] for i in range(self.window_size) for feat in features]
        state.extend([self.balance, self.shares_held])
        return np.array(state, dtype=np.float32)

    def step(self, action, amount=None):
        price = float(self.data['close'].iat[self.current_step])
        prev_total_assets = self.balance + self.shares_held * price

        if action == 1:  # BUY
            spend = self.balance if amount is None else min(self.balance, float(amount))
            shares = int(spend // price)
            if shares > 0:
                self.shares_held += shares
                self.balance -= shares * price

        elif action == 2 and self.shares_held > 0:  # SELL
            self.balance += self.shares_held * price
            self.total_profit += self.shares_held * price
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        new_price = float(self.data['close'].iat[self.current_step])
        current_total_assets = self.balance + self.shares_held * new_price
        reward = current_total_assets - prev_total_assets
        return self._get_state(), reward, done, {}

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("ReplayBuffer is empty; push() some transitions first.")
        if batch_size > n:
            idxs = [random.randrange(n) for _ in range(batch_size)]
            batch = [self.buffer[i] for i in idxs]
        else:
            batch = random.sample(self.buffer, batch_size)
        S, A, R, NS, D = zip(*batch)
        return np.vstack(S), A, R, np.vstack(NS), D
    def __len__(self):
        return len(self.buffer)
#Q network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.mem = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.e_end = epsilon_end
        self.e_decay = epsilon_decay
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.target.load_state_dict(self.policy.state_dict())
    def select_action(self, state):
        eps_thresh = self.e_end + (self.epsilon - self.e_end) * np.exp(-self.steps / self.e_decay)
        self.steps += 1
        if random.random() > eps_thresh:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return int(self.policy(s).max(1)[1])
        return random.randrange(self.policy.fc3.out_features)
    def learn(self, batch_size=64):
        if len(self.mem) < batch_size:
            return
        S, A, R, NS, D = self.mem.sample(batch_size)
        S = torch.tensor(S, dtype=torch.float32).to(self.device)
        A = torch.tensor(A, dtype=torch.long).unsqueeze(1).to(self.device)
        R = torch.tensor(R, dtype=torch.float32).unsqueeze(1).to(self.device)
        NS = torch.tensor(NS, dtype=torch.float32).to(self.device)
        D = torch.tensor(D, dtype=torch.float32).unsqueeze(1).to(self.device)
        curr_q = self.policy(S).gather(1, A)
        next_q = self.target(NS).max(1)[0].detach().unsqueeze(1)
        expected_q = R + self.gamma * next_q * (1.0 - D)
        loss = nn.MSELoss()(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
# LSTM for price prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(data, seq_len=10, epochs=20, batch_size=32, lr=1e-3):
    vals = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i])
        y.append(scaled[i])
    X, y = np.stack(X), np.stack(y)
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, scaler

def get_hint(df, step):
    rsi = df['rsi'].iat[step]
    sma = df['sma20'].iat[step]
    price = df['close'].iat[step]
    if rsi > 70: rsi_msg = f"RSI {rsi:.1f} → overbought"
    elif rsi < 30: rsi_msg = f"RSI {rsi:.1f} → oversold"
    else: rsi_msg = f"RSI {rsi:.1f} → neutral"
    trend_msg = f"Price ${price:.2f} > SMA (${sma:.2f}) → uptrend" if price > sma else \
                f"Price ${price:.2f} < SMA (${sma:.2f}) → downtrend"
    return f"{rsi_msg}. {trend_msg}."

# Forecast-driven recommendation helper
def rec_from_forecast(pred_next, curr_close, thr=0.005):
    if curr_close is None or np.isnan(curr_close):
        return 0
    if pred_next > curr_close * (1 + thr):
        return 1  # Buy
    if pred_next < curr_close * (1 - thr):
        return 2  # Sell
    return 0  # Hold

def compute_lstm_bias(df, scaler, lstm_model, seq_len=10, val_ratio=0.2):
    lstm_model.eval()
    prices = df["close"].astype(float).to_numpy()
    if len(prices) <= seq_len + 1:
        return 0.0
    scaled = scaler.transform(prices.reshape(-1, 1)).reshape(-1)
    Xs, ys = [], []
    for i in range(seq_len, len(scaled)):
        Xs.append(scaled[i - seq_len : i])
        ys.append(scaled[i])
    X = np.stack(Xs); y = np.array(ys)
    n = len(X)
    split = max(1, int(n * (1 - val_ratio)))
    if split >= n:
        return 0.0
    X_val = X[split:]; y_val = y[split:]
    with torch.no_grad():
        X_t = torch.tensor(X_val[:, :, None], dtype=torch.float32)
        yhat_scaled = lstm_model(X_t).squeeze(1).numpy()
    y_true = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).reshape(-1)
    bias = float(np.mean(y_true - y_pred))  # >0 => underpredicting on average
    return bias

# 3) TRAINING & STORAGE
trained_models = {}
sessions = {}

def train_all_tickers():
    global trained_models
    files = glob.glob("*.csv")
    raw_tickers = [f.replace(".csv", "") for f in files]
    print("Tickers found:", raw_tickers)
    for raw in raw_tickers:
        ticker = raw.upper()
        print(f"=== TRAINING: {raw} (storing as {ticker}) ===")
        df = pd.read_csv(f"{raw}.csv", parse_dates=["Date"], index_col="Date")

        # normalize columns
        df.columns = df.columns.str.lower().str.replace("/", "_").str.replace(" ", "_")
        for col in ["close_last", "open", "high", "low"]:
            if col in df.columns:
                df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)
        if "close_last" in df.columns:
            df.rename(columns={"close_last": "close"}, inplace=True)
        df["volume"] = df["volume"].replace(r"[,\s]", "", regex=True).fillna(0).astype(int)
        df.sort_index(inplace=True)

        # indicators
        df["sma20"] = ta.trend.sma_indicator(df["close"], 20)
        df["rsi"] = ta.momentum.rsi(df["close"], 14)
        try:
            df["macd"] = ta.trend.macd(df["close"])  
        except Exception:
            df["macd"] = 0.0
        df.bfill(inplace=True)

        # models
        lstm_model, scaler = train_lstm(df, seq_len=10, epochs=10)
        env = TradingEnv(df, window_size=10, initial_balance=10_000.0)
        agent = DQNAgent(state_dim=len(env.reset()), action_dim=3)

        # quick warmup
        for _ in range(50):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.mem.push(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
            agent.target.load_state_dict(agent.policy.state_dict())

        try:
            lstm_bias = compute_lstm_bias(df, scaler, lstm_model, seq_len=10, val_ratio=0.2)
        except Exception:
            lstm_bias = 0.0

        trained_models[ticker] = {
            "df": df,
            "scaler": scaler,
            "lstm": lstm_model,
            "agent": agent,
            "bias": lstm_bias,
        }
    print("\n*** All tickers trained! ***")

# 4) ROUTES
@app.route("/tickers", methods=["GET"])
def get_tickers():
    return jsonify({"tickers": list(trained_models.keys())})

@app.route("/start_session", methods=["POST"])
def start_session():
    data = request.get_json()
    ticker = (data.get("ticker") or "").upper()
    if ticker not in trained_models:
        return jsonify({"error": f"Ticker '{ticker}' not found."}), 400

    entry = trained_models[ticker]
    df = entry["df"]; scaler = entry["scaler"]
    lstm_model = entry["lstm"]; agent = entry["agent"]

    env = TradingEnv(df, window_size=10, initial_balance=10_000.0)
    initial_state = env.reset()

    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "env": env,
        "df": df,
        "scaler": scaler,
        "lstm": lstm_model,
        "agent": agent,
        "bias": entry.get("bias", 0.0),
    }
    return jsonify({"session_id": session_id, "initial_state": initial_state.tolist()})

@app.route("/state_and_hint", methods=["GET"])
def state_and_hint():
    session_id = request.args.get("session_id", "")
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id."}), 400
    sess = sessions[session_id]; env = sess["env"]; df = sess["df"]
    step = env.current_step
    hint_text = get_hint(df, step)
    window_size = env.window_size
    idx_start = step - window_size
    bars = []
    for i in range(idx_start, step):
        bars.append({
            "date": str(df.index[i]),
            "open": float(df["open"].iat[i]),
            "high": float(df["high"].iat[i]),
            "low": float(df["low"].iat[i]),
            "close": float(df["close"].iat[i]),
            "volume": int(df["volume"].iat[i])
        })
    return jsonify({
        "step": step,
        "balance": env.balance,
        "shares_held": env.shares_held,
        "hint": hint_text,
        "recent_bars": bars
    })

@app.route("/take_action", methods=["POST"])
def take_action():
    data = request.get_json()
    session_id = data.get("session_id", "")
    action = data.get("action", None)
    amount = data.get("amount", None)

    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id."}), 400
    if action not in [0, 1, 2]:
        return jsonify({"error": "Action must be 0 (Hold), 1 (Buy), or 2 (Sell)."}), 400

    sess = sessions[session_id]
    env = sess["env"]; df = sess["df"]
    scaler = sess["scaler"]; lstm_model = sess["lstm"]

    # 1) context before stepping
    curr_step = env.current_step
    curr_close = float(df["close"].iat[curr_step])

    # LSTM next-price prediction
    seq_all = scaler.transform(df["close"].values.reshape(-1, 1))
    window = seq_all[curr_step - 10 : curr_step].reshape(1, 10, 1)
    seq_tensor = torch.tensor(window, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = lstm_model(seq_tensor).item()
    predicted_price = float(scaler.inverse_transform([[pred_scaled]])[0][0])
    predicted_price += sess.get("bias", 0.0)

    # 2) agent recommendation from forecast
    recommended = rec_from_forecast(predicted_price, curr_close, thr=0.005)

    # 3) user correctness = did user follow agent?
    user_matched_agent = (action == recommended)

    # 4) execute user's action
    next_state, reward, done, _ = env.step(action, amount=amount)
    outcome_profit_positive = bool(reward >= 0)

    # 5) transparency on forecast trend
    if predicted_price > curr_close * 1.005:
        forecast_trend = "up"
    elif predicted_price < curr_close * 0.995:
        forecast_trend = "down"
    else:
        forecast_trend = "flat"

    # 6) LLM explanation (or fallback)
    close_str = f"${curr_close:.2f}"
    pred_str = f"${predicted_price:.2f}"
    ua_str = ["hold", "buy", "sell"][action]
    ba_str = ["hold", "buy", "sell"][recommended]
    amt_str = f"{float(amount):.2f}" if amount not in (None, "") else "0"
    llm_explanation = explain_with_llm(close_str, pred_str, ua_str, ba_str, amt_str)

    return jsonify({
        "next_state": next_state.tolist(),
        "reward": float(reward),
        "done": bool(done),
        "correct": bool(user_matched_agent),
        "user_matched_agent": bool(user_matched_agent),
        "outcome_profit_positive": bool(outcome_profit_positive),
        "forecast_trend": forecast_trend,
        "agent_recommendation": recommended,
        "predicted_price": predicted_price,
        "current_close": curr_close,
        "llm_explanation": llm_explanation
    })

# 5) MAIN
if __name__ == "__main__":
    train_all_tickers()
    app.run(host="0.0.0.0", port=5001, debug=False)
