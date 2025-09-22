
# MT-02: DQN Trading Policy Effectiveness + Visualizations
import os, csv, argparse, math, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import api  
# --------- metrics helpers ----------
def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / np.where(peaks == 0, 1, peaks)
    return float(dd.min() * 100.0) if len(dd) else 0.0

def drawdown_curve(equity_curve: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peaks) / np.where(peaks == 0, 1, peaks)
    return dd * 100.0  # %

def profit_factor(pnls):
    gains = sum(x for x in pnls if x > 0)
    losses = -sum(x for x in pnls if x < 0)
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)

def pct(x): return float(x * 100.0)

# --------- simulate one episode with the agent ----------
def run_episode(env, agent, df):
    """
    Returns:
      equity_curve: np.ndarray of total assets per step
      trade_pnls: list of closed-trade PnLs ($)
      trade_rets: list of closed-trade returns (% of entry cost, as fraction)
      net_profit: float ($)
      n_trades: int
      win_rate: float in [0,1]
      closes_eval: np.ndarray of close prices aligned to equity_curve
      actions_eval: list[int] actions taken per step (0,1,2)
      buy_idxs, sell_idxs: indices where trades executed (for plotting)
    """
    # greedy eval
    if hasattr(agent, "epsilon"): agent.epsilon = 0.0
    if hasattr(agent, "steps"): agent.steps = 10**9

    state = env.reset()
    equity, actions, closes = [], [], []

    trade_pnls, trade_rets = [], []
    in_position = False
    entry_cost = 0.0
    entry_shares = 0
    buy_idxs, sell_idxs = [], []

    done = False
    while not done:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
            q = agent.policy(s)
            action = int(q.argmax(1).item())

        price_before = float(df["close"].iat[env.current_step])
        prev_balance, prev_shares = env.balance, env.shares_held

        next_state, reward, done, _ = env.step(action)

        price_after = float(df["close"].iat[env.current_step])
        total_assets = env.balance + env.shares_held * price_after

        actions.append(action)
        equity.append(total_assets)
        closes.append(price_after)

        # trade bookkeeping
        if action == 1:  # BUY
            if env.shares_held > prev_shares:
                # entry
                in_position = True
                cash_spent = prev_balance - env.balance
                entry_cost += cash_spent
                entry_shares += (env.shares_held - prev_shares)
                buy_idxs.append(len(equity)-1)

        if action == 2:  # SELL all
            if prev_shares > 0 and env.shares_held == 0:
                proceeds = env.balance - prev_balance
                pnl = proceeds - entry_cost
                trade_pnls.append(float(pnl))
                trade_rets.append(float((pnl / entry_cost) if entry_cost > 0 else 0.0))
                in_position, entry_cost, entry_shares = False, 0.0, 0
                sell_idxs.append(len(equity)-1)

        state = next_state

    # Close open position notionally at end
    if env.shares_held > 0 and entry_cost > 0:
        last_price = float(df["close"].iat[env.current_step])
        proceeds = env.shares_held * last_price
        pnl = proceeds - entry_cost
        trade_pnls.append(float(pnl))
        trade_rets.append(float((pnl / entry_cost)))
        sell_idxs.append(len(equity)-1)

    equity_curve = np.array(equity, dtype=float)
    closes_eval = np.array(closes, dtype=float)
    net_profit = float(equity_curve[-1] - equity_curve[0]) if len(equity_curve) >= 2 else 0.0
    n_trades = len(trade_pnls)
    wins = sum(1 for x in trade_pnls if x > 0)
    win_rate = (wins / n_trades) if n_trades else 0.0
    return (equity_curve, trade_pnls, trade_rets, net_profit, n_trades, win_rate,
            closes_eval, actions, buy_idxs, sell_idxs)

# --------- baseline (buy & hold on same span) ----------
def buy_and_hold_equity(closes_eval: np.ndarray, initial_balance=10_000.0):
    if len(closes_eval) == 0:
        return np.array([initial_balance], float)
    p0 = closes_eval[0] if closes_eval[0] != 0 else 1e-8
    bh = initial_balance * (closes_eval / p0)
    return bh

def buy_and_hold_summary(closes_eval: np.ndarray, initial_balance=10_000.0):
    bh_curve = buy_and_hold_equity(closes_eval, initial_balance)
    ret = (bh_curve[-1] - bh_curve[0]) / (bh_curve[0] if bh_curve[0] != 0 else 1e-8)
    return bh_curve, pct(ret), float(bh_curve[-1] - bh_curve[0])

# --------- visualization (one combined figure per ticker) ----------
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def combined_figure(ticker, equity_curve, bh_curve, trade_rets, win_rate, pf, net_profit, dd_curve):
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.2], hspace=0.35, wspace=0.30)

    # Top: equity vs buy&hold
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_curve, label="DQN Equity")
    ax1.plot(bh_curve, label="Buy & Hold", linestyle="--")
    ax1.set_title(f"{ticker} — Equity vs Buy & Hold")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="best")

    # Bottom-left: drawdown curve
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(dd_curve)
    ax2.set_title("Drawdown (%)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Drawdown (%)")

    # Bottom-right: trade returns + win/loss bars
    ax3 = fig.add_subplot(gs[1, 1])
    if len(trade_rets):
        ax3.hist(np.array(trade_rets)*100.0, bins=30)
    ax3.set_title("Trade Return Histogram (%)")
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Count")

    # Small inset bars for W/L
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax3, width="40%", height="40%", loc="upper right")
    wins = sum(1 for r in trade_rets if r > 0)
    losses = len(trade_rets) - wins
    ax_in.bar(["Wins","Losses"], [wins, losses])
    ax_in.set_title("W/L", fontsize=9)

    # Footer annotation (key numbers)
    fig.text(0.01, 0.02,
             f"Win Rate: {win_rate*100:.1f}%   Profit Factor: {pf:.2f}   Net Profit: ${net_profit:.2f}",
             fontsize=10)

    return fig

# --------- main runner ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default="logs/model_testing_results_dqn.csv")
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--initial-balance", type=float, default=10_000.0)
    ap.add_argument("--fig-dir", default="dqn_figs", help="Folder to save combined figures")
    ap.add_argument("--report-pdf", default="", help="Optional multi-page PDF path")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    api.train_all_tickers()
    trained = api.trained_models

    header = [
        "Test ID","Model","Ticker",
        "WinRate(%)","ProfitFactor","AvgTradeRet(%)","MaxDD(%)","NetProfit($)",
        "BaselineBH(%)","BaselineBH($)",
        "Pass/Fail","Notes"
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |"*len(header))

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    if not args.no_plots:
        ensure_dir(args.fig_dir)
        pdf = PdfPages(args.report_pdf) if args.report_pdf else None
    else:
        pdf = None

    rows = []

    for ticker, entry in trained.items():
        df    = entry["df"]
        agent = entry["agent"]
        env   = api.TradingEnv(df, window_size=args.window_size, initial_balance=args.initial_balance)

        try:
            (equity_curve, trade_pnls, trade_rets, net_profit, n_trades, win_rate,
             closes_eval, actions, buy_idxs, sell_idxs) = run_episode(env, agent, df)

            mdd = max_drawdown(equity_curve)
            pf  = profit_factor(trade_pnls) if n_trades else 0.0
            avg_ret = np.mean(trade_rets)*100.0 if n_trades else 0.0

            bh_curve, bh_ret_pct, bh_profit = buy_and_hold_summary(closes_eval, args.initial_balance)

            pass_fail = "pass" if (net_profit >= 0.0 and win_rate >= 0.50) else "fail"
            note = f"Trades={n_trades}"

            row = [
                "MT-2","DQN",ticker,
                f"{win_rate*100.0:.1f}", f"{pf:.2f}", f"{avg_ret:.2f}", f"{mdd:.2f}", f"{net_profit:.2f}",
                f"{bh_ret_pct:.2f}", f"{bh_profit:.2f}",
                pass_fail, note
            ]
        except Exception as e:
            row = ["MT-2","DQN",ticker,"-","-","-","-","-","-","-","fail",f"{type(e).__name__}: {e}"]
            equity_curve = np.array([])
            bh_curve = np.array([])
            pf = net_profit = win_rate = 0.0
            trade_rets = []
            mdd = 0.0

        rows.append(row)
        print("| " + " | ".join(map(str, row)) + " |")

        # Combined figure
        if not args.no_plots and len(equity_curve) > 1:
            dd_curve = drawdown_curve(equity_curve)
            fig = combined_figure(ticker, equity_curve, bh_curve, trade_rets, float(row[3]), float(row[4]), float(row[7]), dd_curve)
            out_dir = os.path.join(args.fig_dir, ticker.replace(" ", "_"))
            ensure_dir(out_dir)
            fig.savefig(os.path.join(out_dir, "combined.png"), dpi=150, bbox_inches="tight")
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Write CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    print("\n=== INLINE CSV OUTPUT ===")
    print(",".join(header))
    for r in rows:
        print(",".join(map(str, r)))

    print(f"\nSaved: {args.csv_out}")
    if not args.no_plots:
        print(f"Figures at: {os.path.abspath(args.fig_dir)}")
        if pdf is not None:
            pdf.close()
            print(f"Multi-page PDF: {os.path.abspath(args.report_pdf)}")
    print("Done.")

if __name__ == "__main__":
    main()
