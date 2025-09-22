import requests, time, statistics

BASE = "http://127.0.0.1:5001"

def get_ticker():
    r = requests.get(f"{BASE}/tickers", timeout=30); r.raise_for_status()
    tks = r.json()["tickers"]; assert tks, "No tickers trained"
    return tks[0]

def start_session(t):
    r = requests.post(f"{BASE}/start_session", json={"ticker": t}, timeout=30); r.raise_for_status()
    return r.json()["session_id"]

def take_action(sid, action=0, explain=False):
    r = requests.post(f"{BASE}/take_action",
                      json={"session_id": sid, "action": action, "explain": explain},
                      timeout=30)
    r.raise_for_status()
    return r.json()

def state_and_hint(sid):
    r = requests.get(f"{BASE}/state_and_hint", params={"session_id": sid}, timeout=30)
    r.raise_for_status()
    return r.json()

def run_eval(steps=150):
    t = get_ticker()
    sid = start_session(t)
    preds, trues = [], []

    s0 = state_and_hint(sid)
    last_close = s0["recent_bars"][-1]["close"]

    for _ in range(steps):
        js = take_action(sid, action=0, explain=False)
        pred = js["predicted_price"]
        curr = js["current_close"]  

        s1 = state_and_hint(sid)
        next_close = s1["recent_bars"][-1]["close"]

        preds.append(pred)
        trues.append(next_close)

        last_close = next_close
        if js["done"]:
            # restart if env ended
            sid = start_session(t)
            s0 = state_and_hint(sid)
            last_close = s0["recent_bars"][-1]["close"]

    # metrics
    abs_err = [abs(p - a) for p, a in zip(preds, trues)]
    mae = statistics.mean(abs_err)

    dir_pred = [1 if (p2 - c) > 0 else -1 if (p2 - c) < 0 else 0
                for p2, c in zip(preds, trues[:-1] + [trues[-1]])]
    dir_true = [1 if (a - c) > 0 else -1 if (a - c) < 0 else 0
                for a, c in zip(trues[1:] + [trues[-1]], trues)]
    da = sum(int(x==y) for x,y in zip(dir_pred, dir_true)) / len(dir_pred) * 100.0

    print(f"PredQuality: MAE={mae:.4f}  DirectionalAcc={da:.1f}%  N={len(preds)}")

    assert da >= 55.0, "Directional accuracy below gate"
   

if __name__ == "__main__":
    run_eval(steps=120)
