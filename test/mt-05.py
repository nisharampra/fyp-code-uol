# mt-05.py
# MT-5: DQN Action Sequence Robustness (Behavior Cloning on Teacher States)
# Pass criterion: MatchRate >= 90% on a fixed synthetic test set.


import os, csv, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from api import TradingEnv, DQNAgent  

# ---------- Synthetic data ----------
def make_synth_df(n=800, seed=2025, amp=0.8, noise=0.03):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 10 + 0.02 * t + amp * np.sin(2 * np.pi * t / 50.0)
    close = base + rng.normal(0.0, noise, size=n)
    open_  = close + rng.normal(0.0, noise/2, size=n)
    high   = np.maximum(open_, close) + np.abs(rng.normal(0, noise/2, size=n))
    low    = np.minimum(open_, close) - np.abs(rng.normal(0, noise/2, size=n))
    vol    = (rng.integers(8_000, 12_000, size=n)).astype(int)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})

def baseline_action(prices, idx, thr=0.003):
    if idx == 0: return 0
    prev, curr = prices[idx-1], prices[idx]
    if curr > prev * (1 + thr): return 1  # buy
    if curr < prev * (1 - thr): return 2  # sell
    return 0  # hold

# ---------- Teacher datasets (no student stepping) ----------
def build_teacher_dataset(env: TradingEnv, thr=0.003):
    closes = env.data["close"].reset_index(drop=True).to_numpy()
    states, labels = [], []
    env.reset()
    done = False
    while not done:
        idx = env.current_step
        a_teacher = baseline_action(closes, idx, thr=thr)
        # capture CURRENT state (teacher state)
        states.append(env._get_state().copy())
        labels.append(a_teacher)
        # advance using TEACHER action to keep distribution consistent
        _, _, done, _ = env.step(a_teacher)
    X = torch.tensor(np.stack(states), dtype=torch.float32)  # [N, state_dim]
    y = torch.tensor(np.array(labels), dtype=torch.long)     # [N]
    return X, y

# ---------- Normalization ----------
def zfit(train_x: torch.Tensor):
    mean = train_x.mean(dim=0)
    std  = train_x.std(dim=0, unbiased=False).clamp_min(1e-8)
    def apply(x: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std
    return apply, mean, std

# ---------- Behavior cloning ----------
def behavior_clone(agent: DQNAgent, X, y, epochs=35, batch_size=256, lr=3e-3):
    device = agent.device
    policy = agent.policy.to(device)
    policy.train()

    # class weights to mitigate label imbalance (Hold >> Buy/Sell)
    with torch.no_grad():
        counts = torch.bincount(y, minlength=3).float()
        inv = 1.0 / torch.clamp(counts, min=1.0)
        weights = (inv / inv.sum()) * 3.0
    ce = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    n = len(X)
    for _ in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            sl = idx[i:i+batch_size]
            xb = X[sl].to(device)
            yb = y[sl].to(device)
            logits = policy(xb)         # [B, 3]
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

# ---------- Plot ----------
def plot_actions_teacher(y_true: torch.Tensor, y_pred: torch.Tensor, out_path: str):
    idxs = np.arange(len(y_true))
    yt = y_true.cpu().numpy()
    yp = y_pred.cpu().numpy()
    plt.figure()
    plt.plot(idxs, yt, label="Baseline", linewidth=1.5)
    plt.plot(idxs, yp, label="DQN (cloned)", linestyle="--", linewidth=1.0)
    plt.yticks([0, 1, 2], ["Hold", "Buy", "Sell"])
    plt.xlabel("Step"); plt.ylabel("Action")
    plt.title("Action Sequences — Baseline vs DQN (teacher states)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=600)
    ap.add_argument("--test-n",  type=int, default=200)
    ap.add_argument("--window",  type=int, default=10)
    ap.add_argument("--thr",     type=float, default=0.005)   
    ap.add_argument("--epochs",  type=int, default=50)       
    ap.add_argument("--csv-out", default="logs/mt05_action_robustness.csv")
    ap.add_argument("--viz-dir", default="mt05_figs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    # Repro
    random.seed(123); np.random.seed(123); torch.manual_seed(123)

    # Paths
    if os.path.dirname(args.csv_out):
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)

    # Synthetic data + split
    df = make_synth_df(n=args.train_n + args.test_n, seed=2025, amp=0.8, noise=0.03)
    df_train = df.iloc[:args.train_n].copy()
    df_test  = df.iloc[args.train_n:].copy()

    # Envs
    env_train = TradingEnv(df_train, window_size=args.window, initial_balance=10_000)
    env_test  = TradingEnv(df_test,  window_size=args.window, initial_balance=10_000)

    # Determine state_dim from env
    state_dim = len(env_train.reset())

    # Teacher datasets (states + labels); step ONLY with teacher
    Xtr_raw, ytr = build_teacher_dataset(env_train, thr=args.thr)
    Xte_raw, yte = build_teacher_dataset(env_test,  thr=args.thr)

    # Mask account features: zero-out the last 2 dims (balance, shares_held)
    Xtr_raw[:, -2:] = 0.0
    Xte_raw[:, -2:] = 0.0

    # Z-normalize using TRAIN stats only
    zn, _, _ = zfit(Xtr_raw)
    Xtr = zn(Xtr_raw)
    Xte = zn(Xte_raw)

    # Agent + behavior cloning
    agent = DQNAgent(state_dim=state_dim, action_dim=3, lr=1e-3)
    behavior_clone(agent, Xtr, ytr, epochs=args.epochs, batch_size=256, lr=3e-3)

    # Evaluate OFF-POLICY on teacher states (no env stepping by student)
    agent.policy.eval()
    with torch.no_grad():
        logits = agent.policy(Xte.to(agent.device))
        preds  = logits.argmax(dim=1).cpu()

    matches = int((preds == yte).sum())
    total   = int(yte.numel())
    mr = matches / max(1, total)
    status = "pass" if mr >= 0.90 else "fail"

    # Report
    header = ["Test ID","Model","Dataset","MatchRate(%)","Pass/Fail","Notes"]
    row = ["MT-5","DQN","Synthetic",f"{mr*100:.1f}",status,
           f"Matches={matches}/{total}; window={args.window}; epochs={args.epochs}; thr={args.thr}"]

    print("| " + " | ".join(header) + " |")
    print("|" + " --- |"*len(header))
    print("| " + " | ".join(map(str,row)) + " |")

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerow(row)

    if not args.no_plots:
        plot_actions_teacher(yte, preds, os.path.join(args.viz_dir, "actions_teacher_vs_dqn.png"))

    print("\n=== INLINE CSV OUTPUT ===")
    print(",".join(header))
    print(",".join(map(str,row)))
    print(f"\nSaved: {args.csv_out}")
    if not args.no_plots:
        print(f"Figures at: {os.path.abspath(args.viz_dir)}")
    print("Done.")

if __name__ == "__main__":
    main()
