# mt-03_lstm_learning_curves.py
# MT-3: LSTM Learning Curve Monitoring (Train vs Val Loss over epochs)

import os, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import api 

# ----------------------------
# utils
# ----------------------------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def build_sequences(series: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(series)):
        Xs.append(series[i - seq_len: i])
        ys.append(series[i])
    X = np.stack(Xs) if Xs else np.empty((0, seq_len))
    y = np.array(ys) if ys else np.empty((0,))
    return X, y

def count_consecutive_increases(arr):
    """Return the longest run length of strictly increasing consecutive elements."""
    if len(arr) < 2: return 0
    longest = cur = 0
    for i in range(1, len(arr)):
        if arr[i] > arr[i-1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest

def plot_learning_curve(ticker, losses_train, losses_val, out_png):
    plt.figure(figsize=(8,5))
    plt.plot(losses_train, label="Train Loss")
    plt.plot(losses_val, label="Validation Loss")
    plt.title(f"{ticker} — Learning Curve (Train vs Val)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

# ----------------------------
# one-ticker training (fresh model just for curves)
# ----------------------------
def train_lstm_with_logging(df, scaler, seq_len=10, epochs=20, batch_size=32, lr=1e-3, val_ratio=0.2, device=None):
    """
    Uses 'close' from df; scales with provided scaler (from backend) to match preprocessing.
    Returns: train_losses, val_losses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prices = df["close"].astype(float).to_numpy()
    scaled = scaler.transform(prices.reshape(-1, 1)).reshape(-1)

    X, y = build_sequences(scaled, seq_len)
    if len(X) == 0:
        raise ValueError("Not enough samples to build sequences")

    n = len(X)
    split = max(1, int(n * (1 - val_ratio)))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    X_train_t = torch.tensor(X_train[:, :, None], dtype=torch.float32)
    y_train_t = torch.tensor(y_train[:, None], dtype=torch.float32)
    X_val_t   = torch.tensor(X_val[:, :, None], dtype=torch.float32)
    y_val_t   = torch.tensor(y_val[:, None], dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = api.LSTMModel(input_size=1, hidden_size=50, num_layers=2).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        train_loss = total / len(train_ds)

        # val
        model.eval()
        with torch.no_grad():
            xb, yb = X_val_t.to(device), y_val_t.to(device)
            pred = model(xb)
            vloss = float(loss_fn(pred, yb).item())

        train_losses.append(train_loss)
        val_losses.append(vloss)

    return train_losses, val_losses

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=20, help="Sequence length for curves (report-friendly)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--csv-out", default="logs/mt03_learning_curves.csv")
    ap.add_argument("--fig-dir", default="mt03_figs")
    ap.add_argument("--report-pdf", default="", help="Optional multi-page PDF path")
    ap.add_argument("--tickers", default="", help="Comma-separated subset, e.g. 'APPLE,TESLA'")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.csv_out)) if os.path.dirname(args.csv_out) else None
    ensure_dir(args.fig_dir)
    pdf = PdfPages(args.report_pdf) if args.report_pdf else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    api.train_all_tickers()
    trained = api.trained_models

    tickers_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or list(trained.keys())

    header = ["Test ID","Model","Ticker","SeqLen","Epochs","LongestValIncreaseRun","Pass/Fail","Notes"]
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |"*len(header))

    rows = []

    for ticker in tickers_list:
        if ticker not in trained:
            row = ["MT-3","LSTM",ticker,args.seq_len,args.epochs,"-","fail","Ticker not in trained_models"]
            rows.append(row)
            print("| " + " | ".join(map(str,row)) + " |")
            continue

        entry = trained[ticker]
        df = entry["df"]
        scaler = entry["scaler"]

        try:
            tr_losses, va_losses = train_lstm_with_logging(
                df, scaler,
                seq_len=args.seq_len, epochs=args.epochs,
                batch_size=args.batch_size, val_ratio=args.val_ratio,
                device=device
            )

            longest_run = count_consecutive_increases(va_losses)
            passed = longest_run < 3
            pf = "pass" if passed else "fail"
            note = f"Final ValLoss={va_losses[-1]:.6f}; TrainLoss={tr_losses[-1]:.6f}"

            # figure
            png_dir = os.path.join(args.fig_dir, ticker.replace(" ", "_"))
            ensure_dir(png_dir)
            png_path = os.path.join(png_dir, f"seq{args.seq_len}_learning_curve.png")
            plot_learning_curve(ticker, tr_losses, va_losses, png_path)
            if pdf is not None:
                fig = plt.figure(figsize=(8,5))
                plt.plot(tr_losses, label="Train Loss")
                plt.plot(va_losses, label="Validation Loss")
                plt.title(f"{ticker} — Learning Curve (SeqLen={args.seq_len})")
                plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend(); plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            row = ["MT-3","LSTM",ticker,args.seq_len,args.epochs,longest_run,pf,note]

        except Exception as e:
            row = ["MT-3","LSTM",ticker,args.seq_len,args.epochs,"-","fail",f"{type(e).__name__}: {e}"]

        rows.append(row)
        print("| " + " | ".join(map(str,row)) + " |")

    # write CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    print("\n=== INLINE CSV OUTPUT ===")
    print(",".join(header))
    for r in rows:
        print(",".join(map(str, r)))

    print(f"\nSaved summary: {args.csv_out}")
    print(f"Figures in: {os.path.abspath(args.fig_dir)}")
    if pdf is not None:
        pdf.close()
        print(f"Multi-page PDF: {os.path.abspath(args.report_pdf)}")
    print("Done.")

if __name__ == "__main__":
    main()
