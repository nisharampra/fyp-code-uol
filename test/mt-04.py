# mt-04_residual_diagnostics.py
# MT-4: LSTM Residual Diagnostics (mean≈0, low autocorrelation via Ljung–Box)

import os, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

import api  

# ---------- helpers ----------
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def eval_lstm_from_backend(entry, seq_len=20, val_ratio=0.2):
    """
    Reuse the backend's trained scaler + model to get validation predictions.
    Returns y_true, y_pred (both in original price space).
    """
    df      = entry["df"]
    scaler  = entry["scaler"]
    model   = entry["lstm"]
    model.eval()

    prices = df["close"].astype(float).to_numpy()
    if len(prices) <= seq_len + 1:
        raise ValueError(f"Not enough rows ({len(prices)}) for seq_len={seq_len}")

    scaled = scaler.transform(prices.reshape(-1, 1)).reshape(-1)

    Xs, ys = [], []
    for i in range(seq_len, len(scaled)):
        Xs.append(scaled[i - seq_len : i])
        ys.append(scaled[i])
    X = np.stack(Xs)
    y = np.array(ys)

    n = len(X)
    split = max(1, int(n * (1 - val_ratio)))
    X_eval = X[split:]
    y_eval = y[split:]
    if len(X_eval) == 0:
        raise ValueError("Validation split ended up empty")

    with torch.no_grad():
        X_t = torch.tensor(X_eval[:, :, None], dtype=torch.float32)
        yhat_scaled = model(X_t).squeeze(1).numpy()

    y_true = scaler.inverse_transform(y_eval.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).reshape(-1)
    return y_true, y_pred

def combined_residual_figure(ticker, resid, ljung_pvals, out_png):
    """One figure (2x2): residual series, histogram, ACF, Q-Q plot."""
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)

    # Residual over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid)
    ax1.set_title(f"{ticker} — Residuals over Validation Steps")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Residual")

    # Histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(resid, bins=30)
    ax2.set_title("Residual Histogram"); ax2.set_xlabel("Residual"); ax2.set_ylabel("Count")

    # ACF
    ax3 = fig.add_subplot(gs[1, 0])
    plot_acf(resid, lags=min(40, len(resid)//2), ax=ax3)
    ax3.set_title("Residual ACF")

    # Q-Q Plot
    ax4 = fig.add_subplot(gs[1, 1])
    sm.qqplot(resid, line="s", ax=ax4)
    ax4.set_title("Residual Q–Q Plot")

    # Footer with Ljung–Box p-values
    pv_str = ", ".join([f"lag{lag} p={p:.3f}" for lag, p in ljung_pvals.items()])
    fig.text(0.01, 0.02, f"Ljung–Box p-values: {pv_str}", fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=20, help="Sequence length for eval")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (last chunk)")
    ap.add_argument("--lags", default="5,10,20", help="Comma-separated Ljung–Box lags")
    ap.add_argument("--csv-out", default="logs/mt04_residuals.csv")
    ap.add_argument("--fig-dir", default="mt04_figs")
    ap.add_argument("--report-pdf", default="", help="Optional multi-page PDF path")
    ap.add_argument("--mean-thresh-mult", type=float, default=0.01,
                    help="Resid mean |mean| < mean_thresh_mult * std(y_true) to pass '≈0'")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",") if x.strip()]
    ensure_dir(os.path.dirname(args.csv_out)) if os.path.dirname(args.csv_out) else None
    ensure_dir(args.fig_dir)
    pdf = PdfPages(args.report_pdfs) if False else None  # keep var for clarity
    if args.report_pdf:
        pdf = PdfPages(args.report_pdf)

    # Train via backend (won’t start Flask)
    api.train_all_tickers()
    trained = api.trained_models

    header = ["Test ID","Model","Ticker","SeqLen","MeanResidual","StdResidual"] + \
             [f"LjungBox_p(lag={l})" for l in lags] + ["Pass/Fail","Notes"]
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |"*len(header))

    rows = []

    for ticker, entry in trained.items():
        try:
            y_true, y_pred = eval_lstm_from_backend(entry, seq_len=args.seq_len, val_ratio=args.val_ratio)
            resid = y_true - y_pred

            mean_resid = float(np.mean(resid))
            std_resid  = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

            # Ljung-Box p-values at requested lags
            ljung = acorr_ljungbox(resid, lags=lags, return_df=True)
            pvals = {lag: float(ljung.loc[lag, "lb_pvalue"]) for lag in lags}

            # Pass rule:
            # 1) |mean residual| < (mean_thresh_mult * std of y_true)  → “≈ 0”
            # 2) All Ljung–Box p-values > 0.05                       → “low autocorr”
            mean_thresh = args.mean_thresh_mult * float(np.std(y_true, ddof=1))
            mean_ok = abs(mean_resid) < (mean_thresh if mean_thresh > 0 else 1e-12)
            ac_ok   = all(p > 0.05 for p in pvals.values())
            passed  = (mean_ok and ac_ok)
            pf = "pass" if passed else "fail"
            note = f"thr={mean_thresh:.6g}; valN={len(resid)}"

            # Figure (single combined per ticker)
            out_dir = os.path.join(args.fig_dir, ticker.replace(" ", "_"))
            ensure_dir(out_dir)
            fig_path = os.path.join(out_dir, f"seq{args.seq_len}_residual_diagnostics.png")
            combined_residual_figure(ticker, resid, pvals, fig_path)
            if pdf is not None:
                img = plt.imread(fig_path)
                fig = plt.figure(figsize=(11, 8))
                plt.imshow(img); plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

            row = ["MT-4","LSTM",ticker,args.seq_len,f"{mean_resid:.6g}",f"{std_resid:.6g}"] + \
                  [f"{pvals[l]:.3f}" for l in lags] + [pf, note]

        except Exception as e:
            row = ["MT-4","LSTM",ticker,args.seq_len,"-","-"] + ["-"]*len(lags) + ["fail",f"{type(e).__name__}: {e}"]

        rows.append(row)
        print("| " + " | ".join(map(str, row)) + " |")

    # Save CSV
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # Inline CSV
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
