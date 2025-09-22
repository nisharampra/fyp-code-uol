
# model_testing_from_backend.py
import os, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import api

# ----------------------------
# Metrics helpers
# ----------------------------
def reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100.0)
    return mae, rmse, mape

def to_dir_labels(y):
    return (np.diff(y, prepend=y[0]) > 0).astype(int)

def cls_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, (tp, fp, fn, tn)

def directional_accuracy(y_true, y_pred):
    return float((to_dir_labels(y_true) == to_dir_labels(y_pred)).mean())

# ----------------------------
# LSTM eval using YOUR trained scaler + model
# ----------------------------
def eval_lstm_from_backend(entry, seq_len=10, val_ratio=0.2):
    """
    entry: api.trained_models[ticker] dict with keys: "df" (includes 'close'), "scaler", "lstm"
    Returns: y_pred, y_true (both in original price space)
    """
    df = entry["df"]
    scaler = entry["scaler"]
    model = entry["lstm"]
    model.eval()

    prices = df["close"].astype(float).to_numpy()
    if len(prices) <= seq_len + 1:
        raise ValueError(f"Not enough rows ({len(prices)}) for seq_len={seq_len}")

    # Scale with the SAME scaler used during training
    scaled = scaler.transform(prices.reshape(-1, 1)).reshape(-1)

    # Build (X, y) windows in scaled space
    Xs, ys = [], []
    for i in range(seq_len, len(scaled)):
        Xs.append(scaled[i - seq_len : i])
        ys.append(scaled[i])
    X = np.stack(Xs)       # (N, seq_len)
    y = np.array(ys)       # (N,)

    # Use last val_ratio as evaluation split
    n = len(X)
    split = max(1, int(n * (1 - val_ratio)))
    X_eval = X[split:]
    y_eval = y[split:]
    if len(X_eval) == 0:
        raise ValueError("Validation split ended up empty")

    import torch
    with torch.no_grad():
        X_t = torch.tensor(X_eval[:, :, None], dtype=torch.float32)
        yhat_scaled = model(X_t).squeeze(1).numpy()

    # Inverse-scale back to price space
    y_true = scaler.inverse_transform(y_eval.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).reshape(-1)
    return y_pred, y_true

# Visualization: single combined figure (3 panels)
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def combined_figure(y_true, y_pred, tp, fp, fn, tn, title_top, title_resid, title_conf):
    """Returns a matplotlib Figure with 3 panels:
       [0, :] line plot (Actual vs Predicted)
       [1, 0] residual histogram
       [1, 1] direction bars (TP/FP/FN/TN)
    """
    fig = plt.figure(figsize=(10, 7.5))
    # top (spans both columns)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.35)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(y_true, label="Actual")
    ax_top.plot(y_pred, label="Predicted")
    ax_top.set_title(title_top)
    ax_top.set_xlabel("Validation Steps")
    ax_top.set_ylabel("Price")
    ax_top.legend(loc="best")

    # bottom-left: residuals
    resid = np.asarray(y_true) - np.asarray(y_pred)
    ax_res = fig.add_subplot(gs[1, 0])
    ax_res.hist(resid, bins=30)
    ax_res.set_title(title_resid)
    ax_res.set_xlabel("Residual (Actual - Predicted)")
    ax_res.set_ylabel("Count")

    # bottom-right: TP/FP/FN/TN
    ax_conf = fig.add_subplot(gs[1, 1])
    labels = ["TP","FP","FN","TN"]
    values = [tp, fp, fn, tn]
    ax_conf.bar(labels, values)
    ax_conf.set_title(title_conf)
    ax_conf.set_xlabel("Category")
    ax_conf.set_ylabel("Count")

    return fig

# ----------------------------
# Runner
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="5,10,20", help="Comma-separated seq lens to test")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation split (last part)")
    ap.add_argument("--f1-threshold", type=float, default=0.50, help="Pass if F1 >= this")
    ap.add_argument("--csv-out", default="logs/model_testing_results.csv",
                    help="Output CSV path (kept in logs/ to avoid backend training on it)")
    ap.add_argument("--fig-dir", default="figs", help="Folder for combined figures")
    ap.add_argument("--report-pdf", default="", help="Optional: path to multi-page PDF report (e.g., report/model_testing_report.pdf)")
    ap.add_argument("--no-plots", action="store_true", help="Disable figure generation")
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    if os.path.dirname(args.csv_out):
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    if not args.no_plots:
        ensure_dir(args.fig_dir)
        if args.report_pdf:
            ensure_dir(os.path.dirname(args.report_pdf))

    # Train models via your backend (scans CSVs as your api.py already does)
    api.train_all_tickers()
    trained = api.trained_models  # dict {TICKER: {"df","scaler","lstm","agent"}}

    header = [
        "Test ID","Component","Ticker","SeqLen",
        "MAE","RMSE","MAPE","DirectionalAccuracy",
        "Precision","Recall","F1","Pass/Fail","Notes"
    ]

    # Markdown table (pretty)
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |"*len(header))

    rows = []
    pdf = None
    if not args.no_plots and args.report_pdf:
        pdf = PdfPages(args.report_pdf)

    if not trained:
        line = ["MT-1","LSTM Forecast","-","-","-","-","-","-","-","-","-","fail","No trained models found"]
        rows.append(line)
        print("| " + " | ".join(map(str, line)) + " |")
    else:
        for ticker, entry in trained.items():
            safe_tkr = ticker.replace(" ", "_")
            for sl in seq_lens:
                try:
                    y_pred, y_true = eval_lstm_from_backend(entry, seq_len=sl, val_ratio=args.val_ratio)
                    mae, rmse, mape = reg_metrics(y_true, y_pred)
                    da = directional_accuracy(y_true, y_pred)
                    p, r, f1, (tp, fp, fn, tn) = cls_metrics(to_dir_labels(y_true), to_dir_labels(y_pred))
                    pf = "pass" if f1 >= args.f1_threshold else "fail"

                    note = ""
                    row = [
                        "MT-1","LSTM Forecast",ticker,sl,
                        f"{mae:.4f}",f"{rmse:.4f}",f"{mape:.2f}",f"{da:.3f}",
                        f"{p:.3f}",f"{r:.3f}",f"{f1:.3f}",pf,note
                    ]
                    rows.append(row)
                    print("| " + " | ".join(map(str, row)) + " |")

                    # Combined figure
                    if not args.no_plots:
                        title_top  = f"{ticker}  (seq_len={sl}) — Actual vs Predicted"
                        title_res  = f"{ticker}  (seq_len={sl}) — Residuals"
                        title_conf = f"{ticker}  (seq_len={sl}) — Direction (TP/FP/FN/TN)"

                        fig = combined_figure(y_true, y_pred, tp, fp, fn, tn,
                                              title_top, title_res, title_conf)

                        # Save PNG
                        out_dir = os.path.join(args.fig_dir, safe_tkr)
                        ensure_dir(out_dir)
                        png_path = os.path.join(out_dir, f"seq{sl}_combined.png")
                        fig.savefig(png_path, dpi=150, bbox_inches="tight")

                        # Also add to multi-page PDF, if requested
                        if pdf is not None:
                            pdf.savefig(fig, bbox_inches="tight")

                        plt.close(fig)

                except Exception as e:
                    row = ["MT-1","LSTM Forecast",ticker,sl,"-","-","-","-","-","-","-","fail",f"{type(e).__name__}: {e}"]
                    rows.append(row)
                    print("| " + " | ".join(map(str, row)) + " |")

    # Save CSV file
    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # Inline CSV 
    print("\n=== INLINE CSV OUTPUT ===")
    print(",".join(header))
    for r in rows:
        print(",".join(map(str, r)))

    print(f"\nSaved: {args.csv_out}")
    if not args.no_plots:
        print(f"Combined PNGs saved under: {os.path.abspath(args.fig_dir)}")
    if pdf is not None:
        pdf.close()
        print(f"Multi-page PDF saved: {os.path.abspath(args.report_pdf)}")
    print("Done.")

if __name__ == "__main__":
    main()
