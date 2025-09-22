# baselines.py — SMA(50/200) + RSI(14) long/flat baselines with fees, sizing, and OOS alignment
import argparse, os, re, math
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- IO / cleaning ----------
def detect_columns(df: pd.DataFrame, close_hint=None, date_hint=None):
    orig_cols = list(df.columns)
    lower = {c: re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns}
    date_col, close_col = None, None

    if date_hint:
        for c in df.columns:
            if lower[c] == date_hint.lower() or date_hint.lower() in lower[c]:
                date_col = c; break
    if date_col is None:
        for c in df.columns:
            if lower[c] in ["date","timestamp","time"]:
                date_col = c; break

    if close_hint:
        for c in df.columns:
            if lower[c] == close_hint.lower() or close_hint.lower() in lower[c]:
                close_col = c; break
    if close_col is None:
        syns = {"close","close_last","adj close","adj_close","adjusted close","adjusted_close","price","last","closing price"}
        for c in df.columns:
            lc = lower[c]; base = re.sub(r"[^a-z ]","", lc)
            if lc in syns or base in syns or lc.startswith("close"):
                close_col = c; break
    if close_col is None:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num: close_col = num[-1]
    return date_col, close_col, orig_cols

def load_close_series(csv_path: str, close_hint=None, date_hint=None) -> pd.Series:
    df = pd.read_csv(csv_path)
    dcol, ccol, orig = detect_columns(df, close_hint, date_hint)
    if ccol is None:
        raise ValueError(f"Could not find a close/price column. Available: {orig}")

    # parse date index (if any)
    if dcol:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=[dcol]).set_index(dcol).sort_index()
    else:
        df = df.sort_index()

    # CLEAN the price column: strip $, commas, spaces -> numeric
    s = (df[ccol].astype(str)
                 .str.replace(r"[\$,]", "", regex=True)
                 .str.replace(r"\s+", "", regex=True)
                 .replace("", np.nan))
    s = pd.to_numeric(s, errors="coerce").ffill().bfill()

    if s.isna().all():
        raise ValueError("Price column could not be parsed (all NaN) after cleaning.")

    s.name = "close"
    return s

def perf_metrics(equity: pd.Series, freq_per_year=252) -> Dict[str, float]:
    if len(equity) < 2:
        return {"Total Return": 0.0, "CAGR": 0.0, "Max Drawdown": 0.0, "Sharpe (naive)": 0.0, "Bars": int(len(equity))}
    rets = equity.pct_change(fill_method=None).dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    yrs = max(len(equity) / freq_per_year, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = (equity / equity.cummax() - 1.0).min()
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * math.sqrt(freq_per_year) if len(rets) else 0.0
    return {"Total Return": float(total_return), "CAGR": float(cagr), "Max Drawdown": float(dd), "Sharpe (naive)": float(sharpe), "Bars": int(len(equity))}


def read_equity_index_for_alignment(path: Optional[str]) -> Optional[pd.DatetimeIndex]:
    if not path: return None
    if not os.path.exists(path): 
        print(f"[WARN] align-to file not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # Try parse first column as datetime index
        cand = df.columns[0]
        idx = pd.to_datetime(df[cand], errors="coerce")
        idx = idx[~idx.isna()]
        if len(idx) == 0:
            return None
        return pd.DatetimeIndex(idx).sort_values()
    except Exception as e:
        print("[WARN] failed to parse align-to index:", e)
        return None

# ---------- Indicators ----------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ---------- Simulation from positions ----------
def simulate_from_positions(close: pd.Series, pos: pd.Series, fee: float = 0.0005, position_size: float = 1.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    pos: 1 for long, 0 for flat; trades triggered on pos changes; fees applied per entry/exit.
    Fully invests 'position_size' fraction of cash when going long.
    """
    pos = pos.reindex(close.index).fillna(0).astype(int)
    cash = 1.0
    shares = 0.0
    in_pos = False
    equity = []
    trades = []
    for i in range(len(close)-1):
        price = close.iloc[i]
        want_long = (pos.iloc[i] == 1)

        # exit
        if in_pos and not want_long:
            cash = shares * price * (1 - fee)
            trades.append({"date": close.index[i], "side": "SELL", "price": float(price)})
            shares = 0.0; in_pos = False

        # enter
        if (not in_pos) and want_long:
            alloc = cash * position_size
            if alloc > 0:
                shares = (alloc * (1 - fee)) / price
                cash -= alloc
                trades.append({"date": close.index[i], "side": "BUY", "price": float(price)})
                in_pos = True

        equity.append(cash + shares * price)

    # final liquidation
    last_price = close.iloc[-1]
    if in_pos:
        cash = shares * last_price * (1 - fee)
        trades.append({"date": close.index[-1], "side": "SELL", "price": float(last_price)})
        shares = 0.0; in_pos = False
    equity.append(cash + shares * last_price)

    eq = pd.Series(equity, index=close.index, name="equity")
    trades_df = pd.DataFrame(trades)
    return eq, trades_df

# ---------- Metrics ----------
def perf_metrics(equity: pd.Series, freq_per_year=252) -> Dict[str, float]:
    if len(equity) < 2:
        return {"Total Return": 0.0, "CAGR": 0.0, "Max Drawdown": 0.0, "Sharpe (naive)": 0.0, "Bars": int(len(equity))}
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    yrs = max(len(equity) / freq_per_year, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = (equity / equity.cummax() - 1.0).min()
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * math.sqrt(freq_per_year) if len(rets) else 0.0
    return {"Total Return": float(total_return), "CAGR": float(cagr), "Max Drawdown": float(dd), "Sharpe (naive)": float(sharpe), "Bars": int(len(equity))}

# ---------- Strategies ----------
def positions_sma(close: pd.Series, short_n: int = 50, long_n: int = 200) -> pd.Series:
    s = sma(close, short_n)
    l = sma(close, long_n)
    pos = (s > l).astype(int)
    pos.name = "pos_sma"
    return pos

def positions_rsi(close: pd.Series, n: int = 14, enter_above: float = 50.0, exit_below: float = 50.0) -> pd.Series:
    r = rsi(close, n)
    pos = pd.Series(0, index=close.index, dtype=int)
    in_pos = False
    for i in range(len(close)):
        val = r.iloc[i]
        if not np.isfinite(val):
            pos.iloc[i] = int(in_pos)
            continue
        if (not in_pos) and val > enter_above:
            in_pos = True
        elif in_pos and val < exit_below:
            in_pos = False
        pos.iloc[i] = int(in_pos)
    pos.name = "pos_rsi"
    return pos

# ---------- Runner ----------
def run_and_save(name: str, close: pd.Series, pos: pd.Series, fee: float, position_size: float, outdir: str, plot: bool):
    eq, trades = simulate_from_positions(close, pos, fee=fee, position_size=position_size)
    m = perf_metrics(eq)

    os.makedirs(outdir, exist_ok=True)
    eq.to_csv(os.path.join(outdir, f"{name}_equity.csv"), index=True)
    trades.to_csv(os.path.join(outdir, f"{name}_trades.csv"), index=False)
    pd.DataFrame([{"Strategy": name, **m}]).to_csv(os.path.join(outdir, f"{name}_summary.csv"), index=False)

    print(f"\n== {name} ==")
    for k, v in m.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    if plot:
        plt.figure(figsize=(9,5))
        plt.plot(eq.index, eq/eq.iloc[0], label=f"{name} equity (norm=1)")
        plt.title(f"{name} – Equity")
        plt.xlabel("Time"); plt.ylabel("Growth (x)"); plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(os.path.join(outdir, f"{name}_equity.png"), bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Baseline backtests: SMA(50/200) and RSI(14) long/flat.")
    ap.add_argument("--csv", required=True, type=str, help="Path to OHLCV CSV")
    ap.add_argument("--close-col", type=str, default=None)
    ap.add_argument("--date-col", type=str, default=None)
    ap.add_argument("--fee", type=float, default=0.0005, help="Per-side fee")
    ap.add_argument("--position-size", type=float, default=1.0, help="Fraction of equity when long")
    # SMA
    ap.add_argument("--sma-short", type=int, default=50)
    ap.add_argument("--sma-long", type=int, default=200)
    # RSI
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--rsi-enter", type=float, default=50.0)
    ap.add_argument("--rsi-exit", type=float, default=50.0)
    # Align to OOS dates
    ap.add_argument("--align-to", type=str, default=None, help="Path to an equity CSV (e.g., apple_tuning_combined_equity.csv) to align dates")
    ap.add_argument("--outdir", type=str, default="runs_baselines")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    close = load_close_series(args.csv, args.close_col, args.date_col)

    # Align to OOS dates if provided (uses index of the given equity CSV)
    align_idx = read_equity_index_for_alignment(args.align_to)
    if align_idx is not None and isinstance(close.index, pd.DatetimeIndex):
        close = close.loc[close.index.intersection(align_idx)].copy()
        close = close.sort_index()

    # SMA baseline
    pos_sma = positions_sma(close, short_n=args.sma_short, long_n=args.sma_long)
    run_and_save(f"sma_{args.sma_short}_{args.sma_long}", close, pos_sma, args.fee, args.position_size, args.outdir, plot=not args.no_plot)

    # RSI baseline
    pos_rsi = positions_rsi(close, n=args.rsi_len, enter_above=args.rsi_enter, exit_below=args.rsi_exit)
    run_and_save(f"rsi_{args.rsi_len}_{int(args.rsi_enter)}_{int(args.rsi_exit)}", close, pos_rsi, args.fee, args.position_size, args.outdir, plot=not args.no_plot)

if __name__ == "__main__":
    main()
