#!/usr/bin/env python3
"""
Backtesting (CSV) — fuzzy columns + numeric cleaning + debug

- Works with one CSV (--file) or all CSVs in a folder (--folder).
- Cleans numbers like "$78.74", "1,234.56", "12.3M" -> numeric.
- Strategy: MA crossover (fast vs slow).
- Outputs: trades.csv, equity.csv, equity.png, summary.csv.


"""

from __future__ import annotations
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.strip().lower())

def _pick_col(df: pd.DataFrame, need: str, debug=False) -> str | None:
    originals = list(df.columns)
    norm_map = {_norm(c): c for c in originals}
    norms = list(norm_map.keys())

    easy = {
        "date":   {"date","datetime","timestamp","time"},
        "open":   {"open","o","openprice"},
        "high":   {"high","h"},
        "low":    {"low","l"},
        "volume": {"volume","vol","v"},
    }
    if need in easy:
        for k in easy[need]:
            if k in norm_map: return norm_map[k]

    if need == "close":
        close_like = [
            "close","adjclose","adjustedclose","closeprice","closing","price",
            "closelast","last","settle","settlement","closingprice","closeadj",
        ]
        for n in norms:
            if any(k in n for k in close_like):
                return norm_map[n]
        for raw in originals:
            r = raw.strip().lower()
            if ("close" in r) or ("last" in r) or ("settle" in r) or r in ("adj close","close/last","close last","closing price"):
                return raw

    # fallback contains
    if need in ("open","high","low","volume"):
        key = need
        for n in norms:
            if key in n: return norm_map[n]
    if need == "date":
        for n in norms:
            if "date" in n or "time" in n: return norm_map[n]

    if debug:
        print(f"[DEBUG] Could not find column for '{need}'. Columns present: {originals}")
    return None

_num_rx = re.compile(r"[^0-9eE+\-\.]")

def _parse_number(x: str) -> float | np.nan:
    """
    Convert strings like '$78.74', '1,234', '12.3M', '45.6K', '7.8B', '  99 % ' to floats.
    """
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan","none","null"}:
        return np.nan

    mult = 1.0
    # suffix multipliers for volume and some vendors
    if s[-1:].upper() in {"K","M","B"}:
        suf = s[-1:].upper()
        if suf == "K": mult = 1e3
        elif suf == "M": mult = 1e6
        elif suf == "B": mult = 1e9
        s = s[:-1]

    # remove currency symbols, commas, spaces, percent, etc.
    s = _num_rx.sub("", s)  

    try:
        return float(s) * mult
    except Exception:
        return np.nan

def _to_numeric_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(_parse_number)

# ---------- data loading & indicators ----------

def load_csv(path: Path, debug=False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if debug:
        print(f"[DEBUG] Reading: {path}")
        print(f"[DEBUG] Original columns: {list(df.columns)}")

    picks = {}
    for need in ("date","open","high","low","close","volume"):
        col = _pick_col(df, need, debug=debug)
        if col is not None:
            picks[col] = need

    df = df.rename(columns=picks)
    missing = [k for k in ("date","open","high","low","close","volume") if k not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns after normalization: {missing}\n"
            f"Detected/renamed: {picks}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Tip: run with --debug to see detection details."
        )

    # Clean types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open","high","low","close","volume"]:
        df[col] = _to_numeric_series(df[col])

    # Drop rows without a valid date or close price
    df = df.dropna(subset=["date","close"]).sort_values("date").reset_index(drop=True)

    if debug:
        print("[DEBUG] Normalized columns:", list(df.columns))
        print("[DEBUG] First 3 cleaned rows:\n", df.head(3).to_string(index=False))
    return df

def add_indicators(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(int(fast)).mean()
    df["ma_slow"] = df["close"].rolling(int(slow)).mean()
    return df

# ---------- strategy ----------

def generate_signals(df: pd.DataFrame) -> pd.Series:
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[df["ma_fast"] > df["ma_slow"]] = 1
    sig[df["ma_fast"] < df["ma_slow"]] = -1
    return sig

# ---------- backtest engine ----------

def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_cash: float = 10_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    allow_short: bool = False,
    position_size: float = 1.0,
):
    cash = initial_cash
    position = 0.0
    equity_curve, trades = [], []

    def px_with_slippage(px: float, is_buy: bool) -> float:
        adj = px * (slippage_bps / 10_000.0)
        return px + adj if is_buy else px - adj

    signals = signals.reindex(df.index).fillna(0).astype(int)

    for i, row in df.iterrows():
        date = row["date"]
        px = float(row["close"])

        desired = signals.iloc[i]
        if not allow_short and desired < 0:
            desired = 0

        equity_now = cash + position * px
        target_value = position_size * equity_now if desired != 0 else 0.0
        target_units = (target_value / px) * (1 if desired > 0 else -1 if desired < 0 else 0)

        if abs(target_units - position) > 1e-12:
            diff = target_units - position
            is_buy = diff > 0
            exec_px = px_with_slippage(px, is_buy=is_buy)
            trade_value = diff * exec_px
            fee = abs(trade_value) * (fee_bps / 10_000.0)

            cash -= trade_value
            cash -= fee
            position = target_units

            trades.append({
                "date": date,
                "action": "BUY" if is_buy else "SELL",
                "price": exec_px,
                "units": diff,
                "fee": fee,
                "cash_after": cash,
                "position_after": position,
            })

        equity_curve.append({"date": date, "equity": cash + position * px})

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

# ---------- metrics & plot ----------

def compute_metrics(equity_df: pd.DataFrame, initial_cash: float, freq: str = "trading"):
    if equity_df.empty:
        return {"total_return_pct": np.nan, "cagr_pct": np.nan, "max_drawdown_pct": np.nan, "sharpe": np.nan}

    periods_per_year = 252 if freq == "trading" else 365
    n = len(equity_df)
    years = max(1e-9, n / periods_per_year)

    eq = equity_df["equity"].astype(float).values
    final_equity = float(eq[-1])
    total_return = (final_equity - initial_cash) / initial_cash

    rets = np.diff(eq) / eq[:-1]
    sharpe = np.nan
    if rets.size > 1 and np.nanstd(rets) > 0:
        sharpe = np.nanmean(rets) / np.nanstd(rets) * np.sqrt(periods_per_year)

    roll_max = np.maximum.accumulate(eq)
    dd = (eq - roll_max) / roll_max
    max_dd = float(np.min(dd)) if dd.size else np.nan

    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0 if final_equity > 0 else np.nan

    return {
        "total_return_pct": total_return * 100.0,
        "cagr_pct": (cagr * 100.0) if cagr == cagr else np.nan,
        "max_drawdown_pct": (max_dd * 100.0) if max_dd == max_dd else np.nan,
        "sharpe": sharpe if sharpe == sharpe else np.nan,
    }

def save_equity_plot(equity_df: pd.DataFrame, out_png: Path, title: str):
    plt.figure()
    plt.plot(equity_df["date"], equity_df["equity"], label="Equity")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ---------- runner ----------

def run_one(
    csv_path: Path,
    out_dir: Path,
    fast: int,
    slow: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    freq: str,
    allow_short: bool,
    position_size: float,
    debug: bool,
):
    df = load_csv(csv_path, debug=debug)
    df = add_indicators(df, fast=fast, slow=slow)
    sig = generate_signals(df)

    equity_df, trades_df = backtest(
        df, sig,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        allow_short=allow_short,
        position_size=position_size,
    )

    symbol = csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_path = out_dir / f"{symbol}_equity.csv"
    trades_path = out_dir / f"{symbol}_trades.csv"
    plot_path = out_dir / f"{symbol}_equity.png"

    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    save_equity_plot(equity_df, plot_path, title=f"Equity Curve - {symbol}")

    m = compute_metrics(equity_df, initial_cash=initial_cash, freq=freq)
    return {"symbol": symbol, **m}

def main():
    ap = argparse.ArgumentParser(description="CSV backtesting (MA crossover, fuzzy columns, numeric cleaning, debug).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a single CSV")
    g.add_argument("--folder", type=str, help="Folder with one or more CSV files")

    ap.add_argument("--out", type=str, required=True, help="Output directory (e.g., runs_baselines)")
    ap.add_argument("--fast", type=int, default=10, help="Fast MA window")
    ap.add_argument("--slow", type=int, default=50, help="Slow MA window")
    ap.add_argument("--initial_cash", type=float, default=10000.0)
    ap.add_argument("--fee_bps", type=float, default=10.0, help="Transaction fee (basis points)")
    ap.add_argument("--slippage_bps", type=float, default=5.0, help="Slippage (basis points per side)")
    ap.add_argument("--freq", type=str, choices=["trading", "calendar"], default="trading")
    ap.add_argument("--allow_short", action="store_true", help="Enable shorting")
    ap.add_argument("--position_size", type=float, default=1.0, help="0..1 fraction of equity per entry")
    ap.add_argument("--debug", action="store_true", help="Print detection/cleaning samples")

    args = ap.parse_args()
    out_dir = Path(args.out)

    results = []
    if args.file:
        res = run_one(
            Path(args.file), out_dir,
            fast=args.fast, slow=args.slow,
            initial_cash=args.initial_cash,
            fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
            freq=args.freq, allow_short=args.allow_short,
            position_size=args.position_size, debug=args.debug,
        )
        results.append(res)
    else:
        folder = Path(args.folder)
        csvs = sorted([p for p in folder.glob("*.csv")])
        if not csvs:
            raise SystemExit(f"No CSV files found in {folder.resolve()}")
        for p in csvs:
            res = run_one(
                p, out_dir,
                fast=args.fast, slow=args.slow,
                initial_cash=args.initial_cash,
                fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                freq=args.freq, allow_short=args.allow_short,
                position_size=args.position_size, debug=args.debug,
            )
            results.append(res)

    if results:
        summary = pd.DataFrame(results)
        summary_path = out_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print("\nSummary:")
        print(summary.to_string(index=False))
        print(f"\nSaved: {summary_path.resolve()}")

if __name__ == "__main__":
    main()
