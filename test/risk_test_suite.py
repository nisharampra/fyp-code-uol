#!/usr/bin/env python3
"""
risk_test_suite.py

What it does
------------
Given a CSV with prices (column 'close' or 'Close/Last'), this script:
1) Cleans and loads prices
2) Computes historical daily log-returns
3) Runs risk analytics:
   - VaR (95%/99%) and Expected Shortfall (ES)
   - Sharpe & Sortino ratios (annualized)
4) Monte Carlo simulations (two flavors):
   - GBM (parametric) using mu/sigma of log-returns
   - Bootstrap (non-parametric) by resampling historical returns
5) Saves:
   - risk_summary.csv (all key metrics)
   - equity_curves_gbm.png (100 sample paths)
   - equity_curves_bootstrap.png (100 sample paths)
   - returns_hist.png (distribution of returns)
   - final_pnl_hist.png (distribution of simulated final returns)

Usage
-----
python risk_test_suite.py --file apple.csv --days 252 --paths 2000 --seed 42

Notes
-----
- If your CSV uses '$' or commas, they will be cleaned automatically.
- If your column is 'Close/Last', that is also supported.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_prices(path):
    df = pd.read_csv(path)
    # Normalize column names for robustness
    cols = {c.lower().replace(" ", "").replace("/", "_"): c for c in df.columns}
    price_col = None
    for cand in ["close_last","close"]:
        if cand in cols:
            price_col = cols[cand]
            break
    if price_col is None:
        raise ValueError("Could not find a 'close' or 'Close/Last' column in CSV.")
    s = df[price_col].replace(r"[\$,]", "", regex=True).astype(float).to_numpy()
    return s

def sharpe_sortino(daily_returns, rf_daily=0.0):
    """Return annualized Sharpe and Sortino ratios."""
    mean = daily_returns.mean()
    vol = daily_returns.std(ddof=1)
    downside = daily_returns[daily_returns < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else np.nan

    sharpe_daily = (mean - rf_daily) / vol if vol > 0 else np.nan
    sortino_daily = (mean - rf_daily) / dd if dd and dd > 0 else np.nan

    # Annualize (252 trading days)
    sharpe_ann = sharpe_daily * np.sqrt(252) if not np.isnan(sharpe_daily) else np.nan
    sortino_ann = sortino_daily * np.sqrt(252) if not np.isnan(sortino_daily) else np.nan
    return float(sharpe_ann), float(sortino_ann)

def var_es(returns, p=0.95):
    """Value-at-Risk and Expected Shortfall for a given confidence p (e.g., 0.95)."""
    # For losses, we work on the left tail
    q = np.quantile(returns, 1 - p)
    es = returns[returns <= q].mean() if (returns <= q).any() else np.nan
    return float(q), float(es)

def max_drawdown(equity):
    peak = equity[0]
    dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = min(dd, (x - peak) / peak)
    return abs(dd)

def simulate_gbm(s0, mu, sigma, days, paths, rng):
    dt = 1.0
    z = rng.standard_normal((paths, days))
    rets = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    s = s0 * np.exp(np.cumsum(rets, axis=1))
    s = np.concatenate([np.full((paths,1), s0), s], axis=1)
    return s

def simulate_bootstrap(s0, hist_rets, days, paths, rng):
    idx = rng.integers(low=0, high=len(hist_rets), size=(paths, days))
    boot_rets = hist_rets[idx]
    s = s0 * np.exp(np.cumsum(boot_rets, axis=1))
    s = np.concatenate([np.full((paths,1), s0), s], axis=1)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="CSV with 'close' or 'Close/Last' prices")
    ap.add_argument("--days", type=int, default=252, help="Horizon in trading days")
    ap.add_argument("--paths", type=int, default=2000, help="Number of Monte Carlo paths")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    outdir = Path("risk_outputs"); outdir.mkdir(exist_ok=True, parents=True)

    prices = read_prices(args.file)
    if len(prices) < 60:
        raise SystemExit("Need at least 60 rows of prices for risk testing.")
    s0 = float(prices[-1])

    # Historical daily log returns
    logret = np.diff(np.log(prices))
    mu = float(logret.mean())
    sigma = float(logret.std(ddof=1))

    # Basic risk metrics on historical returns
    sharpe, sortino = sharpe_sortino(logret)
    var95, es95 = var_es(logret, 0.95)
    var99, es99 = var_es(logret, 0.99)

    rng = np.random.default_rng(args.seed)

    # Monte Carlo simulations
    gbm = simulate_gbm(s0, mu, sigma, args.days, args.paths, rng)
    boot = simulate_bootstrap(s0, logret, args.days, args.paths, rng)

    # Summaries
    def summarize(paths_eq):
        finals = (paths_eq[:,-1] - s0) / s0
        mdd = np.array([max_drawdown(eq) for eq in paths_eq])
        return finals, mdd

    finals_gbm, mdd_gbm = summarize(gbm)
    finals_boot, mdd_boot = summarize(boot)

    # Save summary CSV
    import pandas as pd
    summary = pd.DataFrame([{
        "S0": s0,
        "mu": mu, "sigma": sigma,
        "Sharpe_annual": sharpe, "Sortino_annual": sortino,
        "VaR95_daily": var95, "ES95_daily": es95,
        "VaR99_daily": var99, "ES99_daily": es99,
        "Final_Return_Median_GBM": float(np.median(finals_gbm)),
        "Final_Return_P05_GBM": float(np.quantile(finals_gbm, 0.05)),
        "Final_Return_P95_GBM": float(np.quantile(finals_gbm, 0.95)),
        "MaxDD_Median_GBM": float(np.median(mdd_gbm)),
        "MaxDD_P95_GBM": float(np.quantile(mdd_gbm, 0.95)),
        "Final_Return_Median_BOOT": float(np.median(finals_boot)),
        "Final_Return_P05_BOOT": float(np.quantile(finals_boot, 0.05)),
        "Final_Return_P95_BOOT": float(np.quantile(finals_boot, 0.95)),
        "MaxDD_Median_BOOT": float(np.median(mdd_boot)),
        "MaxDD_P95_BOOT": float(np.quantile(mdd_boot, 0.95)),
        "Paths": args.paths, "Days": args.days, "Seed": args.seed
    }])
    summary_path = outdir / "risk_summary.csv"
    summary.to_csv(summary_path, index=False)

    # --- Charts ---
    # 1) Equity curves (GBM)
    plt.figure()
    n_show = min(100, args.paths)
    idx = np.linspace(0, args.paths-1, n_show, dtype=int)
    for i in idx:
        plt.plot(gbm[i], alpha=0.25)
    plt.title("Monte Carlo Equity Curves (GBM)")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves_gbm.png")
    plt.close()

    # 2) Equity curves (Bootstrap)
    plt.figure()
    for i in idx:
        plt.plot(boot[i], alpha=0.25)
    plt.title("Monte Carlo Equity Curves (Bootstrap Returns)")
    plt.xlabel("Days")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(outdir / "equity_curves_bootstrap.png")
    plt.close()

    # 3) Histogram of historical returns
    plt.figure()
    plt.hist(logret, bins=50)
    plt.title("Historical Daily Log-Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outdir / "returns_hist.png")
    plt.close()

    # 4) Distribution of simulated final returns (GBM vs Bootstrap)
    plt.figure()
    plt.hist(finals_gbm, bins=60, alpha=0.6, label="GBM")
    plt.hist(finals_boot, bins=60, alpha=0.6, label="Bootstrap")
    plt.title(f"Distribution of Final {args.days}-Day Returns")
    plt.xlabel("Final Return (fraction of S0)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "final_pnl_hist.png")
    plt.close()

    print("Saved:")
    print(" -", summary_path.resolve())
    print(" -", (outdir / "equity_curves_gbm.png").resolve())
    print(" -", (outdir / "equity_curves_bootstrap.png").resolve())
    print(" -", (outdir / "returns_hist.png").resolve())
    print(" -", (outdir / "final_pnl_hist.png").resolve())

if __name__ == "__main__":
    main()