#!/usr/bin/env python3
"""
Bot Test Suite — one file for everything

Subcommands
-----------
latency       Measure per-request latency (avg / p90 / p95 / p99) and save CSV
stress        Threaded stress/load test (N users × K steps) → throughput
resource      Monitor CPU/RAM (system or a specific PID) each second
risk          Monte Carlo risk on a single price CSV → VaR/ES/Drawdown
write-locust  Writes a ready-to-run locustfile.py next to this script

Quick Start
----------
# 1) Start your API (choose one)
python api.py
# or:
gunicorn api:app -k gevent -w 4 -b 127.0.0.1:5001

# 2) Latency (saves runs_perf/latency_run.csv)
python bot_test_suite.py latency --base http://127.0.0.1:5001 --steps 100

# 3) Stress (no extra deps)
python bot_test_suite.py stress --base http://127.0.0.1:5001 --users 25 --steps 40

# 4) Resource (system-wide or by PID)
python bot_test_suite.py resource
python bot_test_suite.py resource --pid 12345

# 5) Risk (Monte Carlo on CSV)
python bot_test_suite.py risk --file apple.csv --paths 2000 --days 252

# 6) Optional: Locust (writes locustfile.py)
python bot_test_suite.py write-locust
locust -f locustfile.py
"""

import argparse
import csv
import os
import statistics
import sys
import time
import random
from pathlib import Path

try:
    import requests
except Exception:
    requests = None

try:
    import psutil
except Exception:
    psutil = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None


# -----------------------
# Helpers
# -----------------------
def _require(module, name):
    if module is None:
        raise SystemExit(
            f"Missing dependency for '{name}'. Please install it first.\n"
            f"Try: pip install {name}"
        )

def _pctile(xs, p):
    if not xs:
        return float('nan')
    _require(np, "numpy")
    return float(np.percentile(xs, p))

def _mk_runs_dir():
    outdir = Path("runs_perf")
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir


# -----------------------
# latency subcommand
# -----------------------
def cmd_latency(args):
    _require(requests, "requests")
    base = args.base.rstrip("/")

    # 1) get tickers
    t0 = time.time()
    r = requests.get(f"{base}/tickers", timeout=10)
    r.raise_for_status()
    tickers = r.json().get("tickers", [])
    if not tickers:
        raise SystemExit("No tickers found. Ensure CSVs loaded & training finished.")
    ticker = args.ticker.upper() if args.ticker else tickers[0]
    t1 = time.time()

    # 2) start session
    t2 = time.time()
    r = requests.post(f"{base}/start_session", json={"ticker": ticker}, timeout=30)
    r.raise_for_status()
    session_id = r.json()["session_id"]
    t3 = time.time()

    print(f"[OK] Using ticker: {ticker} | session_id={session_id}")
    print(f"Latency: /tickers={(t1 - t0)*1000:.2f}ms | /start_session={(t3 - t2)*1000:.2f}ms")

    lats_state, lats_action, lats_total = [], [], []
    outdir = _mk_runs_dir()
    csv_path = outdir / "latency_run.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "lat_state_ms", "lat_action_ms", "lat_total_ms", "action", "reward", "done"])
        for i in range(args.steps):
            t_state0 = time.time()
            r = requests.get(f"{base}/state_and_hint", params={"session_id": session_id}, timeout=10)
            r.raise_for_status()
            t_state1 = time.time()

            action = random.choice([0, 1, 2])
            payload = {"session_id": session_id, "action": action}
            if args.amount > 0:
                payload["amount"] = args.amount

            t_act0 = time.time()
            r2 = requests.post(f"{base}/take_action", json=payload, timeout=20)
            r2.raise_for_status()
            j = r2.json()
            t_act1 = time.time()

            lat_state = (t_state1 - t_state0) * 1000
            lat_action = (t_act1 - t_act0) * 1000
            lat_total  = (t_act1 - t_state0) * 1000

            lats_state.append(lat_state)
            lats_action.append(lat_action)
            lats_total.append(lat_total)

            w.writerow([
                i,
                f"{lat_state:.2f}",
                f"{lat_action:.2f}",
                f"{lat_total:.2f}",
                action,
                j.get("reward"),
                j.get("done")
            ])

            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{args.steps}] state={lat_state:.1f}ms action={lat_action:.1f}ms total={lat_total:.1f}ms")

    def summarize(name, xs):
        if not xs:
            return
        print(f"\n{name}:")
        print(f"  avg={statistics.mean(xs):.2f}ms  p90={_pctile(xs,90):.2f}ms  p95={_pctile(xs,95):.2f}ms  p99={_pctile(xs,99):.2f}ms  max={max(xs):.2f}ms")

    summarize("STATE   latency", lats_state)
    summarize("ACTION  latency", lats_action)
    summarize("TOTAL   latency", lats_total)
    print(f"\nSaved detailed log to: {csv_path.resolve()}")


# -----------------------
# stress subcommand
# -----------------------
def _stress_user(base, steps):
    r = requests.get(f"{base}/tickers", timeout=10)
    t = (r.json().get("tickers") or ["APPLE"])[0]
    r = requests.post(f"{base}/start_session", json={"ticker": t}, timeout=30)
    s = r.json()["session_id"]
    ok = 0
    for _ in range(steps):
        requests.get(f"{base}/state_and_hint", params={"session_id": s}, timeout=10)
        a = random.choice([0, 1, 2])
        rr = requests.post(f"{base}/take_action", json={"session_id": s, "action": a}, timeout=20)
        if rr.status_code == 200:
            ok += 1
    return ok

def cmd_stress(args):
    _require(requests, "requests")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    base = args.base.rstrip("/")

    t0 = time.time()
    futures = []
    with ThreadPoolExecutor(max_workers=args.users) as ex:
        for _ in range(args.users):
            futures.append(ex.submit(_stress_user, base, args.steps))
        total_ok = 0
        for fut in as_completed(futures):
            total_ok += fut.result()
    dt = time.time() - t0
    total_calls = args.users * args.steps
    print(f"Completed {total_calls} POST /take_action calls in {dt:.2f}s ({total_calls/dt:.1f} req/s).")
    print(f"Successful POSTs: {total_ok} / {total_calls}")


# -----------------------
# resource subcommand
# -----------------------
def cmd_resource(args):
    _require(psutil, "psutil")
    proc = None
    if args.pid:
        try:
            proc = psutil.Process(args.pid)
            print(f"Monitoring PID {args.pid}: {' '.join(proc.cmdline())}")
        except psutil.NoSuchProcess:
            print(f"PID {args.pid} not found; falling back to system-wide metrics.")
            proc = None

    print("Press Ctrl+C to stop.")
    while True:
        try:
            if proc:
                cpu = proc.cpu_percent(interval=1.0)
                mem = proc.memory_info().rss / (1024 * 1024)
                print(f"PROC  cpu={cpu:5.1f}% | rss={mem:7.1f} MB")
            else:
                cpu = psutil.cpu_percent(interval=1.0)
                mem = psutil.virtual_memory().percent
                print(f"SYSTEM cpu={cpu:5.1f}% | ram%={mem:5.1f}%")
        except KeyboardInterrupt:
            break


# -----------------------
# risk subcommand (Monte Carlo)
# -----------------------
def _read_prices(path):
    _require(pd, "pandas")
    df = pd.read_csv(path)
    # flexible column name handling
    cols = {c.lower().replace(" ", "").replace("/", "_"): c for c in df.columns}
    price_col = None
    for cand in ["close_last", "close"]:
        if cand in cols:
            price_col = cols[cand]
            break
    if price_col is None:
        raise ValueError("Could not find a 'close' or 'Close/Last' column in CSV.")
    s = df[price_col].replace(r"[\$,]", "", regex=True).astype(float).to_numpy()
    return s

def _max_drawdown(equity_curve):
    peak = equity_curve[0]
    dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = min(dd, (x - peak) / peak)
    return abs(dd)

def cmd_risk(args):
    _require(np, "numpy")
    prices = _read_prices(args.file)
    if len(prices) < 50:
        raise SystemExit("Not enough history in CSV. Need at least 50 rows.")
    logret = np.diff(np.log(prices))
    mu = float(np.mean(logret))
    sigma = float(np.std(logret, ddof=1))

    S0 = float(prices[-1])
    D = args.days
    rng = np.random.default_rng(args.seed)
    Z = rng.standard_normal((args.paths, D))
    dt = 1.0
    rets = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    S = S0 * np.exp(np.cumsum(rets, axis=1))
    S = np.concatenate([np.full((args.paths, 1), S0), S], axis=1)

    eq = S
    final_rets = (eq[:, -1] - S0) / S0
    var95 = float(np.percentile(final_rets, 5))
    var99 = float(np.percentile(final_rets, 1))
    es95 = float(final_rets[final_rets <= var95].mean())
    es99 = float(final_rets[final_rets <= var99].mean())
    dds = np.array([_max_drawdown(e) for e in eq])
    dd50 = float(np.percentile(dds, 50))
    dd95 = float(np.percentile(dds, 95))

    outdir = _mk_runs_dir()
    out = outdir / "risk_summary.csv"
    _require(pd, "pandas")
    pd.DataFrame([{
        "S0": S0, "mu": mu, "sigma": sigma,
        "days": D, "paths": args.paths,
        "VaR95": var95, "VaR99": var99, "ES95": es95, "ES99": es99,
        "MedianMaxDD": dd50, "P95MaxDD": dd95
    }]).to_csv(out, index=False)

    print("Monte Carlo Risk Summary")
    print("------------------------")
    print(f"S0={S0:.2f}  mu={mu:.6f}  sigma={sigma:.6f}  days={D}  paths={args.paths}")
    print(f"VaR95={var95:.3f}  VaR99={var99:.3f}  ES95={es95:.3f}  ES99={es99:.3f}")
    print(f"Max Drawdown median={dd50:.3f}  p95={dd95:.3f}")
    print(f"Saved CSV: {out.resolve()}")


# -----------------------
# write-locust subcommand
# -----------------------
_LOCUST_TEMPLATE = """\
from locust import HttpUser, task, between
import random

class BotUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        r = self.client.get("/tickers")
        self.ticker = (r.json().get("tickers") or ["APPLE"])[0]
        r = self.client.post("/start_session", json={"ticker": self.ticker})
        self.session_id = r.json()["session_id"]

    @task(3)
    def state_and_hint(self):
        self.client.get("/state_and_hint", params={"session_id": self.session_id})

    @task(2)
    def take_action(self):
        action = random.choice([0,1,2])
        self.client.post("/take_action", json={"session_id": self.session_id, "action": action})
"""

def cmd_write_locust(_args):
    path = Path("locustfile.py")
    path.write_text(_LOCUST_TEMPLATE, encoding="utf-8")
    print(f"Wrote {path.resolve()}")
    print("Run: pip install locust && locust -f locustfile.py")
    print("Open http://localhost:8089, set Users/Spawn rate/Host, then Start.")


# -----------------------
# Main CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="One-file test suite for your Financial Advisor Bot API")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # latency
    ap_lat = sub.add_parser("latency", help="Measure per-request latency (avg/p90/p95/p99) and save CSV")
    ap_lat.add_argument("--base", type=str, default="http://127.0.0.1:5001", help="Base URL of the API")
    ap_lat.add_argument("--steps", type=int, default=100, help="Number of (state+action) iterations")
    ap_lat.add_argument("--ticker", type=str, default="", help="Specific ticker (default: first from /tickers)")
    ap_lat.add_argument("--amount", type=float, default=0.0, help="Fixed trade amount for BUY/SELL (0 uses balance)")
    ap_lat.set_defaults(func=cmd_latency)

    # stress
    ap_str = sub.add_parser("stress", help="Threaded stress/load test (users × steps) → throughput")
    ap_str.add_argument("--base", type=str, default="http://127.0.0.1:5001", help="Base URL of the API")
    ap_str.add_argument("--users", type=int, default=20, help="Number of concurrent users")
    ap_str.add_argument("--steps", type=int, default=50, help="Steps per user")
    ap_str.set_defaults(func=cmd_stress)

    # resource
    ap_res = sub.add_parser("resource", help="Monitor CPU/RAM (system or PID) each second")
    ap_res.add_argument("--pid", type=int, default=0, help="PID of process (gunicorn/python). 0 = system-wide")
    ap_res.set_defaults(func=cmd_resource)

    # risk
    ap_risk = sub.add_parser("risk", help="Monte Carlo risk stats (VaR/ES/Drawdown) from a price CSV")
    ap_risk.add_argument("--file", required=True, help="CSV file with 'close' or 'Close/Last' column")
    ap_risk.add_argument("--paths", type=int, default=2000, help="Number of GBM paths")
    ap_risk.add_argument("--days", type=int, default=252, help="Horizon in days")
    ap_risk.add_argument("--seed", type=int, default=42, help="Random seed")
    ap_risk.set_defaults(func=cmd_risk)

    # write-locust
    ap_loc = sub.add_parser("write-locust", help="Write locustfile.py for Locust load testing")
    ap_loc.set_defaults(func=cmd_write_locust)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
