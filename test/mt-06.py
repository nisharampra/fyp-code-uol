import csv, collections, sys

CSV_IN = "logs/mt01_seq_sensitivity.csv"
THRESH = 0.02  # pass if (max(F1) - min(F1)) ≤ 0.02

# ticker -> {seq_len: f1}
f1s = collections.defaultdict(dict)

with open(CSV_IN, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["Test ID"] != "MT-1":
            continue
        t = row["Ticker"]
        sl = int(row["SeqLen"])
        try:
            f1 = float(row["F1"])
        except:
            continue
        f1s[t][sl] = f1

print("Both,Sequence Length Sensitivity,Repeat MT-1/MT-2 for seq_len ∈ {5, 10, 20},Minimal performance drop")
for ticker, d in f1s.items():
    f5  = d.get(5,  None)
    f10 = d.get(10, None)
    f20 = d.get(20, None)
    fs = [x for x in [f5, f10, f20] if x is not None]
    if not fs:
        status = "✘ (no F1s)"
        detail = ""
    else:
        delta = max(fs) - min(fs)
        status = "✔" if delta <= THRESH else "✘"
        detail = f"F1 len5={f5:.3f}, len10={f10:.3f}, len20={f20:.3f} (Δ={delta:.3f})"
    print(f"LSTM ({ticker}),Yes,Done,{status} {detail}")
