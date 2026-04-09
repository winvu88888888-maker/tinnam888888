"""
HOT-ZONE CEILING: If we ONLY play most frequent values per column,
what % of draws can we cover?

For each column, test:
  - Top-1, Top-2, Top-3... Top-N values (by frequency)
  - What % of draws have the actual value in that top-N set?
  - What's the JOINT probability (all 6 columns hit)?
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" HOT-ZONE CEILING ANALYSIS")
print(f" What % if we ONLY play the most frequent values per column?")
print(f"{'='*90}\n")

# ================================================================
# For each column, rank values by frequency
# ================================================================
for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    vc = Counter(vals)
    n = len(vals)

    print(f"  ── Column {pos+1} ──")
    ranked = vc.most_common()

    # Show coverage as we add values
    cum = 0
    line = ""
    for i, (v, c) in enumerate(ranked):
        cum += c
        pct = cum / n * 100
        if i < 15 or pct >= 100:
            line = f"    Top-{i+1:>2}: covers {pct:>6.2f}% (add {v:>2}, freq={c:>4}, {c/n*100:.1f}%)"
            print(line)
        if pct >= 99.9:
            break
    print()

# ================================================================
# JOINT analysis: if we pick Top-N per column, what % of draws
# have ALL 6 columns hit?
# ================================================================
print(f"\n{'='*90}")
print(f" JOINT COVERAGE: Top-N values per column → all 6 hit")
print(f"{'='*90}\n")

# Get ranked values per position
pos_ranked = []
for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    vc = Counter(vals)
    pos_ranked.append([v for v, _ in vc.most_common()])

# Test uniform K
print(f"  === UNIFORM K (same N for all columns) ===\n")
header = f"  {'N':<4}"
for p in range(6):
    header += f"  {'Col'+str(p+1):<9}"
header += f"  {'JOINT':<9}  {'Total_combos':<12}"
print(header)
print(f"  {'-'*85}")

for N in range(1, 25):
    col_cov = []
    for pos in range(6):
        top_set = set(pos_ranked[pos][:N])
        hits = sum(1 for sd in sorted_draws if sd[pos] in top_set)
        col_cov.append(hits / total * 100)

    # Joint: all 6 must be in their respective top-N
    joint = sum(1 for sd in sorted_draws
                if all(sd[p] in set(pos_ranked[p][:N]) for p in range(6)))
    joint_pct = joint / total * 100

    line = f"  {N:<4}"
    for c in col_cov:
        line += f"  {c:<9.2f}"
    combos = N**6
    line += f"  {joint_pct:<9.2f}  {combos:>12,}"
    print(line)

# ================================================================
# ADAPTIVE K: different N per column based on coverage target
# ================================================================
print(f"\n  === ADAPTIVE K (different N per column) ===\n")

for target in [50, 60, 70, 80, 85, 90, 95, 99]:
    ks = []
    col_covs = []
    for pos in range(6):
        vals = [sd[pos] for sd in sorted_draws]
        vc = Counter(vals)
        ranked = vc.most_common()
        cum = 0
        for i, (v, c) in enumerate(ranked):
            cum += c
            if cum / total * 100 >= target:
                ks.append(i+1)
                col_covs.append(cum / total * 100)
                break

    total_combos = 1
    for k in ks:
        total_combos *= k

    # Joint coverage with these Ks
    joint = 0
    for sd in sorted_draws:
        hit = True
        for pos in range(6):
            top_set = set(pos_ranked[pos][:ks[pos]])
            if sd[pos] not in top_set:
                hit = False
                break
        if hit:
            joint += 1
    joint_pct = joint / total * 100

    ks_str = " ".join(f"C{p+1}:{ks[p]:>2}" for p in range(6))
    print(f"  Target {target}%: {ks_str}")
    print(f"    Per-col: {' '.join(f'{c:.1f}%' for c in col_covs)}")
    print(f"    Joint: {joint_pct:.2f}% ({joint}/{total})")
    print(f"    Total combos: {total_combos:,}")
    print()

# ================================================================
# BEST FOCUSED ANALYSIS: What if we pick ONLY the core hot zone?
# ================================================================
print(f"\n{'='*90}")
print(f" CORE HOT ZONES — Values that cover 50%+ per column")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    vc = Counter(vals)
    ranked = vc.most_common()

    # Find minimum values to cover 50%
    cum = 0
    core_vals = []
    for v, c in ranked:
        cum += c
        core_vals.append(v)
        if cum / total * 100 >= 50:
            break

    core_str = ", ".join(f"{v}" for v in sorted(core_vals))
    print(f"  Col {pos+1}: [{core_str}] → {len(core_vals)} values cover "
          f"{cum/total*100:.1f}%")

# ================================================================
# TRANSITION-AWARE HOT ZONE: Given last draw, what top-N covers most?
# ================================================================
print(f"\n{'='*90}")
print(f" CONDITIONAL HOT ZONE — Given last draw's value at each position,")
print(f" how many values do we need to cover X% of possible next values?")
print(f"{'='*90}\n")

for pos in range(6):
    vals = [sd[pos] for sd in sorted_draws]
    n = len(vals)

    # Build transition
    trans = {}
    for i in range(n-1):
        cur = vals[i]
        nxt = vals[i+1]
        if cur not in trans:
            trans[cur] = Counter()
        trans[cur][nxt] += 1

    # For each source value, how many top destinations cover 50%, 80%?
    vc = Counter(vals)
    top_sources = vc.most_common(5)

    print(f"  Col {pos+1}:")
    for src, src_count in top_sources:
        if src not in trans: continue
        t = sum(trans[src].values())
        ranked_dest = trans[src].most_common()
        cum = 0
        for target_pct in [50, 80]:
            cum = 0
            for i, (v, c) in enumerate(ranked_dest):
                cum += c
                if cum / t * 100 >= target_pct:
                    vals_needed = i + 1
                    vals_list = [str(vv) for vv, _ in ranked_dest[:vals_needed]]
                    print(f"    After {src:>2} (n={t:>3}): {target_pct}% → "
                          f"need {vals_needed} values [{','.join(vals_list)}]")
                    break
    print()

print(f"{'='*90}")
print(f" BOTTOM LINE: The ceiling is determined by how concentrated")
print(f" each column's distribution is. Columns 1 & 6 are most")
print(f" concentrated, Columns 3 & 4 most spread out.")
print(f"{'='*90}")
