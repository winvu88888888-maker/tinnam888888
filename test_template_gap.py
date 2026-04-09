"""
TEMPLATE-PRIMARY + GAP-FILTER — FINAL EXPLOIT
================================================
KEY INSIGHT from backtest data:
- Template matching alone = BEST for ≥3/6 (37% at N=20)
- Gap scoring alone = BEST for ≥5/6 (0.156% at N=20)
- Hybrid v1 = worst of both worlds (too concentrated)

NEW STRATEGY: Use template matching as PRIMARY generator,
then FILTER/RERANK using gap scores and co-occurrence.
This preserves template's diversity while boosting gap-flagged combos.

Also: use WIDER candidate pool — template gives 30+ unique combos,
then gap score reranks them.
"""
import sys, time, os, warnings, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from math import comb as mcomb
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
all_records = get_mega645_all()
total = len(data)
MAX_NUM = 45
PICK = 6
sorted_draws = [sorted(d[:6]) for d in data]

try:
    from datetime import datetime, timedelta
    dates = [r.get('draw_date', '') for r in all_records]
    days_of_week = []
    for d in dates:
        try:
            dt = datetime.strptime(str(d)[:10], '%Y-%m-%d')
            days_of_week.append(dt.strftime('%A'))
        except:
            days_of_week.append('')
except:
    days_of_week = [''] * total

ANOMALOUS = {8, 3, 24, 1, 38, 33, 42, 34, 12, 19}
AUTOCORR_NEG = {8, 24}
SUNDAY_BOOST = {5}

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" TEMPLATE-PRIMARY + GAP-FILTER EXPLOIT")
print(f"{'='*90}\n")

# ================================================================
# TEMPLATE GENERATOR (expanded — produce 200+ diverse combos)
# ================================================================
def template_generate(history, N_pool=200):
    """Generate large pool of combos using template matching."""
    n = len(history)
    last = set(history[-1])
    last2 = set(history[-2]) if n >= 2 else set()
    last3 = set(history[-3]) if n >= 3 else set()
    curr_sum = sum(history[-1])
    curr_range = max(history[-1]) - min(history[-1])

    # Multi-similarity: exact overlap + sum/range proximity
    similarities = []
    for i in range(max(0, n-1000), n-2):
        h = set(history[i])
        sim = len(h & last) * 5
        sim += len(h & last2) * 2
        sim += len(h & last3) * 1
        sim += max(0, 6 - abs(sum(history[i]) - curr_sum) / 8)
        sim += max(0, 4 - abs((max(history[i])-min(history[i])) - curr_range) / 5)
        # Bonus: same even/odd composition
        eo1 = sum(1 for x in history[-1] if x%2==0)
        eo2 = sum(1 for x in history[i] if x%2==0)
        if eo1 == eo2: sim += 2
        similarities.append((sim, i))

    similarities.sort(key=lambda x: -x[0])

    # Take next draws after top-50 similar states → these ARE our combos
    combos = []
    for _, idx in similarities[:80]:
        if idx + 1 < n:
            combo = tuple(sorted(history[idx+1][:6]))
            combos.append(combo)

    # Also: number-level scoring from template → generate combos
    num_votes = Counter()
    for _, idx in similarities[:40]:
        if idx + 1 < n:
            for num in history[idx+1]:
                num_votes[num] += 1

    top15 = [v for v, _ in num_votes.most_common(15)]
    if len(top15) >= 6:
        for _ in range(200):
            chosen = sorted(random.sample(top15, 6))
            combos.append(tuple(chosen))

    return list(set(combos))


# ================================================================
# GAP SCORING (for reranking)
# ================================================================
# Build gap models
gap_models = {}
for num in range(1, MAX_NUM+1):
    appearances = [i for i, draw in enumerate(data) if num in draw[:6]]
    if len(appearances) < 10:
        gap_models[num] = {'mean': 7.5, 'std': 7.0}
        continue
    gaps = [appearances[i+1]-appearances[i] for i in range(len(appearances)-1)]
    med = np.median(gaps)
    short_gaps = [gaps[i+1] for i in range(len(gaps)-1) if gaps[i] <= med]
    long_gaps = [gaps[i+1] for i in range(len(gaps)-1) if gaps[i] > med]
    gap_models[num] = {
        'mean': np.mean(gaps), 'std': np.std(gaps), 'median': med,
        'cond_short': np.mean(short_gaps) if short_gaps else np.mean(gaps),
        'cond_long': np.mean(long_gaps) if long_gaps else np.mean(gaps),
    }


def gap_rerank_score(combo, history, day=''):
    """Score a combo using gap timing — for reranking."""
    n = len(history)
    score = 0
    for num in combo:
        gm = gap_models[num]
        last_seen = -1
        for i in range(n-1, max(n-200, -1), -1):
            if num in history[i]:
                last_seen = i
                break
        current_gap = n - last_seen if last_seen >= 0 else 100

        if gm['mean'] > 0:
            due = current_gap / gm['mean']
        else:
            due = 1.0

        # Bell curve: peak at due=1.0-1.5
        if due < 0.5:
            score -= 2
        elif due < 0.8:
            score += 0
        elif due < 1.2:
            score += 2  # Prime time
        elif due < 1.8:
            score += 3  # Overdue
        elif due < 2.5:
            score += 2
        else:
            score += 1

        if num in ANOMALOUS:
            score += 0.5
        if num in AUTOCORR_NEG and current_gap <= 1:
            score -= 3
        if day == 'Sunday' and num in SUNDAY_BOOST:
            score += 2

    return score


# ================================================================
# CO-OCCURRENCE SCORE
# ================================================================
def cooc_score(combo, history):
    """Score combo by how often its pairs appeared together."""
    score = 0
    pair_freq = defaultdict(int)
    for h in history[-200:]:
        for p in combinations(sorted(h[:6]), 2):
            pair_freq[p] += 1
    for p in combinations(sorted(combo), 2):
        score += pair_freq.get(p, 0)
    return score * 0.02


# ================================================================
# COMBINED RERANKING
# ================================================================
def rerank_combos(combos, history, day='', N=20):
    """Rerank combos using gap + cooc scores."""
    # Pre-compute cooc pairs once
    pair_freq = defaultdict(int)
    for h in history[-200:]:
        for p in combinations(sorted(h[:6]), 2):
            pair_freq[p] += 1

    scored = []
    for combo in combos:
        s = 0
        s += gap_rerank_score(combo, history, day) * 0.6  # Gap primary filter
        # Inline cooc score
        for p in combinations(sorted(combo), 2):
            s += pair_freq.get(p, 0) * 0.01

        # Sum regularity bonus
        avg_sum = np.mean([sum(h[:6]) for h in history[-50:]])
        s -= abs(sum(combo) - avg_sum) * 0.01

        # Range bonus
        r = combo[-1] - combo[0]
        if 20 < r < 40: s += 0.5

        scored.append((combo, s))

    scored.sort(key=lambda x: -x[1])

    # Diversify: don't pick too similar combos
    selected = []
    for combo, sc in scored:
        if len(selected) >= N: break
        cs = set(combo)
        # Must differ by at least 2 numbers from all selected
        if all(len(cs & set(prev)) <= 4 for prev in selected):
            selected.append(combo)

    # Fill if not enough diverse combos
    for combo, sc in scored:
        if len(selected) >= N: break
        if combo not in selected:
            selected.append(combo)

    return selected[:N]


# ================================================================
# BACKTEST
# ================================================================
print(f"  BACKTESTING template-primary + gap-filter...\n")

START = 200
TESTED = total - START - 1
N_VALUES = [1, 3, 5, 10, 20]
np.random.seed(42)
random.seed(42)

results = {N: {'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0, 'best_match': []}
           for N in N_VALUES}

# Also: track how template+gap compares to template-only and gap-only
results_template_only = {N: {'h4': 0, 'h3': 0} for N in N_VALUES}
results_gap_only = {N: {'h4': 0, 'h3': 0} for N in N_VALUES}

t0 = time.time()
for idx in range(START, total-1):
    history = data[:idx+1]
    actual = set(data[idx+1][:6])
    actual_tuple = tuple(sorted(actual))
    day = days_of_week[idx+1] if idx+1 < len(days_of_week) else ''

    # Generate template pool
    pool = template_generate(history, 200)

    if not pool:
        pool = [tuple(sorted(history[-1][:6]))]

    for N in N_VALUES:
        # Template-primary with gap-filter reranking
        selected = rerank_combos(pool, history, day, N)

        best_match = 0
        hit6 = False
        for combo in selected:
            m = len(set(combo) & actual)
            best_match = max(best_match, m)
            if set(combo) == actual: hit6 = True

        results[N]['best_match'].append(best_match)
        if hit6: results[N]['h6'] += 1
        if best_match >= 5: results[N]['h5'] += 1
        if best_match >= 4: results[N]['h4'] += 1
        if best_match >= 3: results[N]['h3'] += 1

        # Template-only baseline (no reranking, just take first N)
        template_top = pool[:N]
        bm_t = max(len(set(c) & actual) for c in template_top) if template_top else 0
        if bm_t >= 4: results_template_only[N]['h4'] += 1
        if bm_t >= 3: results_template_only[N]['h3'] += 1

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        r = results[10]
        rt = results_template_only[10]
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        print(f"    Hybrid N=10:   6/6={r['h6']} >=5={r['h5']} "
              f">=4={r['h4']}({r['h4']/done*100:.1f}%) "
              f">=3={r['h3']}({r['h3']/done*100:.1f}%)")
        print(f"    Template N=10: >=4={rt['h4']} >=3={rt['h3']}({rt['h3']/done*100:.1f}%)")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS
# ================================================================
print(f"{'='*90}")
print(f" RESULTS — {TESTED} draws")
print(f"{'='*90}\n")

print(f"  HYBRID (Template + Gap-Filter):")
print(f"  {'N':<5} {'6/6':<8} {'≥5/6':<8} {'≥4/6':<10} {'≥3/6':<10} {'avg':<6}")
print(f"  {'-'*50}")
for N in N_VALUES:
    r = results[N]
    avg = np.mean(r['best_match'])
    print(f"  {N:<5} {r['h6']/TESTED*100:<8.3f} {r['h5']/TESTED*100:<8.3f} "
          f"{r['h4']/TESTED*100:<10.2f} {r['h3']/TESTED*100:<10.2f} {avg:<6.2f}")

print(f"\n  TEMPLATE-ONLY (no reranking):")
print(f"  {'N':<5} {'≥4/6':<10} {'≥3/6':<10}")
print(f"  {'-'*30}")
for N in N_VALUES:
    r = results_template_only[N]
    print(f"  {N:<5} {r['h4']/TESTED*100:<10.2f} {r['h3']/TESTED*100:<10.2f}")

# Random comparison
print(f"\n  RANDOM BASELINE:")
print(f"  {'N':<5} {'≥4/6':<10} {'≥3/6':<10}")
print(f"  {'-'*30}")
for N in N_VALUES:
    r4 = N * mcomb(6,4) * mcomb(39,2) / mcomb(45,6) * 100
    r3 = N * mcomb(6,3) * mcomb(39,3) / mcomb(45,6) * 100
    print(f"  {N:<5} {r4:<10.3f} {r3:<10.2f}")

# Improvement analysis
print(f"\n  IMPROVEMENT vs RANDOM:")
for N in [5, 10, 20]:
    r = results[N]
    r4_h = r['h4'] / TESTED * 100
    r3_h = r['h3'] / TESTED * 100
    r4_r = N * mcomb(6,4) * mcomb(39,2) / mcomb(45,6) * 100
    r3_r = N * mcomb(6,3) * mcomb(39,3) / mcomb(45,6) * 100
    print(f"  N={N:<3}: ≥4/6 = {r4_h/r4_r:.2f}x random | ≥3/6 = {r3_h/r3_r:.2f}x random")

# ================================================================
# PREDICTION FOR NEXT DRAW
# ================================================================
print(f"\n{'='*90}")
print(f" FINAL PREDICTION FOR NEXT DRAW")
print(f"{'='*90}\n")

next_day = ''
try:
    last_date = datetime.strptime(str(dates[-1])[:10], '%Y-%m-%d')
    draw_days = {2: 'Wednesday', 4: 'Friday', 6: 'Sunday'}
    for delta in range(1, 4):
        nd = last_date + timedelta(days=delta)
        if nd.weekday() in draw_days:
            next_day = draw_days[nd.weekday()]
            print(f"  Next draw: {nd.strftime('%Y-%m-%d')} ({next_day})")
            break
except:
    pass

print(f"  Last draw: {sorted(data[-1][:6])}")

pool = template_generate(data, 200)
print(f"  Template pool: {len(pool)} unique combos")

selected = rerank_combos(pool, data, next_day, 20)

print(f"\n  TOP 20 PREDICTED SETS (Template + Gap-Filter):")
print(f"  {'#':<4} {'Numbers':<35} {'Sum':<6} {'Range':<6} {'GapSc':<8}")
print(f"  {'-'*65}")
for i, combo in enumerate(selected):
    nums = " ".join(f"{n:>2}" for n in combo)
    gs = gap_rerank_score(combo, data, next_day)
    print(f"  {i+1:<4} [{nums}]  {sum(combo):<6} {combo[-1]-combo[0]:<6} {gs:<8.1f}")

# Number frequency in predictions
freq = Counter()
for combo in selected:
    for n in combo: freq[n] += 1
print(f"\n  Numbers appearing most:")
for n, c in freq.most_common(10):
    anom = " *ANOM*" if n in ANOMALOUS else ""
    print(f"    #{n:>2}: {c}/20 sets{anom}")

print(f"\n{'='*90}")
