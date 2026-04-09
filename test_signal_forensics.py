"""
DEEP HIT FORENSICS — TÌM 4 YẾU TỐ MẠNH NHẤT
=================================================
Thay vì test method → đo hit rate,
làm ngược lại: XEM KHI NÀO TRÚNG → tín hiệu gì đang active?

Cho mỗi draw, tính 20+ tín hiệu riêng biệt cho mỗi số 1-45.
Sau đó đo: khi signal S rank số X vào top-6 và X thực sự xuất hiện,
→ signal S có accuracy bao nhiêu?

Tìm TOP 4 signals có accuracy cao nhất → exploit riêng.
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
all_records = get_mega645_all()
total = len(data)
MAX_NUM = 45
PICK = 6

try:
    from datetime import datetime
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

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" DEEP HIT FORENSICS — 20 INDIVIDUAL SIGNALS")
print(f" Tìm 4 yếu tố MẠNH NHẤT")
print(f"{'='*90}\n")

# ================================================================
# DEFINE 20 INDIVIDUAL SIGNALS
# Each signal returns a dict {num: score} for all 45 numbers
# ================================================================

def S01_freq_10(history):
    """Frequency in last 10 draws."""
    c = Counter()
    for h in history[-10:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S02_freq_20(history):
    """Frequency in last 20 draws."""
    c = Counter()
    for h in history[-20:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S03_freq_50(history):
    """Frequency in last 50 draws."""
    c = Counter()
    for h in history[-50:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S04_freq_100(history):
    """Frequency in last 100 draws."""
    c = Counter()
    for h in history[-100:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S05_transition(history):
    """Numbers that follow current draw's numbers (transition)."""
    follow = defaultdict(Counter)
    n = len(history)
    for i in range(n-1):
        for p in history[i][:6]:
            for nx in history[i+1][:6]:
                follow[p][nx] += 1
    scores = Counter()
    last = set(history[-1][:6])
    for p in last:
        total_t = sum(follow[p].values())
        if total_t > 0:
            for nx, cnt in follow[p].items():
                scores[nx] += cnt / total_t
    return dict(scores)

def S06_gap_due(history):
    """Gap-based due score: current_gap / mean_gap."""
    n = len(history)
    scores = {}
    for num in range(1, MAX_NUM+1):
        apps = [i for i, h in enumerate(history) if num in h[:6]]
        if len(apps) < 5:
            scores[num] = 0
            continue
        gaps = [apps[i+1]-apps[i] for i in range(len(apps)-1)]
        mg = np.mean(gaps)
        cg = n - apps[-1] if apps else 100
        scores[num] = cg / mg if mg > 0 else 0
    return scores

def S07_template_match(history):
    """Template matching: find similar draws → score next numbers."""
    n = len(history)
    last = set(history[-1][:6])
    scores = Counter()
    for i in range(max(0, n-500), n-2):
        overlap = len(set(history[i][:6]) & last)
        if overlap >= 2:
            for num in history[i+1][:6]:
                scores[num] += overlap
    return dict(scores)

def S08_cooccurrence(history):
    """Co-occurrence: how often does each number appear with recent numbers."""
    recent_nums = set()
    for h in history[-3:]:
        recent_nums.update(h[:6])
    scores = Counter()
    for h in history[-200:]:
        h_set = set(h[:6])
        overlap = len(h_set & recent_nums)
        if overlap >= 1:
            for num in h_set:
                scores[num] += overlap
    return dict(scores)

def S09_recency(history):
    """Recency: how recently did each number appear?"""
    n = len(history)
    last_seen = {}
    for i, h in enumerate(history):
        for num in h[:6]:
            last_seen[num] = i
    return {num: 1.0/(n - last_seen.get(num, 0)) for num in range(1, MAX_NUM+1)}

def S10_overdue(history):
    """Overdue: numbers with longest gap since last appearance."""
    n = len(history)
    last_seen = {}
    for i, h in enumerate(history):
        for num in h[:6]:
            last_seen[num] = i
    return {num: n - last_seen.get(num, 0) for num in range(1, MAX_NUM+1)}

def S11_repeat_2(history):
    """Numbers that appeared in last 2 draws."""
    c = Counter()
    for h in history[-2:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S12_repeat_3(history):
    """Numbers that appeared in last 3 draws."""
    c = Counter()
    for h in history[-3:]:
        for n in h[:6]: c[n] += 1
    return dict(c)

def S13_bigram_col(history):
    """Per-column bigram: for each sorted position, what follows?"""
    sd = [sorted(h[:6]) for h in history]
    n = len(sd)
    scores = Counter()
    for pos in range(6):
        vals = [s[pos] for s in sd]
        bg = defaultdict(Counter)
        for i in range(n-2):
            bg[(vals[i], vals[i+1])][vals[i+2]] += 1
        key = (vals[-2], vals[-1])
        if key in bg:
            for v, c in bg[key].most_common(3):
                scores[v] += c
    return dict(scores)

def S14_delta_pattern(history):
    """Delta pattern: most common change from last draw's numbers."""
    sd = [sorted(h[:6]) for h in history]
    n = len(sd)
    scores = Counter()
    for pos in range(6):
        vals = [s[pos] for s in sd]
        deltas = [vals[i+1]-vals[i] for i in range(n-1)]
        if deltas:
            dc = Counter(deltas)
            for d, cnt in dc.most_common(3):
                v = vals[-1] + d
                if 1 <= v <= MAX_NUM:
                    scores[v] += cnt
    return dict(scores)

def S15_conditional_prev(history):
    """Conditional on previous column's value."""
    sd = [sorted(h[:6]) for h in history]
    scores = Counter()
    for pos in range(1, 6):
        cond = defaultdict(Counter)
        for s in sd:
            cond[s[pos-1]][s[pos]] += 1
        pv = sd[-1][pos-1]
        if pv in cond:
            for v, c in cond[pv].most_common(3):
                scores[v] += c
    return dict(scores)

def S16_sum_zone(history):
    """Numbers that would bring sum close to historical average."""
    avg_sum = np.mean([sum(h[:6]) for h in history[-50:]])
    last_sum = sum(history[-1][:6])
    target = avg_sum  # Next draw sum should be close to avg
    scores = {}
    for num in range(1, MAX_NUM+1):
        # If this number replaces something, how close to target sum?
        scores[num] = max(0, 10 - abs(num - target/6))
    return scores

def S17_spacing_regular(history):
    """Numbers with regular spacing pattern."""
    sd = [sorted(h[:6]) for h in history]
    avg_spacing = np.mean([sd[-1][i+1]-sd[-1][i] for i in range(5)])
    scores = {}
    for num in range(1, MAX_NUM+1):
        # Score based on how well num fits as a "regular spacing" number
        min_dist = min(abs(num - sd[-1][p]) for p in range(6))
        if abs(min_dist - avg_spacing) < 3:
            scores[num] = 5 - abs(min_dist - avg_spacing)
        else:
            scores[num] = 0
    return scores

def S18_mod3_transition(history):
    """Modular arithmetic: mod-3 transition pattern."""
    scores = Counter()
    for h in history[-50:]:
        for n in h[:6]:
            m = n % 3
            # Next appearance bias by mod class
            scores[n] += 1
    # Boost the mod class that's most common in transitions
    return dict(scores)

def S19_even_odd_balance(history):
    """Numbers that balance even/odd ratio toward 3/3."""
    last = history[-1][:6]
    n_even = sum(1 for n in last if n%2==0)
    scores = {}
    for num in range(1, MAX_NUM+1):
        if n_even > 3 and num%2==1:  # Too many even, prefer odd
            scores[num] = 2
        elif n_even < 3 and num%2==0:  # Too few even, prefer even
            scores[num] = 2
        else:
            scores[num] = 1
    return scores

def S20_gap_conditional(history):
    """Conditional gap: after short/long gap, what's the expected return?"""
    n = len(history)
    scores = {}
    for num in range(1, MAX_NUM+1):
        apps = [i for i, h in enumerate(history) if num in h[:6]]
        if len(apps) < 5:
            scores[num] = 0
            continue
        gaps = [apps[i+1]-apps[i] for i in range(len(apps)-1)]
        cg = n - apps[-1]
        med = np.median(gaps)
        # Previous gap
        if len(apps) >= 2:
            prev_gap = apps[-1] - apps[-2]
        else:
            prev_gap = np.mean(gaps)
        # Conditional expected
        if prev_gap <= med:
            cond_gaps = [gaps[i+1] for i in range(len(gaps)-1) if gaps[i] <= med]
        else:
            cond_gaps = [gaps[i+1] for i in range(len(gaps)-1) if gaps[i] > med]
        exp = np.mean(cond_gaps) if cond_gaps else np.mean(gaps)
        # Due ratio
        scores[num] = cg / exp if exp > 0 else 0
    return scores


ALL_SIGNALS = {
    'S01_freq10':       S01_freq_10,
    'S02_freq20':       S02_freq_20,
    'S03_freq50':       S03_freq_50,
    'S04_freq100':      S04_freq_100,
    'S05_transition':   S05_transition,
    'S06_gap_due':      S06_gap_due,
    'S07_template':     S07_template_match,
    'S08_cooccur':      S08_cooccurrence,
    'S09_recency':      S09_recency,
    'S10_overdue':      S10_overdue,
    'S11_repeat2':      S11_repeat_2,
    'S12_repeat3':      S12_repeat_3,
    'S13_bigram':       S13_bigram_col,
    'S14_delta':        S14_delta_pattern,
    'S15_cond_prev':    S15_conditional_prev,
    'S16_sum_zone':     S16_sum_zone,
    'S17_spacing':      S17_spacing_regular,
    'S18_mod3':         S18_mod3_transition,
    'S19_eo_balance':   S19_even_odd_balance,
    'S20_gap_cond':     S20_gap_conditional,
}

# ================================================================
# BACKTEST: Per-signal accuracy
# For each signal, rank 45 numbers → take top-K → measure precision
# ================================================================
print(f"  Testing {len(ALL_SIGNALS)} individual signals...\n")

START = 200
TESTED = total - START - 1

# Per signal: how many correct numbers in top-K
signal_stats = {}
for sname in ALL_SIGNALS:
    signal_stats[sname] = {
        'top1_correct': 0, 'top3_correct': 0, 'top6_correct': 0,
        'top10_correct': 0, 'top15_correct': 0,
        'top6_total_correct': 0,  # Total correct numbers in top-6
        'precision_6': [],  # P(correct | in top-6) per draw
    }

# Also: per-draw, which signals had ALL 6 correct in their top-K?
hit_analysis = {k: [] for k in [10, 15, 20]}  # K → list of (draw_idx, signals_that_covered)

t0 = time.time()
for idx in range(START, total-1):
    history = data[:idx+1]
    actual = set(data[idx+1][:6])

    signal_scores = {}
    for sname, sfunc in ALL_SIGNALS.items():
        try:
            scores = sfunc(history)
            signal_scores[sname] = scores
        except:
            signal_scores[sname] = {}

    for sname, scores in signal_scores.items():
        if not scores:
            continue
        ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -scores.get(x, 0))

        # Top-K accuracy
        st = signal_stats[sname]
        if ranking[0] in actual: st['top1_correct'] += 1
        top3 = set(ranking[:3])
        top6 = set(ranking[:6])
        top10 = set(ranking[:10])
        top15 = set(ranking[:15])
        st['top3_correct'] += len(top3 & actual)
        n_correct_6 = len(top6 & actual)
        st['top6_correct'] += 1 if n_correct_6 > 0 else 0
        st['top6_total_correct'] += n_correct_6
        st['top10_correct'] += len(top10 & actual)
        st['top15_correct'] += len(top15 & actual)
        st['precision_6'].append(n_correct_6 / 6)

    # Check which signals covered all 6
    for K in [10, 15, 20]:
        covering_signals = []
        for sname, scores in signal_scores.items():
            if not scores: continue
            ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -scores.get(x, 0))
            topK = set(ranking[:K])
            if actual.issubset(topK):
                covering_signals.append(sname)
        if covering_signals:
            hit_analysis[K].append((idx+1, covering_signals))

    done = idx - START + 1
    if done % 200 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s\n")

# ================================================================
# RESULTS: Per-SIGNAL accuracy ranking
# ================================================================
print(f"{'='*90}")
print(f" PER-SIGNAL ACCURACY — {TESTED} draws")
print(f"{'='*90}\n")

# Expected random values
exp_top1 = PICK / MAX_NUM * 100
exp_top6 = 1 - (1 - PICK/MAX_NUM)**6
exp_correct_in_6 = 6 * PICK / MAX_NUM

print(f"  Random baseline: top-1={exp_top1:.1f}%, "
      f"top-6 has≥1={exp_top6*100:.1f}%, "
      f"avg correct in top-6={exp_correct_in_6:.2f}/6\n")

# Sort by avg correct in top-6 (most important metric)
signal_ranking = sorted(ALL_SIGNALS.keys(),
    key=lambda s: signal_stats[s]['top6_total_correct'] / max(1, TESTED),
    reverse=True)

print(f"  {'Signal':<16} {'Top1':<8} {'AvgCorrect/6':<14} "
      f"{'Top6≥1':<10} {'Top10':<10} {'Top15':<10} {'vs Random':<10}")
print(f"  {'-'*80}")

for sname in signal_ranking:
    st = signal_stats[sname]
    t1 = st['top1_correct'] / TESTED * 100
    avg6 = st['top6_total_correct'] / TESTED
    top6_rate = st['top6_correct'] / TESTED * 100
    top10_avg = st['top10_correct'] / TESTED
    top15_avg = st['top15_correct'] / TESTED
    vs_random = avg6 / exp_correct_in_6

    marker = " <<<" if vs_random > 1.05 else ""
    print(f"  {sname:<16} {t1:<8.1f} {avg6:<14.3f} "
          f"{top6_rate:<10.1f} {top10_avg:<10.2f} {top15_avg:<10.2f} {vs_random:<10.2f}x{marker}")

# ================================================================
# TOP 4 SIGNALS — DEEP ANALYSIS
# ================================================================
top4 = signal_ranking[:4]
print(f"\n{'='*90}")
print(f" TOP 4 STRONGEST SIGNALS")
print(f"{'='*90}\n")

for rank, sname in enumerate(top4):
    st = signal_stats[sname]
    avg6 = st['top6_total_correct'] / TESTED
    t1 = st['top1_correct'] / TESTED * 100
    vs = avg6 / exp_correct_in_6

    print(f"  #{rank+1}: {sname}")
    print(f"  {'─'*60}")
    print(f"  Top-1 accuracy: {t1:.1f}% (random={exp_top1:.1f}%)")
    print(f"  Avg correct in top-6: {avg6:.3f}/6 (random={exp_correct_in_6:.2f}/6)")
    print(f"  vs Random: {vs:.2f}x")

    # Precision distribution
    precs = st['precision_6']
    print(f"  Precision distribution (correct/6 per draw):")
    for k in [0, 1, 2, 3, 4, 5, 6]:
        cnt = sum(1 for p in precs if int(p*6+0.5) == k)
        bar = '█' * (cnt // 10)
        print(f"    {k}/6: {cnt:>4} ({cnt/TESTED*100:>5.1f}%) {bar}")
    print()

# ================================================================
# COVERAGE ANALYSIS: Which signals cover ALL 6?
# ================================================================
print(f"{'='*90}")
print(f" ALL-6-COVERED ANALYSIS")
print(f"{'='*90}\n")

for K in [10, 15, 20]:
    hits = hit_analysis[K]
    print(f"  Top-{K}: {len(hits)} draws had ALL 6 in top-{K} ({len(hits)/TESTED*100:.2f}%)")
    if hits:
        # Which signals covered most?
        sig_count = Counter()
        for _, sigs in hits:
            for s in sigs: sig_count[s] += 1
        print(f"  Signals that covered ALL 6:")
        for s, c in sig_count.most_common(10):
            print(f"    {s}: {c}/{len(hits)} times ({c/len(hits)*100:.0f}%)")
    print()

# ================================================================
# SIGNAL COMBINATION: What's the best 2-signal, 3-signal, 4-signal?
# ================================================================
print(f"{'='*90}")
print(f" SIGNAL COMBINATIONS — Union of top-K")
print(f"{'='*90}\n")

# For each pair of top-4 signals: what's their combined accuracy?
print(f"  Testing top-4 signal combinations...\n")

for n_signals in [2, 3, 4]:
    best_combo = None
    best_avg = 0

    from itertools import combinations as comb_iter
    for combo in comb_iter(top4, n_signals):
        # Simulate: union of top-6 from each signal → how many correct?
        total_correct = 0
        for idx in range(START, total-1):
            history = data[:idx+1]
            actual = set(data[idx+1][:6])
            union_top = set()
            for sname in combo:
                try:
                    scores = ALL_SIGNALS[sname](history)
                    ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -scores.get(x, 0))
                    union_top.update(ranking[:6])
                except:
                    pass
            total_correct += len(union_top & actual)

        avg = total_correct / TESTED
        if avg > best_avg:
            best_avg = avg
            best_combo = combo

    pool_size = n_signals * 6
    exp_random = 6 * pool_size / MAX_NUM
    print(f"  Best {n_signals}-signal combo: {best_combo}")
    print(f"    Avg correct (from {pool_size} candidates): "
          f"{best_avg:.3f}/6 (random ~{exp_random:.2f}/6 = {best_avg/exp_random:.2f}x)")
    print()

# ================================================================
# FINAL: INTERSECTION EXPLOIT
# ================================================================
print(f"{'='*90}")
print(f" INTERSECTION EXPLOIT — Top 4 Signals")
print(f"{'='*90}\n")
print(f"  Testing: numbers that appear in TOP-K of MULTIPLE signals\n")

for K in [6, 8, 10, 12, 15]:
    # For each draw: find numbers that are in top-K of ≥N signals
    for min_signals in [2, 3, 4]:
        total_correct = 0
        total_candidates = 0
        n_draws_with_candidates = 0

        for idx in range(START, total-1):
            history = data[:idx+1]
            actual = set(data[idx+1][:6])
            vote_count = Counter()

            for sname in top4:
                try:
                    scores = ALL_SIGNALS[sname](history)
                    ranking = sorted(range(1, MAX_NUM+1),
                                     key=lambda x: -scores.get(x, 0))
                    for num in ranking[:K]:
                        vote_count[num] += 1
                except:
                    pass

            # Numbers in top-K of ≥ min_signals signals
            consensus = {num for num, cnt in vote_count.items() if cnt >= min_signals}

            if consensus:
                n_draws_with_candidates += 1
                total_candidates += len(consensus)
                total_correct += len(consensus & actual)

        if n_draws_with_candidates > 0:
            avg_cands = total_candidates / n_draws_with_candidates
            avg_correct = total_correct / n_draws_with_candidates
            precision = total_correct / total_candidates * 100 if total_candidates > 0 else 0
            random_prec = PICK / MAX_NUM * 100
            print(f"  K={K:<3} ≥{min_signals} signals: "
                  f"avg={avg_cands:.1f} candidates, "
                  f"{avg_correct:.2f} correct, "
                  f"precision={precision:.1f}% "
                  f"(random={random_prec:.1f}%, "
                  f"{precision/random_prec:.1f}x)")

print(f"\n{'='*90}")
