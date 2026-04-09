"""
FAST FULL BACKTEST — Test prediction accuracy across ALL 1486 historical draws.
Uses lighter methods for speed while still being realistic.
"""
import sys, os, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import get_mega645_numbers, get_mega645_all


def fast_consensus_predict(data, max_num=45, pick=6):
    """Fast version of consensus — core signals only, no sub-engine calls."""
    n = len(data)
    if n < 30:
        return sorted(np.random.choice(range(1, max_num+1), pick, replace=False).tolist())
    
    last_set = set(data[-1][:6])
    scores = {num: 0.0 for num in range(1, max_num + 1)}
    
    # Signal 1: Transition (what follows last draw)
    follow = defaultdict(Counter)
    pc = Counter()
    for i in range(n - 1):
        for p in data[i][:6]:
            pc[p] += 1
            for x in data[i + 1][:6]:
                follow[p][x] += 1
    base = pick / max_num
    for num in range(1, max_num + 1):
        total_f = sum(follow[p].get(num, 0) for p in last_set)
        total_p = sum(pc[p] for p in last_set)
        if total_p > 0:
            cond_p = total_f / total_p
            scores[num] += (cond_p / base - 1) * 3
    
    # Signal 2: Momentum (short vs long term frequency)
    if n >= 50:
        for num in range(1, max_num + 1):
            f5 = sum(1 for d in data[-5:] if num in d[:6]) / 5
            f20 = sum(1 for d in data[-20:] if num in d[:6]) / 20
            f50 = sum(1 for d in data[-50:] if num in d[:6]) / 50
            scores[num] += (f5 - f20) * 10 + (f20 - f50) * 5
    
    # Signal 3: Gap timing
    last_seen = {}
    gap_sums = defaultdict(float)
    gap_counts = defaultdict(int)
    for i, d in enumerate(data):
        for num in d[:6]:
            if num in last_seen:
                gap_sums[num] += (i - last_seen[num])
                gap_counts[num] += 1
            last_seen[num] = i
    for num in range(1, max_num + 1):
        gc = gap_counts.get(num, 0)
        if gc < 5: continue
        mg = gap_sums[num] / gc
        cg = n - last_seen.get(num, 0)
        ratio = cg / mg if mg > 0 else 1
        if ratio > 1.0:
            scores[num] += max(0, (ratio - 0.8)) * 2
    
    # Signal 4: KNN similarity
    knn = Counter()
    for i in range(n - 2):
        sim = len(set(data[i][:6]) & last_set)
        if sim >= 3:
            for num in data[i + 1][:6]:
                knn[num] += sim ** 2
    mx = max(knn.values()) if knn else 1
    for num in knn:
        scores[num] += knn[num] / mx * 3
    
    # Signal 5: Co-occurrence
    pf = Counter()
    for d in data[-200:]:
        for p in combinations(sorted(d[:pick]), 2):
            pf[p] += 1
    for num in range(1, max_num + 1):
        scores[num] += sum(pf.get(tuple(sorted([p, num])), 0) for p in last_set) * 0.1
    
    # Signal 6: Markov-2
    if n >= 10:
        last2 = set(data[-2][:6])
        both_count = Counter()
        either_count = Counter()
        total_both = total_either = 0
        for i in range(2, n):
            prev2 = set(data[i-2][:6])
            prev1 = set(data[i-1][:6])
            curr = set(data[i][:6])
            for num in range(1, max_num + 1):
                in_p2 = num in prev2
                in_p1 = num in prev1
                if in_p2 and in_p1:
                    total_both += 1
                    if num in curr: both_count[num] += 1
                elif in_p2 or in_p1:
                    total_either += 1
                    if num in curr: either_count[num] += 1
        for num in range(1, max_num + 1):
            in_l1 = num in last_set
            in_l2 = num in last2
            if in_l1 and in_l2:
                p = both_count[num] / max(total_both / max_num, 1)
            elif in_l1 or in_l2:
                p = either_count[num] / max(total_either / max_num, 1)
            else:
                p = 0
            scores[num] += (p - base) * 8
    
    # Signal 7: Wavelet-like
    if n >= 64:
        for num in range(1, max_num + 1):
            seq = np.array([1.0 if num in d[:6] else 0.0 for d in data[-64:]])
            sigs = []
            for scale in [4, 8, 16]:
                nb = len(seq) // scale
                bm = [np.mean(seq[j*scale:(j+1)*scale]) for j in range(nb)]
                if len(bm) >= 2:
                    sigs.append(bm[-1] - bm[-2])
            if sigs:
                scores[num] += sum(s * w for s, w in zip(sigs, [3, 2, 1])) * 4
    
    # Adaptive anti-repeat
    rcs = []
    for i in range(1, min(n, 100)):
        rcs.append(len(set(data[i-1][:6]) & set(data[i][:6])))
    penalty = (np.mean(rcs) - 1.0) * 1.5 if rcs else -2.0
    for num in last_set:
        scores[num] += penalty
    
    # Build pool and generate best combo
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = [num for num, _ in ranked[:35]]
    
    # Constraints
    recent = data[-50:]
    sums = [sum(d[:pick]) for d in recent]
    sum_lo = int(np.percentile(sums, 5))
    sum_hi = int(np.percentile(sums, 95))
    odds = [sum(1 for x in d[:pick] if x % 2 == 1) for d in recent]
    odd_lo = max(0, int(np.percentile(odds, 5)))
    odd_hi = min(pick, int(np.percentile(odds, 95)))
    ranges = [max(d[:pick]) - min(d[:pick]) for d in recent]
    rng_lo = int(np.percentile(ranges, 5))
    rng_hi = int(np.percentile(ranges, 95))
    
    best = None
    best_score = -float('inf')
    for combo in combinations(pool[:16], pick):
        s = sum(combo)
        if s < sum_lo or s > sum_hi: continue
        r = combo[5] - combo[0]
        if r < rng_lo or r > rng_hi: continue
        o = sum(1 for x in combo if x % 2 == 1)
        if o < odd_lo or o > odd_hi: continue
        sc = sum(scores.get(num, 0) for num in combo)
        if sc > best_score:
            best_score = sc
            best = sorted(combo)
    
    return best if best else sorted(pool[:pick])


def fast_portfolio_predict(data, max_num=45, pick=6, n_sets=50):
    """Generate portfolio of diverse predictions."""
    n = len(data)
    if n < 30:
        return [sorted(np.random.choice(range(1, max_num+1), pick, replace=False).tolist())]
    
    last_set = set(data[-1][:6])
    
    # Build scores (simplified)
    scores = {num: 0.0 for num in range(1, max_num + 1)}
    
    # Frequency + momentum
    for num in range(1, max_num + 1):
        f_recent = sum(1 for d in data[-10:] if num in d[:6]) / 10
        f_overall = sum(1 for d in data if num in d[:6]) / n
        scores[num] = f_recent * 3 + f_overall
    
    # Gap overdue
    last_seen = {}
    for i, d in enumerate(data):
        for num in d[:6]:
            last_seen[num] = i
    for num in range(1, max_num + 1):
        gap = n - last_seen.get(num, 0)
        f = sum(1 for d in data if num in d[:6])
        avg_gap = n / (f + 1)
        if gap > avg_gap:
            scores[num] += (gap / avg_gap - 1) * 2
    
    # KNN
    for i in range(n - 2):
        sim = len(set(data[i][:6]) & last_set)
        if sim >= 3:
            for num in data[i + 1][:6]:
                scores[num] += sim * 0.5
    
    # Anti-repeat
    for num in last_set:
        scores[num] -= 1.5
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = sorted([num for num, _ in ranked[:35]])
    
    # Constraints
    recent = data[-50:]
    sums = [sum(d[:pick]) for d in recent]
    sum_lo = int(np.percentile(sums, 3))
    sum_hi = int(np.percentile(sums, 97))
    
    portfolio = []
    for combo in combinations(pool[:20], pick):
        s = sum(combo)
        if s < sum_lo or s > sum_hi: continue
        sc = sum(scores.get(num, 0) for num in combo)
        portfolio.append((sorted(combo), sc))
    
    portfolio.sort(key=lambda x: -x[1])
    
    # Take top n_sets with diversity
    result = []
    for combo, sc in portfolio:
        if len(result) >= n_sets:
            break
        if all(len(set(combo) - set(p)) >= 2 for p in result):
            result.append(combo)
    
    return result if result else [sorted(pool[:pick])]


def run_fast_backtest():
    print("=" * 70)
    print("  TINNAM AI — COMPREHENSIVE BACKTEST (FAST MODE)")
    print("  Testing accuracy across ALL historical draws")
    print("=" * 70)

    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    n_total = len(data)
    
    print(f"\n  📊 Data: {n_total} draws ({dates[0]} → {dates[-1]})")
    print(f"  🎯 Lottery: Mega 6/45")
    print(f"  🎲 C(45,6) = 8,145,060 total combinations")
    print(f"  📈 Random expected: {6*6/45:.4f}/6 avg matches\n")

    min_train = 80
    
    # Test ALL draws (fast enough with simplified consensus)
    test_indices = list(range(min_train, n_total - 1))
    n_tests = len(test_indices)
    print(f"  Testing {n_tests} draws (walk-forward)...\n")
    
    # Method 1: Fast Consensus (7 signals)
    print("  [1/4] Fast Consensus Engine (7 core signals)...")
    consensus_matches = []
    start = time.time()
    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:6])
        predicted = set(fast_consensus_predict(train))
        consensus_matches.append(len(predicted & actual))
        if (i + 1) % 200 == 0:
            print(f"        [{i+1}/{n_tests}] avg={np.mean(consensus_matches):.4f}/6")
    t1 = time.time() - start
    print(f"        Done in {t1:.1f}s — avg={np.mean(consensus_matches):.4f}/6\n")
    
    # Method 2: Portfolio (50 sets)
    print("  [2/4] Portfolio Coverage (50 diverse sets)...")
    portfolio_best = []
    start = time.time()
    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:6])
        portfolio = fast_portfolio_predict(train, n_sets=50)
        best = max(len(set(p) & actual) for p in portfolio)
        portfolio_best.append(best)
        if (i + 1) % 200 == 0:
            print(f"        [{i+1}/{n_tests}] avg={np.mean(portfolio_best):.4f}/6")
    t2 = time.time() - start
    print(f"        Done in {t2:.1f}s — avg={np.mean(portfolio_best):.4f}/6\n")
    
    # Method 3: Simple Markov
    print("  [3/4] Markov Chain Transition...")
    markov_matches = []
    start = time.time()
    for i, train_end in enumerate(test_indices):
        train = data[:train_end + 1]
        actual = set(data[train_end + 1][:6])
        last = set(train[-1][:6])
        trans = defaultdict(Counter)
        for j in range(len(train) - 1):
            for p in train[j][:6]:
                for x in train[j + 1][:6]:
                    trans[p][x] += 1
        votes = Counter()
        for p in last:
            for num, cnt in trans[p].most_common(10):
                votes[num] += cnt
        predicted = set(num for num, _ in votes.most_common(6))
        markov_matches.append(len(predicted & actual))
    t3 = time.time() - start
    print(f"        Done in {t3:.1f}s — avg={np.mean(markov_matches):.4f}/6\n")
    
    # Method 4: Random baseline
    print("  [4/4] Random Baseline...")
    random_matches = []
    np.random.seed(42)
    for i, train_end in enumerate(test_indices):
        actual = set(data[train_end + 1][:6])
        predicted = set(np.random.choice(range(1, 46), 6, replace=False).tolist())
        random_matches.append(len(predicted & actual))
    t4 = 0
    print(f"        Done — avg={np.mean(random_matches):.4f}/6\n")
    
    # =====================================================
    # RESULTS TABLE
    # =====================================================
    random_avg = 6 * 6 / 45
    
    all_methods = [
        ("🏆 Fast Consensus (7 sig)", consensus_matches, t1),
        ("📦 Portfolio 50 sets", portfolio_best, t2),
        ("🔗 Markov Transition", markov_matches, t3),
        ("🎲 Random Baseline", random_matches, t4),
    ]
    
    print("=" * 70)
    print("  FINAL RESULTS — MATCH DISTRIBUTION")
    print(f"  {n_tests} walk-forward tests on Mega 6/45")
    print("=" * 70)
    
    for name, matches, elapsed in all_methods:
        dist = Counter(matches)
        avg = np.mean(matches)
        improvement = (avg / random_avg - 1) * 100
        
        print(f"\n  {'─' * 66}")
        print(f"  {name}")
        print(f"  Avg: {avg:.4f}/6 | Max: {max(matches)}/6 | "
              f"{'+'if improvement>0 else ''}{improvement:.1f}% vs random | "
              f"Time: {elapsed:.1f}s")
        print(f"  {'─' * 66}")
        
        for k in range(7):
            c = dist.get(k, 0)
            pct = c / len(matches) * 100
            bar_len = int(pct)
            bar = '█' * bar_len
            label = ""
            if k == 6: label = " ← JACKPOT 6/6"
            elif k == 5: label = " ← 5/6"
            elif k == 4: label = " ← 4/6"
            elif k == 3: label = " ← 3/6 (prize)"
            print(f"    {k}/6: {c:5d} ({pct:6.2f}%) |{bar}{label}")
        
        # Key stats
        pct_3plus = sum(1 for m in matches if m >= 3) / len(matches) * 100
        pct_4plus = sum(1 for m in matches if m >= 4) / len(matches) * 100
        pct_5plus = sum(1 for m in matches if m >= 5) / len(matches) * 100
        pct_6 = sum(1 for m in matches if m >= 6) / len(matches) * 100
        print(f"    ≥3/6: {pct_3plus:.2f}% | ≥4/6: {pct_4plus:.2f}% | "
              f"≥5/6: {pct_5plus:.2f}% | 6/6: {pct_6:.2f}%")
    
    # =====================================================
    # REALITY CHECK
    # =====================================================
    print(f"\n{'=' * 70}")
    print("  🧮 MATHEMATICAL REALITY CHECK")
    print(f"{'=' * 70}")
    
    print(f"""
  ┌───────────────────────────────────────────────────────┐
  │  Vietlott 6/45: C(45,6) = 8,145,060 combinations     │
  │                                                        │
  │  To get 6/6 with 60% accuracy:                        │
  │    → Need to predict 1 in 8 million correctly         │
  │    → 60% of the time ← IMPOSSIBLE                    │
  │                                                        │
  │  Maximum theoretical improvement:                      │
  │    → Random: ~0.80/6 avg, 3/6 in ~3.3% of draws      │
  │    → Best model: ~0.85-0.95/6 avg, 3/6 in ~5-10%     │
  │    → 6/6 jackpot: ~0.00% (same as random)             │
  │                                                        │
  │  With 50-set portfolio:                                │
  │    → 50× more chances = 50/8,145,060                  │
  │    → Still only 0.000614% per draw                    │
  │                                                        │
  │  Even buying ALL 8,145,060 tickets:                   │
  │    → Costs 81.4 BILLION VND                           │
  │    → Jackpot usually < 50B VND → NET LOSS             │
  └───────────────────────────────────────────────────────┘
""")
    
    print("=" * 70)
    print("  TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    run_fast_backtest()
