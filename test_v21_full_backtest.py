"""
V21 Fast Backtest — Tests V21 scoring across ALL draws, optimized for speed.
Pre-computes heavy features ONCE and only recomputes fast per-draw features.

Usage: python test_v21_full_backtest.py [mega|power]
"""
import sys
import os
import time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import math

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from scraper.data_manager import get_mega645_numbers, get_power655_numbers


def score_and_predict(history, max_number, pick_count, constraints):
    """Fast V21-style scoring for a single test point."""
    n_draws = len(history)
    flat = [n for d in history for n in d]
    last = set(history[-1])
    
    # Basic stats
    last_seen = {}
    for i, d in enumerate(history):
        for n in d:
            last_seen[n] = i
    
    exp_gap = max_number / pick_count
    freq_10 = Counter(n for d in history[-10:] for n in d)
    freq_30 = Counter(n for d in history[-30:] for n in d)
    freq_50 = Counter(n for d in history[-50:] for n in d)
    r10 = Counter(n for d in history[-10:] for n in d)
    p10 = Counter(n for d in history[-20:-10] for n in d) if n_draws > 20 else r10
    
    # KNN
    knn_scores = Counter()
    for i in range(len(history) - 2):
        ov = len(set(history[i]) & last)
        if ov >= 2:
            for n in history[i+1]:
                knn_scores[n] += ov ** 1.5
    
    # Ngram
    ngram_scores = Counter()
    bigram = defaultdict(Counter)
    for i in range(1, n_draws):
        for pn in history[i-1]:
            for cn in history[i]:
                bigram[pn][cn] += 1
    for pn in history[-1]:
        total = sum(bigram[pn].values())
        if total > 0:
            for nn, cnt in bigram[pn].most_common(10):
                ngram_scores[nn] += cnt / total
    
    # Cycle
    cycle_scores = {}
    for num in range(1, max_number + 1):
        seq = np.array([1.0 if num in d else 0.0 for d in history[-200:]])
        if len(seq) < 30:
            cycle_scores[num] = 0.0
            continue
        seq_c = seq - np.mean(seq)
        fft = np.fft.rfft(seq_c)
        power = np.abs(fft) ** 2
        if len(power) < 3:
            cycle_scores[num] = 0.0
            continue
        freqs = np.fft.rfftfreq(len(seq_c))
        peak_idx = np.argmax(power[2:]) + 2
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
        peak_power = power[peak_idx] if peak_idx < len(power) else 0
        total_power = np.sum(power[1:]) + 1e-10
        spectral_ratio = peak_power / total_power
        if spectral_ratio > 0.15 and peak_freq > 0:
            period = 1.0 / peak_freq
            phase = (len(seq) % period) / period
            cycle_scores[num] = spectral_ratio * max(0, math.cos(2 * math.pi * phase)) * 3.0
        else:
            cycle_scores[num] = 0.0
    
    # Context match
    context_scores = Counter()
    if n_draws >= 20:
        last3 = [set(d) for d in history[-3:]]
        for i in range(3, n_draws - 1):
            hist3 = [set(d) for d in history[i-3:i]]
            sim = sum(len(hist3[j] & last3[j]) for j in range(3))
            if sim >= 4:
                for num in history[i]:
                    context_scores[num] += sim ** 2
    
    # Correlation (fast)
    cond_prob = defaultdict(Counter)
    window = history[-100:] if n_draws > 100 else history
    for k in range(1, len(window)):
        prev = set(window[k-1])
        curr = set(window[k])
        for i in prev:
            for j in curr:
                cond_prob[i][j] += 1
    
    # Triplet mining (V21)
    triplet_scores = Counter()
    if n_draws >= 50:
        tw = history[-100:] if n_draws > 100 else history
        triplet_freq = Counter()
        for d in tw:
            sd = sorted(d)
            for trip in combinations(sd, 3):
                triplet_freq[trip] += 1
        for trip, count in triplet_freq.most_common(200):
            if count < 2:
                break
            trip_set = set(trip)
            overlap = trip_set & last
            if len(overlap) >= 2:
                for num in trip_set - last:
                    triplet_scores[num] += count * len(overlap)
    
    # Sequential pattern (V21)
    seq_pattern_scores = Counter()
    if n_draws >= 30:
        for chain_len in [3, 4]:
            current_chain = [frozenset(d) for d in history[-chain_len:]]
            weight = chain_len
            for i in range(chain_len, min(n_draws - chain_len, n_draws - 1)):
                hist_chain = [frozenset(d) for d in history[i-chain_len:i]]
                sim = sum(len(current_chain[j] & hist_chain[j]) for j in range(chain_len))
                if sim >= chain_len * 2 and i < n_draws - 1:
                    for num in history[i]:
                        if num not in last:
                            seq_pattern_scores[num] += sim * weight * 0.1
    
    # Entropy (V21)
    entropy_scores = {}
    if n_draws >= 60:
        for num in range(1, max_number + 1):
            seq = [1 if num in d else 0 for d in history[-60:]]
            trans = {0: [0, 0], 1: [0, 0]}
            for i in range(1, len(seq)):
                trans[seq[i-1]][seq[i]] += 1
            current_state = seq[-1]
            total = sum(trans[current_state])
            prob_appear = trans[current_state][1] / total if total > 0 else pick_count / max_number
            entropy = 0
            for state in [0, 1]:
                t = sum(trans[state])
                if t == 0: continue
                for c in trans[state]:
                    if c > 0:
                        p = c / t
                        entropy -= p * math.log2(p) * (t / len(seq))
            predictability = max(0, 1.0 - entropy)
            entropy_scores[num] = prob_appear * predictability
    
    # Anti-repeat strength
    repeat_rate = 0
    if n_draws >= 20:
        repeats = sum(len(set(history[i]) & set(history[i+1])) for i in range(max(0, n_draws-20), n_draws-1))
        repeat_rate = repeats / (20 * pick_count)
    anti_str = 1.0 - min(repeat_rate * 5, 0.5)
    
    max_knn = max(knn_scores.values()) if knn_scores else 1
    max_ctx = max(context_scores.values()) if context_scores else 1
    max_trip = max(triplet_scores.values()) if triplet_scores else 1
    max_seq = max(seq_pattern_scores.values()) if seq_pattern_scores else 1
    
    scores = {}
    for num in range(1, max_number + 1):
        s = 0.0
        s += freq_10.get(num, 0) / 10 * 3.0
        s += freq_30.get(num, 0) / 30 * 2.0
        s += freq_50.get(num, 0) / 50 * 1.5
        gap = n_draws - last_seen.get(num, 0)
        s += max(0, gap / exp_gap - 0.8) * 2.5
        if num in last:
            s -= 5.0 * anti_str
        s += (r10.get(num, 0) - p10.get(num, 0)) / 5 * 2.0
        s += knn_scores.get(num, 0) / max(1, max_knn) * 2.5
        f_r = sum(1 for d in history[-15:] if num in d) / 15
        f_o = sum(1 for d in history[-45:-15] if num in d) / 30 if n_draws > 45 else f_r
        s += max(0, f_r - f_o) * 10
        # Run-length
        curr_abs = 0
        for d in reversed(history):
            if num not in d:
                curr_abs += 1
            else:
                break
        if curr_abs > 0:
            seq_arr = [1 if num in d else 0 for d in history]
            abs_runs = []
            run = 0
            for sv in seq_arr:
                if sv == 0: run += 1
                else:
                    if run > 0: abs_runs.append(run)
                    run = 0
            avg_a = np.mean(abs_runs) if abs_runs else exp_gap
            if avg_a > 0:
                s += 1 / (1 + math.exp(-3 * (curr_abs / avg_a - 0.8))) * 2.0
        s += ngram_scores.get(num, 0) * 3.0
        # Multi-scale
        f5 = sum(1 for d in history[-5:] if num in d) / 5
        f15 = sum(1 for d in history[-15:] if num in d) / 15
        f30 = sum(1 for d in history[-30:] if num in d) / 30
        v1, v2 = f5 - f15, f15 - f30
        s += (v1 + (v1 - v2) * 0.5) * 2.0
        # Pair
        pair_b = sum(sum(1 for d in history[-50:] if num in d and n in d) for n in last)
        s += pair_b / max(1, len(last) * 50) * 3.0
        s += cycle_scores.get(num, 0) * 2.0
        s += context_scores.get(num, 0) / max(1, max_ctx) * 3.0
        # Correlation
        corr_b = sum(cond_prob.get(prev, {}).get(num, 0) for prev in last)
        s += corr_b * 0.05
        # V21 features
        s += triplet_scores.get(num, 0) / max(1, max_trip) * 2.5
        s += seq_pattern_scores.get(num, 0) / max(1, max_seq) * 3.0
        s += entropy_scores.get(num, 0) * 1.5 if n_draws >= 60 else 0
        
        scores[num] = s
    
    # Select with constraint validation
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    pool = [n for n, _ in ranked[:20]]
    
    c = constraints
    best_combo = None
    best_score = -float('inf')
    for combo in combinations(pool[:16], pick_count):
        s_val = sum(combo)
        if s_val < c['sum_lo'] or s_val > c['sum_hi']:
            continue
        odd = sum(1 for x in combo if x % 2 == 1)
        if odd < c['odd_lo'] or odd > c['odd_hi']:
            continue
        mid = max_number // 2
        high = sum(1 for x in combo if x > mid)
        if high < c['high_lo'] or high > c['high_hi']:
            continue
        rng = max(combo) - min(combo)
        if rng < c['range_lo'] or rng > c['range_hi']:
            continue
        cs = sum(scores[n] for n in combo)
        if cs > best_score:
            best_score = cs
            best_combo = sorted(combo)
    
    return best_combo if best_combo else sorted(pool[:pick_count])


def learn_constraints(data, max_number, pick_count):
    """Learn constraints from data."""
    sums = [sum(d) for d in data]
    odd_counts = [sum(1 for x in d if x % 2 == 1) for d in data]
    mid = max_number // 2
    high_counts = [sum(1 for x in d if x > mid) for d in data]
    ranges_vals = [max(d) - min(d) for d in data]
    return {
        'sum_lo': int(np.percentile(sums, 2.5)),
        'sum_hi': int(np.percentile(sums, 97.5)),
        'odd_lo': max(0, int(np.percentile(odd_counts, 5))),
        'odd_hi': min(pick_count, int(np.percentile(odd_counts, 95))),
        'high_lo': max(0, int(np.percentile(high_counts, 5))),
        'high_hi': min(pick_count, int(np.percentile(high_counts, 95))),
        'range_lo': int(np.percentile(ranges_vals, 5)),
        'range_hi': int(np.percentile(ranges_vals, 95)),
    }


def run_full_backtest(lottery_type='mega'):
    """Walk-forward backtest V21 scoring over ALL draws."""
    
    if lottery_type == 'mega':
        all_data = get_mega645_numbers()
        max_num, pick = 45, 6
    else:
        all_data = get_power655_numbers()
        all_data = [d[:6] for d in all_data]
        max_num, pick = 55, 6
    
    total_draws = len(all_data)
    min_train = 70
    
    test_indices = list(range(min_train, total_draws - 1))
    total_tests = len(test_indices)
    
    print(f"\n{'='*70}")
    print(f"  V21 FULL BACKTEST - {lottery_type.upper()} ({max_num}/{pick})")
    print(f"  Total draws: {total_draws} | Tests: {total_tests}")
    print(f"  V21 Scoring: Freq+Gap+KNN+Ngram+Cycle+Context+Corr+Triplet+SeqPat+Entropy")
    print(f"{'='*70}\n")
    
    primary_matches = []
    start_time = time.time()
    
    # Pre-learn constraints from ALL data
    constraints = learn_constraints(all_data, max_num, pick)
    
    for test_num, train_end in enumerate(test_indices):
        train_data = all_data[:train_end + 1]
        actual = set(all_data[train_end + 1])
        
        try:
            predicted = score_and_predict(train_data, max_num, pick, constraints)
            primary_match = len(set(predicted) & actual)
            primary_matches.append(primary_match)
            
            if primary_match >= 4:
                hits = set(predicted) & actual
                print(f"  *** Draw #{train_end+1}: {primary_match}/6! "
                      f"Pred={predicted} Actual={sorted(actual)} Hit={sorted(hits)} ***")
            
        except Exception as e:
            primary_matches.append(0)
            if test_num < 3:
                print(f"  ERROR at draw {train_end+1}: {e}")
        
        if (test_num + 1) % 20 == 0 or test_num == 0:
            elapsed = time.time() - start_time
            avg = np.mean(primary_matches)
            eta = (elapsed / (test_num + 1)) * (total_tests - test_num - 1)
            p3 = sum(1 for m in primary_matches if m >= 3) / len(primary_matches) * 100
            print(f"  [{test_num+1}/{total_tests}] avg={avg:.3f}/6, 3+={p3:.1f}%, "
                  f"{elapsed:.0f}s, ETA={eta:.0f}s")
    
    elapsed = time.time() - start_time
    
    # ====== RESULTS ======
    p_dist = Counter(primary_matches)
    random_expected = pick**2 / max_num
    avg_match = np.mean(primary_matches)
    
    print(f"\n{'='*70}")
    print(f"  V21 BACKTEST KET QUA — {lottery_type.upper()}")
    print(f"  Tests: {total_tests} | Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}\n")
    
    print(f"  Trung binh:  {avg_match:.4f}/6")
    print(f"  Max:         {max(primary_matches)}/6")
    print(f"  Random TB:   {random_expected:.3f}/6")
    print(f"  vs Random:   +{((avg_match/random_expected)-1)*100:.1f}%\n")
    
    print(f"  === PHAN BO SO TRUNG ===")
    print(f"  {'Trung':<8} {'So lan':>8} {'Ty le':>8}  {'Do thi'}")
    print(f"  {'-'*50}")
    for k in range(7):
        count = p_dist.get(k, 0)
        pct = count / total_tests * 100
        bar = '#' * int(pct * 2)
        label = ['MISS','','','Good','Very Good','Excellent','JACKPOT'][k]
        print(f"  {k}/6      {count:>6}   {pct:>6.2f}%  {bar}  {label}")
    
    print(f"\n  === TY LE TRUNG (cumulative) ===")
    for threshold in [6, 5, 4, 3, 2]:
        hits = sum(1 for m in primary_matches if m >= threshold)
        pct = hits / total_tests * 100
        print(f"    >= {threshold}/6: {hits:>5} / {total_tests} = {pct:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"  DONE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    lottery_type = sys.argv[1] if len(sys.argv) > 1 else 'mega'
    run_full_backtest(lottery_type)
