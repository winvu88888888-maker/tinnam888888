"""
DEEP MINE V6 — FULL SPECTRUM: EVERY DRAW FROM DAY 1
====================================================
Kiểm tra TOÀN BỘ 1486 kỳ từ kỳ đầu, chia thành micro-periods.
Tìm LỖ HỔNG bằng cách so sánh CÁC GIAI ĐOẠN với nhau.

PHƯƠNG PHÁP:
- Chia data thành 10 giai đoạn ~150 kỳ mỗi giai đoạn
- Mỗi giai đoạn: chạy 10+ tests → tìm anomalies
- So sánh giữa các giai đoạn → nếu pattern ỔN ĐỊNH = lỗ hổng thật
- Tìm REGIME CHANGES: máy thay đổi khi nào?
- Digit transition per epoch
- Sum mod 7 per epoch (confirmed from V5)
- Machine behavior drift detection
"""
import sys, os, math, json, time
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.data_manager import get_mega645_numbers, get_mega645_all


def analyze():
    data = get_mega645_numbers()
    all_rows = get_mega645_all()
    dates = [r['draw_date'] for r in all_rows]
    N = len(data)
    MAX = 45
    PICK = 6
    base_p = PICK / MAX
    t0 = time.time()

    # Chia thành 10 epochs
    n_epochs = 10
    epoch_size = N // n_epochs
    epochs = []
    for e in range(n_epochs):
        start = e * epoch_size
        end = (e + 1) * epoch_size if e < n_epochs - 1 else N
        epochs.append({
            'idx': e,
            'start': start, 'end': end,
            'size': end - start,
            'date_start': dates[start],
            'date_end': dates[end - 1],
            'data': data[start:end],
        })

    print("=" * 80)
    print("  🔍 DEEP MINE V6 — FULL SPECTRUM: EVERY DRAW FROM DAY 1")
    print(f"  {N} draws | {n_epochs} epochs × ~{epoch_size} draws each")
    print("=" * 80)
    for e in epochs:
        print(f"  Epoch {e['idx']}: {e['date_start']} → {e['date_end']} ({e['size']} draws)")

    findings = []

    # ==================================================================
    # ANALYSIS 1: FREQUENCY PER NUMBER PER EPOCH
    # Nếu máy hoàn hảo: mỗi epoch có frequency ~giống nhau
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 1: Per-Number Frequency Stability Across All Epochs")
    print(f"{'━'*80}")

    num_freqs = {num: [] for num in range(1, MAX+1)}
    for e in epochs:
        f = Counter()
        for d in e['data']:
            for x in d[:PICK]:
                f[x] += 1
        for num in range(1, MAX+1):
            rate = f.get(num, 0) / e['size']
            num_freqs[num].append(rate)

    # Chi-squared heterogeneity test per number
    unstable_numbers = []
    for num in range(1, MAX+1):
        rates = num_freqs[num]
        # Test: are rates significantly different across epochs?
        total_appearances = sum(int(r * epochs[i]['size']) for i, r in enumerate(rates))
        overall_rate = total_appearances / N

        chi2 = 0
        for i, e in enumerate(epochs):
            obs = int(rates[i] * e['size'])
            exp = overall_rate * e['size']
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp

        df = n_epochs - 1
        p = 1 - stats.chi2.cdf(chi2, df)
        if p < 0.01:
            # Find which epochs are anomalous
            max_epoch = np.argmax(rates)
            min_epoch = np.argmin(rates)
            unstable_numbers.append({
                'number': num, 'chi2': round(chi2, 2), 'p': round(p, 4),
                'rates': [round(r, 3) for r in rates],
                'max_epoch': max_epoch, 'max_rate': round(rates[max_epoch], 3),
                'min_epoch': min_epoch, 'min_rate': round(rates[min_epoch], 3),
            })

    unstable_numbers.sort(key=lambda x: x['chi2'], reverse=True)
    expected_fails = MAX * 0.01
    print(f"\n  Numbers with UNSTABLE frequency (p<0.01): {len(unstable_numbers)} "
          f"(expected by chance: {expected_fails:.1f})")

    if len(unstable_numbers) > expected_fails * 2:
        print(f"  🔴 MORE UNSTABLE NUMBERS THAN EXPECTED!")
    for u in unstable_numbers[:10]:
        print(f"    #{u['number']:2d}: χ²={u['chi2']:.1f}, p={u['p']:.4f} | "
              f"peak: epoch {u['max_epoch']}({u['max_rate']:.1%}), "
              f"low: epoch {u['min_epoch']}({u['min_rate']:.1%})")
        findings.append({'type': 'freq_instability', 'strength': round(u['chi2']/20, 2), **u})

    # ==================================================================
    # ANALYSIS 2: SUM MOD 7 PER EPOCH (V5 found p=0.0005)
    # Is it consistent across ALL epochs?
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 2: Sum Mod 7 Transition — Per Epoch Deep Dive")
    print(f"{'━'*80}")

    for mod in [7]:
        print(f"\n  === Sum mod {mod} ===")
        epoch_chi2s = []
        for e in epochs:
            sums = [sum(d[:PICK]) for d in e['data']]
            seq_mod = [s % mod for s in sums]
            trans = defaultdict(Counter)
            for i in range(len(seq_mod) - 1):
                trans[seq_mod[i]][seq_mod[i+1]] += 1

            chi2 = 0
            for state, next_counts in trans.items():
                total = sum(next_counts.values())
                for ns in range(mod):
                    obs = next_counts.get(ns, 0)
                    exp = total / mod
                    if exp > 1:
                        chi2 += (obs - exp) ** 2 / exp

            df = mod * (mod - 1)
            p = 1 - stats.chi2.cdf(chi2, df)
            epoch_chi2s.append((e['idx'], chi2, p))
            status = "🔴" if p < 0.05 else "✅"
            print(f"    Epoch {e['idx']} ({e['date_start'][:7]}): χ²={chi2:.1f}, "
                  f"df={df}, p={p:.4f} {status}")

        # Which epochs have the structure?
        sig_epochs = [(e, c, p) for e, c, p in epoch_chi2s if p < 0.05]
        print(f"\n    Significant epochs: {len(sig_epochs)}/{n_epochs}")
        if sig_epochs:
            findings.append({
                'type': 'sum_mod7_structure',
                'significant_epochs': len(sig_epochs),
                'epoch_details': [(e, round(c, 1), round(p, 4)) for e, c, p in sig_epochs],
                'strength': round(len(sig_epochs) / n_epochs * 5, 2),
            })

    # ==================================================================
    # ANALYSIS 3: DIGIT TRANSITIONS PER EPOCH
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 3: Digit Transition Patterns — Per Epoch")
    print(f"{'━'*80}")

    # For each epoch, compute digit transitions at pos 1 and pos 6
    stable_digit_trans = []
    for pos_idx, pos_label in [(0, 'Pos1(smallest)'), (5, 'Pos6(largest)')]:
        print(f"\n  === {pos_label} ===")
        
        # Collect transitions per epoch
        epoch_trans = []
        for e in epochs:
            sorted_data = [sorted(d[:PICK]) for d in e['data']]
            digit_trans = defaultdict(Counter)
            for i in range(len(sorted_data) - 1):
                prev_d = sorted_data[i][pos_idx] % 10
                next_d = sorted_data[i + 1][pos_idx] % 10
                digit_trans[prev_d][next_d] += 1
            epoch_trans.append(digit_trans)

        # Find transitions that are consistently anomalous across epochs
        for from_d in range(10):
            for to_d in range(10):
                rates = []
                for e_idx, dt in enumerate(epoch_trans):
                    total = sum(dt[from_d].values())
                    if total < 5:
                        rates.append(None)
                        continue
                    rate = dt[from_d].get(to_d, 0) / total
                    rates.append(rate)

                valid_rates = [r for r in rates if r is not None]
                if len(valid_rates) < 7:
                    continue

                avg_rate = np.mean(valid_rates)
                # Is this transition consistently above expected (~10%)?
                expected = 0.10  # rough
                if avg_rate > expected * 1.5:
                    consistency = sum(1 for r in valid_rates if r > expected * 1.2) / len(valid_rates)
                    if consistency > 0.6:  # Present in >60% of epochs
                        stable_digit_trans.append({
                            'pos': pos_label, 'from': from_d, 'to': to_d,
                            'avg_rate': round(avg_rate, 3),
                            'rates_per_epoch': [round(r, 3) if r else None for r in rates],
                            'consistency': round(consistency, 2),
                        })
                        print(f"    digit {from_d}→{to_d}: avg={avg_rate:.1%}, "
                              f"consistent in {consistency:.0%} of epochs")

    if stable_digit_trans:
        findings.append({
            'type': 'stable_digit_transitions',
            'count': len(stable_digit_trans),
            'details': stable_digit_trans,
            'strength': round(len(stable_digit_trans) * 0.5, 2),
        })

    # ==================================================================
    # ANALYSIS 4: REGIME DETECTION — When did the machine change?
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 4: Regime Detection — Machine Change Points")
    print(f"{'━'*80}")

    # Feature per draw: [sum, range, odd_count, repeat_from_prev, max_gap_in_draw]
    features = []
    for i, d in enumerate(data):
        sd = sorted(d[:PICK])
        s = sum(sd)
        rng = sd[-1] - sd[0]
        odd = sum(1 for x in sd if x % 2 == 1)
        rep = len(set(sd) & set(data[i-1][:PICK])) if i > 0 else 0
        features.append([s, rng, odd, rep])

    features = np.array(features)

    # Sliding window comparison: compare consecutive windows
    window = 50
    change_scores = []
    for i in range(window, N - window):
        before = features[i-window:i]
        after = features[i:i+window]

        # Multi-variate comparison
        total_z = 0
        for col in range(features.shape[1]):
            t, p = stats.ttest_ind(before[:, col], after[:, col])
            total_z += abs(t)

        change_scores.append((i, total_z))

    # Find top change points
    change_scores.sort(key=lambda x: -x[1])
    print(f"\n  Top regime change points:")
    change_points = []
    # Deduplicate (keep only 1 per 50-draw region)
    used = set()
    for idx, score in change_scores:
        if any(abs(idx - u) < 50 for u in used):
            continue
        used.add(idx)
        date = dates[idx] if idx < len(dates) else '?'
        print(f"    Draw {idx} ({date}): change_score={score:.2f}")
        change_points.append({'draw': idx, 'date': date, 'score': round(score, 2)})
        if len(change_points) >= 8:
            break

    if change_points:
        findings.append({
            'type': 'regime_change',
            'details': change_points,
            'strength': round(change_points[0]['score'] / 10, 2),
        })

    # ==================================================================
    # ANALYSIS 5: PER-EPOCH CONDITIONAL PROBABILITY
    # P(num in draw N+1 | num in draw N) — should = base_p
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 5: Conditional P(repeat) Per Epoch")
    print(f"{'━'*80}")

    for num in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        rates = []
        for e in epochs:
            appeared = 0
            repeated = 0
            for i in range(1, len(e['data'])):
                if num in e['data'][i-1][:PICK]:
                    appeared += 1
                    if num in e['data'][i][:PICK]:
                        repeated += 1
            rate = repeated / max(appeared, 1)
            rates.append(rate)

        avg = np.mean(rates)
        std = np.std(rates)
        z = (avg - base_p) / (std / math.sqrt(len(rates))) if std > 0 else 0
        if abs(z) > 2:
            print(f"  #{num:2d}: avg_repeat={avg:.3f} vs expected {base_p:.3f}, z={z:+.2f} ⚠️")
            findings.append({
                'type': 'conditional_repeat',
                'number': num, 'avg_rate': round(avg, 4),
                'z_score': round(z, 2), 'strength': round(abs(z) / 2, 2),
            })

    # ==================================================================
    # ANALYSIS 6: EXHAUSTIVE PAIR STABILITY (ALL 990 pairs)
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 6: Pair Co-occurrence Stability (all 990 pairs)")
    print(f"{'━'*80}")

    exp_pair_rate = PICK * (PICK - 1) / (MAX * (MAX - 1))
    n_stable_pairs = 0
    top_stable_pairs = []

    for a in range(1, MAX+1):
        for b in range(a+1, MAX+1):
            rates = []
            for e in epochs:
                pair_count = sum(1 for d in e['data']
                                 if a in d[:PICK] and b in d[:PICK])
                rates.append(pair_count / e['size'])

            avg = np.mean(rates)
            z = (avg - exp_pair_rate) / max(np.std(rates) / math.sqrt(len(rates)), 0.001)
            consistency = sum(1 for r in rates if r > exp_pair_rate) / len(rates)

            if z > 2.5 and consistency > 0.7:
                n_stable_pairs += 1
                total = sum(int(r * epochs[i]['size']) for i, r in enumerate(rates))
                top_stable_pairs.append({
                    'pair': (a, b), 'avg_rate': round(avg, 4),
                    'z': round(z, 2), 'consistency': round(consistency, 2),
                    'total': total,
                })

    top_stable_pairs.sort(key=lambda x: -x['z'])
    expected_stable = 990 * 0.005  # ~5 by chance
    print(f"\n  Stable pairs (z>2.5, consistency>70%): {n_stable_pairs} "
          f"(expected by chance: ~{expected_stable:.0f})")
    if n_stable_pairs > expected_stable * 2:
        print(f"  🔴 MORE STABLE PAIRS THAN EXPECTED!")

    for p in top_stable_pairs[:10]:
        print(f"    ({p['pair'][0]:2d},{p['pair'][1]:2d}): avg={p['avg_rate']:.4f} "
              f"vs exp={exp_pair_rate:.4f}, z={p['z']:+.2f}, "
              f"consistent={p['consistency']:.0%} ({p['total']} times)")
        findings.append({'type': 'stable_pair', 'strength': round(p['z'] / 3, 2), **p})

    # ==================================================================
    # ANALYSIS 7: POSITION VALUE DRIFT ACROSS EPOCHS
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 7: Position Value Drift")
    print(f"{'━'*80}")

    for pos in range(PICK):
        means = []
        for e in epochs:
            vals = [sorted(d[:PICK])[pos] for d in e['data']]
            means.append(np.mean(vals))

        slope, intercept, r, p, se = stats.linregress(range(len(means)), means)
        z = slope / max(se, 0.001)
        if abs(z) > 2:
            direction = "INCREASING" if slope > 0 else "DECREASING"
            print(f"  Position {pos+1}: slope={slope:.4f}/epoch, z={z:+.2f} → {direction} ⚠️")
            print(f"    Values: {[f'{m:.1f}' for m in means]}")
            findings.append({
                'type': 'position_drift',
                'position': pos + 1, 'slope': round(slope, 4),
                'z_score': round(z, 2), 'strength': round(abs(z) / 2, 2),
            })

    # ==================================================================
    # ANALYSIS 8: REPEAT PATTERN EVOLUTION
    # How many numbers repeat from draw N to N+1, per epoch
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 8: Repeat Rate Evolution Across Epochs")
    print(f"{'━'*80}")

    repeat_rates = []
    for e in epochs:
        repeats = []
        for i in range(1, len(e['data'])):
            prev = set(e['data'][i-1][:PICK])
            curr = set(e['data'][i][:PICK])
            repeats.append(len(prev & curr))
        avg = np.mean(repeats)
        repeat_rates.append(avg)
        print(f"  Epoch {e['idx']} ({e['date_start'][:7]}): avg_repeat={avg:.3f}")

    slope, _, r, p, se = stats.linregress(range(len(repeat_rates)), repeat_rates)
    z = slope / max(se, 0.001)
    print(f"\n  Repeat rate trend: slope={slope:.5f}/epoch, z={z:+.2f}")
    if abs(z) > 2:
        print(f"  🔴 Repeat rate is CHANGING over time!")
        findings.append({
            'type': 'repeat_drift', 'slope': round(slope, 5),
            'z_score': round(z, 2), 'strength': round(abs(z) / 2, 2),
        })

    # ==================================================================
    # ANALYSIS 9: FIRST DRAW vs RECENT — Machine Aging
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 9: Machine Aging — First 200 vs Last 200 draws")
    print(f"{'━'*80}")

    first200 = data[:200]
    last200 = data[-200:]

    metrics = {
        'sum': ([sum(d[:PICK]) for d in first200], [sum(d[:PICK]) for d in last200]),
        'range': ([max(d[:PICK])-min(d[:PICK]) for d in first200],
                  [max(d[:PICK])-min(d[:PICK]) for d in last200]),
        'odd_count': ([sum(1 for x in d[:PICK] if x%2==1) for d in first200],
                      [sum(1 for x in d[:PICK] if x%2==1) for d in last200]),
    }

    # Per-number frequency comparison
    freq_first = Counter()
    for d in first200:
        for x in d[:PICK]:
            freq_first[x] += 1
    freq_last = Counter()
    for d in last200:
        for x in d[:PICK]:
            freq_last[x] += 1

    for name, (v1, v2) in metrics.items():
        t, p = stats.ttest_ind(v1, v2)
        status = "🔴" if p < 0.05 else "✅"
        print(f"  {name:12s}: first200={np.mean(v1):.2f}, last200={np.mean(v2):.2f}, "
              f"t={t:.3f}, p={p:.4f} {status}")
        if p < 0.05:
            findings.append({
                'type': 'machine_aging', 'metric': name,
                'first': round(np.mean(v1), 2), 'last': round(np.mean(v2), 2),
                'p_value': round(p, 4), 'strength': round(abs(t) / 2, 2),
            })

    # Per-number: which numbers changed most?
    print(f"\n  Numbers with biggest frequency shift (first200 vs last200):")
    shifted = []
    for num in range(1, MAX+1):
        r1 = freq_first.get(num, 0) / 200
        r2 = freq_last.get(num, 0) / 200
        z = (r2 - r1) / math.sqrt(base_p * (1 - base_p) * (1/200 + 1/200))
        if abs(z) > 2:
            shifted.append((num, r1, r2, z))

    shifted.sort(key=lambda x: -abs(x[3]))
    for num, r1, r2, z in shifted[:10]:
        direction = "↑ INCREASED" if z > 0 else "↓ DECREASED"
        print(f"    #{num:2d}: first={r1:.1%} → last={r2:.1%}, z={z:+.2f} {direction}")
        findings.append({
            'type': 'number_shift', 'number': num,
            'first_rate': round(r1, 4), 'last_rate': round(r2, 4),
            'z_score': round(z, 2), 'strength': round(abs(z) / 2, 2),
        })

    # ==================================================================
    # ANALYSIS 10: TRANSITION MATRIX STABILITY
    # ==================================================================
    print(f"\n{'━'*80}")
    print("  ANALYSIS 10: Transition Matrix — Epoch-Level Stability")
    print(f"{'━'*80}")

    # For top follow-pairs: is the transition stable across all epochs?
    # Build overall transition
    overall_follow = defaultdict(Counter)
    overall_appear = Counter()
    for i in range(N - 1):
        for p in data[i][:PICK]:
            overall_appear[p] += 1
            for nx in data[i+1][:PICK]:
                overall_follow[p][nx] += 1

    # Find top transitions overall
    top_trans = []
    for prev in range(1, MAX+1):
        if overall_appear[prev] < 100:
            continue
        for nxt in range(1, MAX+1):
            obs = overall_follow[prev].get(nxt, 0)
            rate = obs / overall_appear[prev]
            z = (rate - base_p) / math.sqrt(base_p * (1 - base_p) / overall_appear[prev])
            if z > 2.5:
                top_trans.append((prev, nxt, rate, z, obs))

    top_trans.sort(key=lambda x: -x[3])

    # Check stability of top transitions across epochs
    print(f"  Top transitions (z>2.5): {len(top_trans)}")
    stable_trans = []
    for prev, nxt, overall_rate, overall_z, total_obs in top_trans[:20]:
        epoch_rates = []
        for e in epochs:
            ep_appear = 0
            ep_follow = 0
            for i in range(len(e['data']) - 1):
                if prev in e['data'][i][:PICK]:
                    ep_appear += 1
                    if nxt in e['data'][i+1][:PICK]:
                        ep_follow += 1
            rate = ep_follow / max(ep_appear, 1)
            epoch_rates.append(rate)

        consistent = sum(1 for r in epoch_rates if r > base_p) / len(epoch_rates)
        if consistent >= 0.7:
            print(f"    {prev:2d}→{nxt:2d}: overall={overall_rate:.3f}(z={overall_z:+.2f}), "
                  f"consistent={consistent:.0%}, total={total_obs}")
            stable_trans.append({
                'type': 'stable_transition', 'from': prev, 'to': nxt,
                'overall_rate': round(overall_rate, 4), 'overall_z': round(overall_z, 2),
                'consistency': round(consistent, 2), 'total': total_obs,
                'strength': round(overall_z * consistent / 2, 2),
            })
            findings.append(stable_trans[-1])

    # ==================================================================
    # SUMMARY
    # ==================================================================
    elapsed = time.time() - t0

    print(f"\n{'='*80}")
    print(f"  📊 FULL SPECTRUM SUMMARY — {len(findings)} total findings")
    print(f"{'='*80}")

    types = Counter(f['type'] for f in findings)
    for t, c in types.most_common():
        print(f"    {t}: {c}")

    # Save
    output = {
        'version': '6.0 — Full Spectrum Analysis',
        'total_draws': N,
        'n_epochs': n_epochs,
        'epoch_size': epoch_size,
        'date_range': [dates[0], dates[-1]],
        'n_findings': len(findings),
        'findings_by_type': dict(types),
        'findings': findings,
        'elapsed_seconds': round(elapsed, 1),
    }

    path = os.path.join(os.path.dirname(__file__), 'models', 'full_spectrum_v6.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Saved to {path}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == '__main__':
    analyze()
