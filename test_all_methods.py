"""
ALL-METHODS 6/6 ATTACK — SINH TỒN
====================================
Mọi phương pháp có thể, tất cả đều backtest.

METHODS:
  A. SET-LEVEL: Predict 6 numbers as unordered set (not per-column)
     A1. Number-level scoring (freq + transition + co-occurrence + gap)
     A2. Template matching (find most similar historical state → next draw)
     A3. Co-occurrence graph (max-weight 6-clique approximation)

  B. MACHINE LEARNING:
     B1. Feature-based scoring (handcrafted features → weighted score)
     B2. Hot/cold ensemble (combine multiple time horizons)

  C. PATTERN EXPLOITATION:
     C1. Repeat pattern (numbers that repeat from recent draws)
     C2. Follow pattern (numbers that frequently follow current draw's numbers)
     C3. Complement pattern (fill gaps in recent coverage)

  D. HYBRID / META:
     D1. Multi-method vote (each method votes for 45 numbers → top-6)
     D2. Genetic-style refinement (mutate best combos)

  E. COVERAGE / WHEELING:
     E1. Balanced wheel (spread picks across range)
     E2. Key number wheel (anchor on high-confidence numbers)

ALL methods generate N sets (1,3,5,10,20) and are backtested for 6/6, ≥5, ≥4, ≥3.
"""
import sys, time, os, warnings, random
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from math import comb as mcomb
from scraper.data_manager import get_mega645_numbers

data = get_mega645_numbers()
total = len(data)
MAX_NUM = 45
sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" ALL-METHODS 6/6 ATTACK")
print(f"{'='*90}\n")

# ================================================================
# METHOD A1: NUMBER-LEVEL SCORING
# Score each number 1-45, pick top-6
# ================================================================
def method_number_score(history, N=10):
    """Score every number 1-45 using multiple signals, pick top-6."""
    scores = np.zeros(MAX_NUM+1)
    n = len(history)
    flat = [num for draw in history for num in draw]
    last = set(history[-1])

    # Frequency (multi-window)
    for w in [10, 20, 50, 100, 200]:
        ww = min(w, n)
        recent = [num for draw in history[-ww:] for num in draw]
        fc = Counter(recent)
        total_r = len(recent)
        for num, cnt in fc.items():
            scores[num] += cnt / total_r * (2 if w <= 20 else 1)

    # Transition: numbers that follow current draw's numbers
    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1
    for p in last:
        if p in follow:
            t = sum(follow[p].values())
            for nx, cnt in follow[p].most_common(10):
                scores[nx] += cnt / t * 3

    # Co-occurrence: pairs that appear together
    pair_freq = Counter()
    for h in history[-200:]:
        for p in combinations(sorted(h), 2):
            pair_freq[p] += 1

    # Gap/overdue
    last_seen = {}
    for i, draw in enumerate(history):
        for num in draw:
            last_seen[num] = i
    gap_avg = defaultdict(list)
    prev_idx = defaultdict(lambda: -1)
    for i, draw in enumerate(history):
        for num in draw:
            if prev_idx[num] >= 0:
                gap_avg[num].append(i - prev_idx[num])
            prev_idx[num] = i

    for num in range(1, MAX_NUM+1):
        if num in last_seen and len(gap_avg[num]) >= 3:
            mg = np.mean(gap_avg[num])
            cg = n - last_seen[num]
            if mg > 0:
                ratio = cg / mg
                if ratio > 1.3:
                    scores[num] += min((ratio-1)*2, 5)
                elif abs(ratio - 1) < 0.3:
                    scores[num] += 2

    # Recency bonus
    for i, draw in enumerate(history[-5:]):
        for num in draw:
            scores[num] += (i+1) * 0.3

    # Pick top candidates
    ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -scores[x])
    top_nums = ranking[:15]  # Top-15 numbers

    # Generate N combos from top-15
    combos = []
    # Combo 1: pure top-6
    combos.append(tuple(sorted(top_nums[:6])))

    # Generate more combos by mixing top candidates
    for _ in range(N * 3):
        # Weighted random sample from top-15
        weights = [scores[v] for v in top_nums]
        total_w = sum(weights)
        if total_w == 0: continue
        probs = [w/total_w for w in weights]
        chosen = list(np.random.choice(top_nums, size=6, replace=False, p=probs))
        combos.append(tuple(sorted(chosen)))

    # Deduplicate and pick top-N by score
    unique = list(set(combos))
    scored = [(c, sum(scores[v] for v in c)) for c in unique]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD A2: TEMPLATE MATCHING
# Find historical state most similar to current → use next draw
# ================================================================
def method_template_match(history, N=10):
    """Find past draws most similar to current state → predict next."""
    n = len(history)
    last = set(history[-1])
    last2 = set(history[-2]) if n >= 2 else set()
    last3 = set(history[-3]) if n >= 3 else set()

    # Features of current state
    curr_sum = sum(history[-1])
    curr_range = max(history[-1]) - min(history[-1])

    similarities = []
    for i in range(max(0, n-500), n-2):  # Search in last 500
        h = set(history[i])
        # Jaccard similarity with last 3 draws
        sim = len(h & last) * 3 + len(h & last2) * 2 + len(h & last3) * 1
        # Sum proximity
        sim += max(0, 5 - abs(sum(history[i]) - curr_sum) / 10)
        # Range proximity
        sim += max(0, 3 - abs((max(history[i])-min(history[i])) - curr_range) / 5)
        similarities.append((sim, i))

    similarities.sort(key=lambda x: -x[0])

    # Take next draws after top-K similar states
    candidates = []
    for _, idx in similarities[:N*2]:
        if idx + 1 < n:
            candidates.append(tuple(sorted(history[idx+1])))

    # Deduplicate
    unique = list(set(candidates))
    if len(unique) < N:
        # Fill with top scored combos
        unique.extend(method_number_score(history, N - len(unique)))

    return unique[:N]


# ================================================================
# METHOD A3: CO-OCCURRENCE GRAPH
# Build weighted graph, find dense subgraphs
# ================================================================
def method_cooccurrence_graph(history, N=10):
    """Build co-occurrence graph, find high-weight 6-subsets."""
    # Build graph
    pair_w = Counter()
    # Recency-weighted
    n = len(history)
    for i, draw in enumerate(history):
        w = 1 + (i / n)  # More recent = higher weight
        for p in combinations(sorted(draw), 2):
            pair_w[p] += w

    # Per-number score
    num_score = Counter()
    for (a, b), w in pair_w.items():
        num_score[a] += w
        num_score[b] += w

    # Top-20 numbers by total co-occurrence weight
    top20 = [v for v, _ in num_score.most_common(20)]

    # Generate all C(20,6) = 38760 subsets, score by sum of pair weights
    combos = []
    for combo in combinations(top20, 6):
        score = sum(pair_w.get(p, 0) for p in combinations(sorted(combo), 2))
        combos.append((combo, score))

    combos.sort(key=lambda x: -x[1])

    # Diversify: take top and skip similar combos
    selected = []
    for combo, score in combos:
        if len(selected) >= N: break
        cs = set(combo)
        # Check diversity (at least 2 numbers different from all selected)
        is_diverse = True
        for prev in selected:
            if len(cs & set(prev)) >= 5:  # Too similar
                is_diverse = False
                break
        if is_diverse:
            selected.append(combo)

    # Fill if needed
    while len(selected) < N and combos:
        c, _ = combos.pop(0)
        if c not in selected:
            selected.append(c)

    return selected[:N]


# ================================================================
# METHOD B1: FEATURE-BASED ML SCORING
# Engineer features per number, weight them
# ================================================================
def method_ml_features(history, N=10):
    """ML-inspired: engineer features per number, score, pick top-6."""
    n = len(history)
    flat_recent = [num for draw in history[-50:] for num in draw]
    last_draw = history[-1]

    features = {}
    for num in range(1, MAX_NUM+1):
        f = []
        # F1: frequency in last 10, 20, 50, 100 draws
        for w in [10, 20, 50, 100]:
            cnt = sum(1 for draw in history[-min(w,n):] if num in draw)
            f.append(cnt / min(w,n))

        # F2: gap since last seen
        gap = n
        for i in range(n-1, -1, -1):
            if num in history[i]:
                gap = n - 1 - i
                break
        f.append(1.0 / (gap + 1))

        # F3: transition from last draw
        trans_score = 0
        for p in last_draw:
            trans_count = sum(1 for i in range(n-1) if p in history[i] and num in history[i+1])
            p_count = sum(1 for draw in history[:-1] if p in draw)
            if p_count > 0:
                trans_score += trans_count / p_count
        f.append(trans_score / 6)

        # F4: in last draw?
        f.append(1.0 if num in last_draw else 0.0)

        # F5: average co-occurrence with last draw's numbers
        cooc = 0
        for p in last_draw:
            if p != num:
                pair = tuple(sorted([p, num]))
                cnt = sum(1 for draw in history[-100:] if p in draw and num in draw)
                cooc += cnt
        f.append(cooc / (6 * 100))

        # F6: position tendency (which column does this number tend to be in?)
        for pos in range(6):
            cnt = sum(1 for sd in sorted_draws[:n] if sd[pos] == num)
            f.append(cnt / n)

        features[num] = np.array(f)

    # Weighted scoring (learned-like weights)
    weights = np.array([
        3.0, 2.5, 2.0, 1.5,  # frequency windows
        2.0,                   # gap
        4.0,                   # transition
        1.0,                   # in last draw
        3.0,                   # co-occurrence
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # position tendency
    ])

    num_scores = {}
    for num, f in features.items():
        num_scores[num] = np.dot(f, weights)

    ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -num_scores[x])
    top15 = ranking[:15]

    # Generate diverse N combos
    combos = set()
    combos.add(tuple(sorted(top15[:6])))

    for _ in range(N * 5):
        ws = [num_scores[v] for v in top15]
        total_w = sum(ws)
        probs = [w/total_w for w in ws]
        chosen = list(np.random.choice(top15, size=6, replace=False, p=probs))
        combos.add(tuple(sorted(chosen)))

    combos = list(combos)
    scored = [(c, sum(num_scores[v] for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD C1: REPEAT PATTERN
# Numbers that repeated recently tend to repeat
# ================================================================
def method_repeat_pattern(history, N=10):
    """Exploit repeat patterns: numbers appearing in recent draws."""
    n = len(history)
    last3 = set()
    for h in history[-3:]: last3.update(h)
    last5 = set()
    for h in history[-5:]: last5.update(h)

    # Count appearances in last 5 draws
    cnt = Counter()
    for h in history[-5:]:
        for num in h: cnt[num] += 1

    # Candidates: numbers that appeared 2+ times in last 5
    hot = [v for v, c in cnt.most_common() if c >= 2]
    warm = [v for v, c in cnt.most_common() if c == 1]

    # Fill to at least 15
    pool = hot + warm[:15-len(hot)]
    if len(pool) < 6:
        # Add frequent overall
        all_freq = Counter()
        for h in history[-50:]:
            for num in h: all_freq[num] += 1
        for v, _ in all_freq.most_common():
            if v not in pool:
                pool.append(v)
            if len(pool) >= 15: break

    pool = pool[:15]

    combos = set()
    combos.add(tuple(sorted(pool[:6])))
    for _ in range(N * 5):
        chosen = sorted(random.sample(pool, 6))
        combos.add(tuple(chosen))

    combos = list(combos)
    scored = [(c, sum(cnt.get(v, 0) for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD C2: FOLLOW PATTERN
# Numbers that often follow current draw
# ================================================================
def method_follow_pattern(history, N=10):
    """After draw X, which 6 numbers are most likely next?"""
    n = len(history)
    last = history[-1]

    follow = Counter()
    for i in range(n-1):
        overlap = len(set(history[i]) & set(last))
        if overlap >= 3:  # Similar draw found
            for num in history[i+1]:
                follow[num] += overlap

    if not follow:
        # Fallback: any transition
        for i in range(n-1):
            for num in history[i+1]:
                follow[num] += 1

    top15 = [v for v, _ in follow.most_common(15)]
    if len(top15) < 6: top15 = list(range(1, 16))

    combos = set()
    combos.add(tuple(sorted(top15[:6])))
    for _ in range(N * 5):
        ws = [follow.get(v, 1) for v in top15]
        tw = sum(ws)
        probs = [w/tw for w in ws]
        chosen = list(np.random.choice(top15, size=6, replace=False, p=probs))
        combos.add(tuple(sorted(chosen)))

    combos = list(combos)
    scored = [(c, sum(follow.get(v, 0) for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD C3: COMPLEMENT (COLD NUMBERS)
# Numbers that HAVEN'T appeared → due for appearance
# ================================================================
def method_complement(history, N=10):
    """Overdue numbers: longest gap → pick 6."""
    n = len(history)
    last_seen = {}
    for i, draw in enumerate(history):
        for num in draw:
            last_seen[num] = i

    gaps = {}
    for num in range(1, MAX_NUM+1):
        gaps[num] = n - last_seen.get(num, 0)

    # Sort by gap (most overdue first)
    ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -gaps[x])
    top15 = ranking[:15]

    combos = set()
    combos.add(tuple(sorted(top15[:6])))
    for _ in range(N * 5):
        ws = [gaps[v] for v in top15]
        tw = sum(ws)
        probs = [w/tw for w in ws]
        chosen = list(np.random.choice(top15, size=6, replace=False, p=probs))
        combos.add(tuple(sorted(chosen)))

    combos = list(combos)
    scored = [(c, sum(gaps[v] for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD D1: MULTI-METHOD VOTE
# Each method scores 45 numbers → weighted vote → top-6
# ================================================================
def method_meta_vote(history, N=10):
    """Combine ALL method predictions into weighted vote."""
    votes = Counter()

    # Get predictions from each sub-method
    methods_results = [
        method_number_score(history, 3),
        method_template_match(history, 3),
        method_cooccurrence_graph(history, 3),
        method_repeat_pattern(history, 3),
        method_follow_pattern(history, 3),
        method_complement(history, 3),
    ]

    for setlist in methods_results:
        for combo in setlist:
            for num in combo:
                votes[num] += 1

    # Also add per-column top predictions
    n = len(history)
    sd = [sorted(h) for h in history]
    for pos in range(6):
        vals = [s[pos] for s in sd]
        lv = vals[-1]
        trans = defaultdict(Counter)
        for i in range(n-1): trans[vals[i]][vals[i+1]] += 1
        if lv in trans:
            for v, c in trans[lv].most_common(3):
                votes[v] += c * 2

    top15 = [v for v, _ in votes.most_common(15)]

    combos = set()
    combos.add(tuple(sorted(top15[:6])))
    for _ in range(N * 5):
        ws = [votes.get(v, 1) for v in top15]
        tw = sum(ws)
        probs = [w/tw for w in ws]
        chosen = list(np.random.choice(top15, size=6, replace=False, p=probs))
        combos.add(tuple(sorted(chosen)))

    combos = list(combos)
    scored = [(c, sum(votes.get(v, 0) for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD D2: GENETIC-STYLE REFINEMENT
# Start with top picks, mutate and select
# ================================================================
def method_genetic(history, N=10):
    """Genetic algorithm: evolve optimal combo."""
    n = len(history)

    # Fitness = historical match rate
    freq = Counter()
    for h in history[-100:]:
        for num in h: freq[num] += 1

    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1

    pair_freq = Counter()
    for h in history[-100:]:
        for p in combinations(sorted(h), 2):
            pair_freq[p] += 1

    last = set(history[-1])

    def fitness(combo):
        s = 0
        for num in combo:
            s += freq.get(num, 0) * 0.1
            for p in last:
                s += follow[p].get(num, 0) * 0.02
        for p in combinations(sorted(combo), 2):
            s += pair_freq.get(p, 0) * 0.05
        # Range penalty
        r = max(combo) - min(combo)
        if r < 20: s -= 2
        if r > 40: s -= 1
        return s

    # Initialize population
    top20 = [v for v, _ in freq.most_common(20)]
    population = []
    for _ in range(100):
        combo = tuple(sorted(random.sample(top20, 6)))
        population.append(combo)

    # Evolve
    for gen in range(50):
        scored = [(c, fitness(c)) for c in population]
        scored.sort(key=lambda x: -x[1])
        survivors = [c for c, _ in scored[:30]]

        # Crossover + mutate
        children = list(survivors)
        while len(children) < 100:
            p1, p2 = random.sample(survivors[:15], 2)
            pool = list(set(p1) | set(p2))
            if len(pool) >= 6:
                child = tuple(sorted(random.sample(pool, 6)))
            else:
                child = p1
            # Mutate: swap 1 number
            if random.random() < 0.3:
                child = list(child)
                idx = random.randint(0, 5)
                child[idx] = random.randint(1, MAX_NUM)
                child = tuple(sorted(set(child)))
                if len(child) == 6:
                    children.append(child)
                    continue
            children.append(child)
        population = children[:100]

    # Final selection
    scored = [(c, fitness(c)) for c in set(population)]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# METHOD E1: BALANCED WHEEL
# Spread picks across number range
# ================================================================
def method_balanced_wheel(history, N=10):
    """Wheeling system: pick key numbers, generate balanced combos."""
    n = len(history)
    # Pick 12 key numbers using frequency + transition
    freq = Counter()
    for h in history[-50:]:
        for num in h: freq[num] += 1

    key12 = [v for v, _ in freq.most_common(12)]

    # Generate all C(12,6) = 924 combos
    all_combos = list(combinations(key12, 6))

    # Score: sum of pair frequencies + transition score
    last = set(history[-1])
    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1

    pair_freq = Counter()
    for h in history[-100:]:
        for p in combinations(sorted(h), 2):
            pair_freq[p] += 1

    scored = []
    for combo in all_combos:
        s = sum(freq.get(v, 0) for v in combo)
        for p in combinations(sorted(combo), 2):
            s += pair_freq.get(p, 0) * 0.1
        for v in combo:
            for p in last:
                s += follow[p].get(v, 0) * 0.02
        scored.append((combo, s))

    scored.sort(key=lambda x: -x[1])

    # Diversified selection
    selected = []
    for combo, _ in scored:
        if len(selected) >= N: break
        cs = set(combo)
        if not selected or all(len(cs & set(p)) <= 4 for p in selected):
            selected.append(combo)

    # Fill if needed
    while len(selected) < N:
        c, _ = scored[len(selected)]
        selected.append(c)

    return selected[:N]


# ================================================================
# METHOD E2: KEY NUMBER ANCHOR
# Find 2-3 "almost certain" numbers, combine with candidates
# ================================================================
def method_key_anchor(history, N=10):
    """Anchor on high-confidence numbers, vary the rest."""
    n = len(history)
    freq = Counter()
    for h in history[-30:]:
        for num in h: freq[num] += 1

    follow = defaultdict(Counter)
    for i in range(n-1):
        for p in history[i]:
            for nx in history[i+1]:
                follow[p][nx] += 1

    # Score each number
    scores = Counter()
    for num in range(1, MAX_NUM+1):
        scores[num] = freq.get(num, 0) * 2
        for p in history[-1]:
            scores[num] += follow[p].get(num, 0) * 0.1

    # Top-3 = anchors, top-15 = pool for remaining 3
    ranking = sorted(range(1, MAX_NUM+1), key=lambda x: -scores[x])
    anchors = ranking[:3]
    pool = ranking[3:15]

    combos = set()
    # Generate combos: 3 anchors + 3 from pool
    for combo_rest in combinations(pool, 3):
        full = tuple(sorted(list(anchors) + list(combo_rest)))
        if len(set(full)) == 6:
            combos.add(full)

    combos = list(combos)
    scored = [(c, sum(scores[v] for v in c)) for c in combos]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


# ================================================================
# COLUMN-BASED (previous approach, as baseline)
# ================================================================
def method_column_based(history, N=10):
    """Previous per-column approach."""
    sd = [sorted(h) for h in history]
    n = len(sd)
    pos_cands = []
    for pos in range(6):
        vals = [s[pos] for s in sd]
        lv = vals[-1]
        scores = Counter()

        # Mode
        for w in [10, 20, 50]:
            seg = vals[-min(w,n):]
            for v, c in Counter(seg).most_common(4):
                scores[v] += c / len(seg) * 2

        # Transition
        trans = defaultdict(Counter)
        for i in range(n-1): trans[vals[i]][vals[i+1]] += 1
        if lv in trans:
            t = sum(trans[lv].values())
            for v, c in trans[lv].most_common(5):
                scores[v] += c / t * 5

        # Bigram
        if n >= 3:
            bg = defaultdict(Counter)
            for i in range(n-2): bg[(vals[i],vals[i+1])][vals[i+2]] += 1
            key = (vals[-2], vals[-1])
            if key in bg:
                t = sum(bg[key].values())
                for v, c in bg[key].most_common(4): scores[v] += c / t * 6

        # Last ±2
        for d in range(-2, 3):
            v = lv + d
            if 1 <= v <= MAX_NUM:
                scores[v] += 2.0 if d==0 else (1.5 if abs(d)==1 else 0.5)

        # Conditional
        if pos > 0:
            cond = defaultdict(Counter)
            for s in sd: cond[s[pos-1]][s[pos]] += 1
            pv = sd[-1][pos-1]
            if pv in cond:
                t = sum(cond[pv].values())
                for v, c in cond[pv].most_common(5): scores[v] += c / t * 4

        K = [3,5,7,7,5,3][pos]
        pos_cands.append([v for v, _ in scores.most_common(K)])

    # Cross product → valid
    valid = []
    for combo in iterproduct_gen(pos_cands):
        valid.append(combo)
        if len(valid) > 50000: break

    if not valid: return [tuple(sorted(history[-1]))] * N

    # Score
    freq = Counter()
    for h in history[-50:]:
        for num in h: freq[num] += 1
    scored = [(c, sum(freq.get(v,0) for v in c)) for c in valid]
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:N]]


def iterproduct_gen(lists, idx=0, current=[]):
    """Generate cross-product with strictly increasing constraint."""
    if idx == len(lists):
        yield tuple(current)
        return
    for v in lists[idx]:
        if idx == 0 or v > current[-1]:
            yield from iterproduct_gen(lists, idx+1, current + [v])


# ================================================================
# BACKTEST ALL METHODS
# ================================================================
np.random.seed(42)
random.seed(42)

ALL_METHODS = {
    'A1_NumScore':     method_number_score,
    'A2_Template':     method_template_match,
    'A3_CoocGraph':    method_cooccurrence_graph,
    'B1_MLFeatures':   method_ml_features,
    'C1_Repeat':       method_repeat_pattern,
    'C2_Follow':       method_follow_pattern,
    'C3_Complement':   method_complement,
    'D1_MetaVote':     method_meta_vote,
    'D2_Genetic':      method_genetic,
    'E1_Wheel':        method_balanced_wheel,
    'E2_KeyAnchor':    method_key_anchor,
    'F_ColumnBased':   method_column_based,
}

N_VALUES = [1, 3, 5, 10, 20]
START = 200
TESTED = total - START - 1

results = {}
for mname in ALL_METHODS:
    results[mname] = {}
    for N in N_VALUES:
        results[mname][N] = {'h6': 0, 'h5': 0, 'h4': 0, 'h3': 0, 'h2': 0}

print(f"  Testing {len(ALL_METHODS)} methods x {len(N_VALUES)} portfolio sizes")
print(f"  {TESTED} draws to backtest\n")

t0 = time.time()
for idx in range(START, total-1):
    history_raw = data[:idx+1]
    history_sorted = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]
    actual_set = set(actual)
    actual_tuple = tuple(actual)

    for mname, mfunc in ALL_METHODS.items():
        try:
            max_n = max(N_VALUES)
            if mname == 'F_ColumnBased':
                preds = mfunc(history_sorted, max_n)
            elif mname in ['D2_Genetic']:
                preds = mfunc(history_sorted, max_n)
            else:
                preds = mfunc(history_sorted, max_n)

            for N in N_VALUES:
                subset = preds[:N]
                best_match = 0
                hit6 = False
                for combo in subset:
                    m = len(set(combo) & actual_set)
                    best_match = max(best_match, m)
                    if set(combo) == actual_set: hit6 = True

                if hit6: results[mname][N]['h6'] += 1
                if best_match >= 5: results[mname][N]['h5'] += 1
                if best_match >= 4: results[mname][N]['h4'] += 1
                if best_match >= 3: results[mname][N]['h3'] += 1
                if best_match >= 2: results[mname][N]['h2'] += 1
        except Exception as e:
            pass

    done = idx - START + 1
    if done % 50 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        # Quick preview
        for mname in list(ALL_METHODS.keys())[:3]:
            r = results[mname][10]
            print(f"    {mname}: 6/6={r['h6']} >=4={r['h4']} >=3={r['h3']}")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS TABLES
# ================================================================
print(f"{'='*90}")
print(f" ALL-METHODS RESULTS — {TESTED} draws")
print(f"{'='*90}\n")

# 6/6 rate
print(f"  === 6/6 HIT RATE ===\n")
print(f"  {'Method':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<8}", end='')
print()
print(f"  {'-'*60}")
for mname in ALL_METHODS:
    print(f"  {mname:<16}", end='')
    for N in N_VALUES:
        rate = results[mname][N]['h6'] / TESTED * 100
        print(f" {rate:<8.3f}", end='')
    print()

# >=4/6
print(f"\n  === >=4/6 HIT RATE ===\n")
print(f"  {'Method':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<8}", end='')
print()
print(f"  {'-'*60}")
for mname in ALL_METHODS:
    print(f"  {mname:<16}", end='')
    for N in N_VALUES:
        rate = results[mname][N]['h4'] / TESTED * 100
        print(f" {rate:<8.2f}", end='')
    print()

# >=3/6
print(f"\n  === >=3/6 HIT RATE ===\n")
print(f"  {'Method':<16}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<8}", end='')
print()
print(f"  {'-'*60}")
for mname in ALL_METHODS:
    print(f"  {mname:<16}", end='')
    for N in N_VALUES:
        rate = results[mname][N]['h3'] / TESTED * 100
        print(f" {rate:<8.2f}", end='')
    print()

# Best method per N
print(f"\n  === BEST METHOD PER PORTFOLIO SIZE ===\n")
for N in N_VALUES:
    best_m = ""
    best_r = -1
    for mname in ALL_METHODS:
        r4 = results[mname][N]['h4']
        if r4 > best_r:
            best_r = r4
            best_m = mname
    h6 = results[best_m][N]['h6']
    h4 = results[best_m][N]['h4']
    h3 = results[best_m][N]['h3']
    print(f"  N={N:<3}: {best_m:<16} 6/6={h6} >=4/6={h4}({h4/TESTED*100:.2f}%) >=3/6={h3}({h3/TESTED*100:.1f}%)")

# Random comparison
random_6 = [N / mcomb(45,6) * 100 for N in N_VALUES]
random_4 = [N * mcomb(6,4) * mcomb(39,2) / mcomb(45,6) * 100 for N in N_VALUES]
print(f"\n  === RANDOM BASELINE ===\n")
print(f"  {'N':<5}", end='')
for N in N_VALUES: print(f" {'N='+str(N):<8}", end='')
print()
print(f"  6/6:  ", end='')
for r in random_6: print(f" {r:<8.6f}", end='')
print()
print(f"  >=4/6:", end='')
for r in random_4: print(f" {r:<8.4f}", end='')
print()

print(f"\n{'='*90}")
print(f" IMPROVEMENT VS RANDOM (>=4/6)")
print(f"{'='*90}\n")
for mname in ALL_METHODS:
    for N in N_VALUES:
        actual_rate = results[mname][N]['h4'] / TESTED * 100
        rand_rate = N * mcomb(6,4) * mcomb(39,2) / mcomb(45,6) * 100
        if actual_rate > 0:
            imp = actual_rate / rand_rate
            print(f"  {mname:<16} N={N:<3}: {actual_rate:.2f}% vs random {rand_rate:.4f}% = {imp:.0f}x")

print(f"\n{'='*90}")
