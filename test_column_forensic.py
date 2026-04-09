"""
DEEP FORENSIC PER-COLUMN VULNERABILITY ANALYSIS
=================================================
Goal: Find 100% accuracy method for EACH of 6 sorted positions.
Then combine 6 perfect columns → 1 perfect set.

40+ specialized methods per column including:
- Statistical: mode, median, MA, WMA, EMA, DEMA, quantile
- Pattern: n-grams (2,3,4,5), modular arithmetic, delta chains
- Cycle: autocorrelation, period detection, FFT harmonics
- Conditional: cross-position, day-of-week, streak-based
- Regime: volatility clustering, distribution shift, HMM-like
- Recurrence: lag-repeat, echo patterns, return timing
- Ensemble: vote, confidence-weighted, adaptive
- Mathematical: fibonacci-like, prime gaps, digit sum patterns
"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter, defaultdict
from scraper.data_manager import get_mega645_numbers, get_mega645_all

data = get_mega645_numbers()
all_records = get_mega645_all()
dates = [r['draw_date'] for r in all_records]
total = len(data)
MAX_NUM = 45

sorted_draws = [sorted(d[:6]) for d in data]

print(f"Data: {total} draws")
print(f"{'='*90}")
print(f" DEEP FORENSIC PER-COLUMN VULNERABILITY ANALYSIS")
print(f" 40+ methods x 6 positions — targeting 100% per column")
print(f"{'='*90}\n")

# ================================================================
# ALL PREDICTION METHODS (each returns 1 predicted value)
# ================================================================

# --- GROUP A: STATISTICAL CENTRAL TENDENCY ---
def m_mode(h, pos, w=50):
    vals = [x[pos] for x in h[-w:]]
    return Counter(vals).most_common(1)[0][0]

def m_mode_10(h, pos): return m_mode(h, pos, 10)
def m_mode_20(h, pos): return m_mode(h, pos, 20)
def m_mode_30(h, pos): return m_mode(h, pos, 30)
def m_mode_50(h, pos): return m_mode(h, pos, 50)
def m_mode_100(h, pos): return m_mode(h, pos, 100)
def m_mode_200(h, pos): return m_mode(h, pos, min(200, len(h)))
def m_mode_all(h, pos): return m_mode(h, pos, len(h))

def m_ma(h, pos, w=10):
    vals = [x[pos] for x in h[-w:]]
    return int(round(np.mean(vals)))
def m_ma5(h, pos): return m_ma(h, pos, 5)
def m_ma10(h, pos): return m_ma(h, pos, 10)
def m_ma20(h, pos): return m_ma(h, pos, 20)
def m_ma30(h, pos): return m_ma(h, pos, 30)
def m_ma50(h, pos): return m_ma(h, pos, 50)

def m_wma(h, pos, w=15):
    vals = [x[pos] for x in h[-w:]]
    weights = np.arange(1, len(vals)+1, dtype=float)
    return max(1, min(MAX_NUM, int(round(np.average(vals, weights=weights)))))
def m_wma10(h, pos): return m_wma(h, pos, 10)
def m_wma15(h, pos): return m_wma(h, pos, 15)
def m_wma20(h, pos): return m_wma(h, pos, 20)
def m_wma30(h, pos): return m_wma(h, pos, 30)

def m_ema(h, pos, alpha=0.3, w=30):
    vals = [x[pos] for x in h[-w:]]
    s = vals[0]
    for v in vals[1:]:
        s = alpha * v + (1-alpha) * s
    return max(1, min(MAX_NUM, int(round(s))))
def m_ema01(h, pos): return m_ema(h, pos, 0.1)
def m_ema02(h, pos): return m_ema(h, pos, 0.2)
def m_ema03(h, pos): return m_ema(h, pos, 0.3)
def m_ema05(h, pos): return m_ema(h, pos, 0.5)
def m_ema07(h, pos): return m_ema(h, pos, 0.7)

def m_dema(h, pos, alpha=0.3, w=30):
    """Double EMA — tracks trend better."""
    vals = [x[pos] for x in h[-w:]]
    s1 = vals[0]; s2 = vals[0]
    for v in vals[1:]:
        s1 = alpha * v + (1-alpha) * s1
        s2 = alpha * s1 + (1-alpha) * s2
    pred = 2*s1 - s2
    return max(1, min(MAX_NUM, int(round(pred))))

def m_median(h, pos, w=20):
    vals = [x[pos] for x in h[-w:]]
    return int(round(np.median(vals)))
def m_median10(h, pos): return m_median(h, pos, 10)
def m_median20(h, pos): return m_median(h, pos, 20)
def m_median50(h, pos): return m_median(h, pos, 50)

def m_last(h, pos):
    return h[-1][pos]

def m_last_minus1(h, pos):
    return h[-2][pos] if len(h) >= 2 else h[-1][pos]

def m_quantile(h, pos, q=0.5, w=30):
    vals = [x[pos] for x in h[-w:]]
    return max(1, min(MAX_NUM, int(round(np.quantile(vals, q)))))

# --- GROUP B: TRANSITION / MARKOV METHODS ---
def m_transition(h, pos):
    trans = defaultdict(Counter)
    for i in range(len(h)-1):
        trans[h[i][pos]][h[i+1][pos]] += 1
    lv = h[-1][pos]
    if lv in trans and trans[lv]:
        return trans[lv].most_common(1)[0][0]
    return lv

def m_transition_w(h, pos, w=100):
    """Windowed transition — only recent transitions."""
    start = max(0, len(h)-w)
    trans = defaultdict(Counter)
    for i in range(start, len(h)-1):
        trans[h[i][pos]][h[i+1][pos]] += 1
    lv = h[-1][pos]
    if lv in trans and trans[lv]:
        return trans[lv].most_common(1)[0][0]
    return lv

def m_bigram(h, pos):
    if len(h) < 3: return h[-1][pos]
    trans = defaultdict(Counter)
    for i in range(len(h)-2):
        key = (h[i][pos], h[i+1][pos])
        trans[key][h[i+2][pos]] += 1
    key = (h[-2][pos], h[-1][pos])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return m_transition(h, pos)

def m_trigram(h, pos):
    if len(h) < 4: return m_bigram(h, pos)
    trans = defaultdict(Counter)
    for i in range(len(h)-3):
        key = (h[i][pos], h[i+1][pos], h[i+2][pos])
        trans[key][h[i+3][pos]] += 1
    key = (h[-3][pos], h[-2][pos], h[-1][pos])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return m_bigram(h, pos)

def m_4gram(h, pos):
    if len(h) < 5: return m_trigram(h, pos)
    trans = defaultdict(Counter)
    for i in range(len(h)-4):
        key = (h[i][pos], h[i+1][pos], h[i+2][pos], h[i+3][pos])
        trans[key][h[i+4][pos]] += 1
    key = (h[-4][pos], h[-3][pos], h[-2][pos], h[-1][pos])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return m_trigram(h, pos)

def m_5gram(h, pos):
    if len(h) < 6: return m_4gram(h, pos)
    trans = defaultdict(Counter)
    for i in range(len(h)-5):
        key = tuple(h[j][pos] for j in range(i, i+5))
        trans[key][h[i+5][pos]] += 1
    key = tuple(h[j][pos] for j in range(len(h)-5, len(h)))
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return m_4gram(h, pos)

# --- GROUP C: DELTA / DIFFERENCE METHODS ---
def m_delta_mode(h, pos, w=50):
    """Most common delta (change between consecutive draws)."""
    vals = [x[pos] for x in h[-w:]]
    deltas = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    if not deltas: return vals[-1]
    d = Counter(deltas).most_common(1)[0][0]
    return max(1, min(MAX_NUM, vals[-1] + d))

def m_delta_last(h, pos):
    """Repeat last delta."""
    if len(h) < 3: return h[-1][pos]
    d = h[-1][pos] - h[-2][pos]
    return max(1, min(MAX_NUM, h[-1][pos] + d))

def m_delta2(h, pos):
    """Second-order delta (acceleration)."""
    if len(h) < 4: return m_delta_last(h, pos)
    d1 = h[-1][pos] - h[-2][pos]
    d2 = h[-2][pos] - h[-3][pos]
    dd = d1 - d2  # acceleration
    pred_delta = d1 + dd
    return max(1, min(MAX_NUM, h[-1][pos] + pred_delta))

def m_delta_bigram(h, pos, w=100):
    """Bigram on deltas: given last 2 deltas, predict next delta."""
    vals = [x[pos] for x in h[-w:]]
    deltas = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    if len(deltas) < 3: return m_delta_mode(h, pos)
    trans = defaultdict(Counter)
    for i in range(len(deltas)-2):
        key = (deltas[i], deltas[i+1])
        trans[key][deltas[i+2]] += 1
    key = (deltas[-2], deltas[-1])
    if key in trans and trans[key]:
        d = trans[key].most_common(1)[0][0]
    else:
        d = Counter(deltas).most_common(1)[0][0]
    return max(1, min(MAX_NUM, vals[-1] + d))

# --- GROUP D: REGRESSION / TREND ---
def m_linreg(h, pos, w=30):
    vals = [x[pos] for x in h[-w:]]
    x = np.arange(len(vals))
    c = np.polyfit(x, vals, 1)
    return max(1, min(MAX_NUM, int(round(c[0]*len(vals)+c[1]))))

def m_linreg50(h, pos): return m_linreg(h, pos, 50)

def m_quadreg(h, pos, w=30):
    """Quadratic regression."""
    vals = [x[pos] for x in h[-w:]]
    x = np.arange(len(vals))
    c = np.polyfit(x, vals, 2)
    pred = c[0]*len(vals)**2 + c[1]*len(vals) + c[2]
    return max(1, min(MAX_NUM, int(round(pred))))

def m_momentum(h, pos):
    v_s = np.mean([x[pos] for x in h[-5:]])
    v_l = np.mean([x[pos] for x in h[-30:]])
    pred = v_s + (v_s - v_l) * 0.5
    return max(1, min(MAX_NUM, int(round(pred))))

# --- GROUP E: CONDITIONAL / CROSS-POSITION ---
def m_conditional_prev(h, pos):
    """Predict pos based on previous position's value in SAME draw."""
    if pos == 0: return m_transition(h, pos)
    trans = defaultdict(Counter)
    for x in h:
        trans[x[pos-1]][x[pos]] += 1
    pv = h[-1][pos-1]
    if pv in trans and trans[pv]:
        return trans[pv].most_common(1)[0][0]
    return h[-1][pos]

def m_conditional_next(h, pos):
    """Predict pos based on NEXT position's recent value (from last draw)."""
    if pos >= 5: return m_transition(h, pos)
    trans = defaultdict(Counter)
    for x in h:
        trans[x[pos+1]][x[pos]] += 1
    nv = h[-1][pos+1]
    if nv in trans and trans[nv]:
        return trans[nv].most_common(1)[0][0]
    return h[-1][pos]

def m_conditional_sum(h, pos):
    """Given total sum of last draw, predict this position."""
    trans = defaultdict(Counter)
    for x in h:
        s = sum(x)
        trans[s][x[pos]] += 1
    ls = sum(h[-1])
    if ls in trans and trans[ls]:
        return trans[ls].most_common(1)[0][0]
    return h[-1][pos]

def m_conditional_range(h, pos):
    """Given range (max-min) of last draw, predict this position."""
    trans = defaultdict(Counter)
    for x in h:
        r = x[5] - x[0]
        trans[r][x[pos]] += 1
    lr = h[-1][5] - h[-1][0]
    if lr in trans and trans[lr]:
        return trans[lr].most_common(1)[0][0]
    return h[-1][pos]

def m_cross_2pos(h, pos):
    """Use 2 adjacent positions from last draw to predict."""
    if pos == 0:
        p1 = 0; p2 = 1
    elif pos == 5:
        p1 = 4; p2 = 5
    else:
        p1 = pos-1; p2 = pos+1
    trans = defaultdict(Counter)
    for i in range(len(h)-1):
        key = (h[i][p1], h[i][p2])
        trans[key][h[i+1][pos]] += 1
    key = (h[-1][p1], h[-1][p2])
    if key in trans and trans[key]:
        return trans[key].most_common(1)[0][0]
    return m_transition(h, pos)

# --- GROUP F: CYCLE / PERIODICITY DETECTION ---
def m_cycle_auto(h, pos, max_lag=50):
    """Autocorrelation-based cycle: find dominant period, project."""
    vals = [x[pos] for x in h[-200:]]
    if len(vals) < 30: return h[-1][pos]
    v = np.array(vals, dtype=float)
    v = v - v.mean()
    if v.std() == 0: return h[-1][pos]
    v = v / v.std()
    n = len(v)
    best_lag, best_corr = 2, 0
    for lag in range(2, min(max_lag, n//3)):
        corr = np.mean(v[:n-lag] * v[lag:])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_corr < 0.1: return m_mode_50(h, pos)
    # Project: value at t-period
    if len(h) > best_lag:
        return h[-best_lag][pos]
    return h[-1][pos]

def m_cycle_exact(h, pos):
    """Test periods 2-20: which period gives best exact repeat?"""
    vals = [x[pos] for x in h]
    n = len(vals)
    best_p, best_acc = 1, 0
    for p in range(2, min(21, n//3)):
        hits = sum(1 for i in range(p, n) if vals[i] == vals[i-p])
        acc = hits / (n-p)
        if acc > best_acc:
            best_acc = acc
            best_p = p
    return vals[-best_p] if best_p <= len(vals) else vals[-1]

def m_cycle_fft(h, pos):
    """FFT dominant frequency → project next value."""
    vals = np.array([x[pos] for x in h[-256:]], dtype=float)
    if len(vals) < 32: return h[-1][pos]
    v = vals - vals.mean()
    fft = np.fft.rfft(v)
    mag = np.abs(fft)
    mag[0] = 0  # skip DC
    if len(mag) < 3: return h[-1][pos]
    top_freq = np.argmax(mag[1:]) + 1
    period = len(v) / top_freq if top_freq > 0 else len(v)
    period = max(2, int(round(period)))
    if period <= len(h):
        return h[-period][pos]
    return h[-1][pos]

# --- GROUP G: GAP / OVERDUE / TIMING ---
def m_gap_overdue(h, pos):
    """Most overdue value at this position."""
    vals = [x[pos] for x in h]
    last_seen = {}
    for i, v in enumerate(vals):
        last_seen[v] = i
    n = len(vals)
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx:
            gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    best_v, best_r = vals[-1], 0
    for v in set(vals):
        if len(gap_data[v]) < 3: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        r = cg / mg if mg > 0 else 0
        if r > best_r:
            best_r = r
            best_v = v
    return best_v

def m_gap_expected(h, pos):
    """Value whose current gap is closest to its mean gap (due NOW)."""
    vals = [x[pos] for x in h]
    n = len(vals)
    last_seen = {}
    for i, v in enumerate(vals):
        last_seen[v] = i
    gap_data = defaultdict(list)
    prev_idx = {}
    for i, v in enumerate(vals):
        if v in prev_idx:
            gap_data[v].append(i - prev_idx[v])
        prev_idx[v] = i
    best_v, best_diff = vals[-1], 999
    for v in set(vals):
        if len(gap_data[v]) < 5: continue
        mg = np.mean(gap_data[v])
        cg = n - last_seen.get(v, 0)
        diff = abs(cg - mg)
        if diff < best_diff:
            best_diff = diff
            best_v = v
    return best_v

# --- GROUP H: LAG-REPEAT / ECHO PATTERNS ---
def m_lag_repeat(h, pos, lag=1):
    """Simply repeat value from `lag` draws ago."""
    idx = min(lag, len(h)-1)
    return h[-idx-1][pos] if idx < len(h) else h[-1][pos]

def m_lag2(h, pos): return m_lag_repeat(h, pos, 2)
def m_lag3(h, pos): return m_lag_repeat(h, pos, 3)
def m_lag5(h, pos): return m_lag_repeat(h, pos, 5)
def m_lag7(h, pos): return m_lag_repeat(h, pos, 7)
def m_lag10(h, pos): return m_lag_repeat(h, pos, 10)

def m_best_lag(h, pos, max_lag=30):
    """Find the lag with highest repeat rate, use it."""
    vals = [x[pos] for x in h]
    n = len(vals)
    best_lag, best_acc = 1, 0
    for lag in range(1, min(max_lag+1, n)):
        hits = sum(1 for i in range(lag, n) if vals[i] == vals[i-lag])
        acc = hits / (n-lag)
        if acc > best_acc:
            best_acc = acc
            best_lag = lag
    return vals[-best_lag] if best_lag <= len(vals) else vals[-1]

# --- GROUP I: KNN / SIMILARITY ---
def m_knn(h, pos, k=5):
    last = h[-1]
    sims = []
    for i in range(len(h)-2):
        sim = sum(1 for j in range(6) if abs(h[i][j]-last[j]) <= 2)
        sims.append((sim, i))
    sims.sort(key=lambda x: -x[0])
    preds = Counter()
    for sim, idx in sims[:k]:
        if idx+1 < len(h):
            preds[h[idx+1][pos]] += sim
    return preds.most_common(1)[0][0] if preds else last[pos]

def m_knn3(h, pos): return m_knn(h, pos, 3)
def m_knn10(h, pos): return m_knn(h, pos, 10)
def m_knn20(h, pos): return m_knn(h, pos, 20)

def m_knn_exact(h, pos, k=5):
    """KNN matching only this position ±1."""
    vals = [x[pos] for x in h]
    lv = vals[-1]
    preds = Counter()
    for i in range(len(vals)-1):
        if abs(vals[i] - lv) <= 1:
            preds[vals[i+1]] += 1
    return preds.most_common(1)[0][0] if preds else lv

def m_knn_delta(h, pos, k=5):
    """KNN on delta sequence: find similar delta patterns."""
    vals = [x[pos] for x in h[-100:]]
    if len(vals) < 10: return h[-1][pos]
    deltas = [vals[i+1]-vals[i] for i in range(len(vals)-1)]
    last_d = deltas[-3:] if len(deltas) >= 3 else deltas
    preds = Counter()
    for i in range(len(deltas)-len(last_d)-1):
        seg = deltas[i:i+len(last_d)]
        if seg == last_d and i+len(last_d) < len(deltas):
            next_d = deltas[i+len(last_d)]
            preds[vals[-1] + next_d] += 1
    if preds:
        v = preds.most_common(1)[0][0]
        return max(1, min(MAX_NUM, v))
    return m_delta_mode(h, pos)

# --- GROUP J: MODULAR / MATHEMATICAL PATTERNS ---
def m_mod_pattern(h, pos, m=3):
    """Track mod-m pattern of values, predict next mod class, then pick mode within."""
    vals = [x[pos] for x in h]
    mods = [v % m for v in vals]
    # Transition on mods
    trans = defaultdict(Counter)
    for i in range(len(mods)-1):
        trans[mods[i]][mods[i+1]] += 1
    lm = mods[-1]
    if lm in trans and trans[lm]:
        next_mod = trans[lm].most_common(1)[0][0]
    else:
        next_mod = Counter(mods).most_common(1)[0][0]
    # Pick most common value with this mod class in recent window
    recent = [v for v in vals[-50:] if v % m == next_mod]
    if recent:
        return Counter(recent).most_common(1)[0][0]
    return vals[-1]

def m_mod2(h, pos): return m_mod_pattern(h, pos, 2)
def m_mod3(h, pos): return m_mod_pattern(h, pos, 3)
def m_mod5(h, pos): return m_mod_pattern(h, pos, 5)
def m_mod7(h, pos): return m_mod_pattern(h, pos, 7)

def m_digit_sum(h, pos):
    """Track digit-sum pattern (e.g., 23→5, 7→7)."""
    vals = [x[pos] for x in h]
    dsums = [sum(int(c) for c in str(v)) for v in vals]
    trans = defaultdict(Counter)
    for i in range(len(dsums)-1):
        trans[dsums[i]][dsums[i+1]] += 1
    ld = dsums[-1]
    if ld in trans and trans[ld]:
        next_ds = trans[ld].most_common(1)[0][0]
    else:
        next_ds = Counter(dsums).most_common(1)[0][0]
    recent = [v for v in vals[-50:] if sum(int(c) for c in str(v)) == next_ds]
    if recent:
        return Counter(recent).most_common(1)[0][0]
    return vals[-1]

# --- GROUP K: STREAK / RUN ANALYSIS ---
def m_streak(h, pos):
    """If value is on a streak (repeated N times), predict continuation vs break."""
    vals = [x[pos] for x in h]
    n = len(vals)
    # Count current streak
    streak_len = 1
    for i in range(n-2, -1, -1):
        if vals[i] == vals[-1]:
            streak_len += 1
        else:
            break
    # Historical: what happens after streak of this length?
    preds = Counter()
    for i in range(n-1):
        sl = 1
        for j in range(i-1, -1, -1):
            if vals[j] == vals[i]:
                sl += 1
            else:
                break
        if sl == streak_len:
            preds[vals[i+1] if i+1 < n else vals[i]] += 1
    if preds:
        return preds.most_common(1)[0][0]
    return vals[-1]

def m_direction(h, pos):
    """Track up/down/same direction, predict next direction, then value."""
    vals = [x[pos] for x in h]
    if len(vals) < 3: return vals[-1]
    dirs = []
    for i in range(1, len(vals)):
        if vals[i] > vals[i-1]: dirs.append(1)
        elif vals[i] < vals[i-1]: dirs.append(-1)
        else: dirs.append(0)
    # Transition on direction
    trans = defaultdict(Counter)
    for i in range(len(dirs)-1):
        trans[dirs[i]][dirs[i+1]] += 1
    ld = dirs[-1]
    if ld in trans and trans[ld]:
        next_dir = trans[ld].most_common(1)[0][0]
    else:
        next_dir = 0
    # Given predicted direction, what's the most common delta?
    dir_deltas = defaultdict(list)
    for i in range(1, len(vals)):
        d = 1 if vals[i]>vals[i-1] else (-1 if vals[i]<vals[i-1] else 0)
        dir_deltas[d].append(vals[i]-vals[i-1])
    if next_dir in dir_deltas and dir_deltas[next_dir]:
        avg_d = int(round(np.median(dir_deltas[next_dir])))
    else:
        avg_d = 0
    return max(1, min(MAX_NUM, vals[-1] + avg_d))

# --- GROUP L: RANGE / BOUNDARY METHODS ---
def m_range_center(h, pos, w=15):
    vals = [x[pos] for x in h[-w:]]
    return int(round((min(vals)+max(vals))/2))

def m_range_bounce(h, pos, w=20):
    """If near boundary of recent range, predict bounce back."""
    vals = [x[pos] for x in h[-w:]]
    lo, hi = min(vals), max(vals)
    mid = (lo+hi)/2
    lv = vals[-1]
    if lv - lo < hi - lv:  # near bottom → bounce up
        return max(1, min(MAX_NUM, int(round(mid + (mid-lo)*0.3))))
    else:  # near top → bounce down
        return max(1, min(MAX_NUM, int(round(mid - (hi-mid)*0.3))))

# --- GROUP M: VOLATILITY / REGIME ---
def m_volatility_regime(h, pos, w=20):
    """Low vs high volatility regime → different prediction."""
    vals = [x[pos] for x in h[-w:]]
    std = np.std(vals)
    # Low vol → mode, high vol → MA
    if std < 3:
        return Counter(vals).most_common(1)[0][0]
    else:
        return int(round(np.mean(vals[-5:])))

def m_regime_switch(h, pos):
    """Detect if distribution shifted recently vs historically."""
    if len(h) < 60: return m_mode_50(h, pos)
    recent = [x[pos] for x in h[-20:]]
    older = [x[pos] for x in h[-60:-20]]
    rm = np.mean(recent)
    om = np.mean(older)
    # If shift detected, follow recent trend
    if abs(rm - om) > 2:
        return int(round(rm + (rm - om) * 0.3))
    return Counter(recent).most_common(1)[0][0]

# --- GROUP N: COMBINED / ENSEMBLE ---
def m_top2_transition(h, pos):
    trans = defaultdict(Counter)
    for i in range(len(h)-1):
        trans[h[i][pos]][h[i+1][pos]] += 1
    lv = h[-1][pos]
    if lv in trans and trans[lv]:
        top2 = trans[lv].most_common(2)
        freq = Counter(x[pos] for x in h[-50:])
        if len(top2) >= 2:
            if freq.get(top2[0][0],0) >= freq.get(top2[1][0],0):
                return top2[0][0]
            return top2[1][0]
        return top2[0][0]
    return lv

def m_vote3(h, pos):
    """Vote of mode_50 + transition + wma_15."""
    v1 = m_mode_50(h, pos)
    v2 = m_transition(h, pos)
    v3 = m_wma15(h, pos)
    c = Counter([v1, v2, v3])
    return c.most_common(1)[0][0]

def m_vote5(h, pos):
    """Vote of 5 methods."""
    preds = [
        m_mode_50(h, pos), m_transition(h, pos), m_wma15(h, pos),
        m_bigram(h, pos), m_knn(h, pos, 5)
    ]
    return Counter(preds).most_common(1)[0][0]

def m_weighted_vote(h, pos):
    """Weighted vote — transition gets 3x, mode 2x, rest 1x."""
    sc = Counter()
    sc[m_transition(h, pos)] += 3
    sc[m_mode_50(h, pos)] += 2
    sc[m_wma15(h, pos)] += 1
    sc[m_bigram(h, pos)] += 2
    sc[m_conditional_prev(h, pos)] += 2
    return sc.most_common(1)[0][0]


# ================================================================
# REGISTER ALL METHODS
# ================================================================
METHODS = {
    # Group A: Statistical
    'mode_10': m_mode_10, 'mode_20': m_mode_20, 'mode_30': m_mode_30,
    'mode_50': m_mode_50, 'mode_100': m_mode_100, 'mode_200': m_mode_200,
    'mode_all': m_mode_all,
    'ma5': m_ma5, 'ma10': m_ma10, 'ma20': m_ma20, 'ma30': m_ma30, 'ma50': m_ma50,
    'wma10': m_wma10, 'wma15': m_wma15, 'wma20': m_wma20, 'wma30': m_wma30,
    'ema01': m_ema01, 'ema02': m_ema02, 'ema03': m_ema03, 'ema05': m_ema05, 'ema07': m_ema07,
    'dema': m_dema,
    'median10': m_median10, 'median20': m_median20, 'median50': m_median50,
    'last': m_last, 'last_minus1': m_last_minus1,
    # Group B: Transition/Markov
    'transition': m_transition, 'transition_w100': m_transition_w,
    'bigram': m_bigram, 'trigram': m_trigram, '4gram': m_4gram, '5gram': m_5gram,
    # Group C: Delta
    'delta_mode': m_delta_mode, 'delta_last': m_delta_last, 'delta2': m_delta2,
    'delta_bigram': m_delta_bigram,
    # Group D: Regression
    'linreg30': m_linreg, 'linreg50': m_linreg50, 'quadreg': m_quadreg,
    'momentum': m_momentum,
    # Group E: Conditional
    'cond_prev': m_conditional_prev, 'cond_next': m_conditional_next,
    'cond_sum': m_conditional_sum, 'cond_range': m_conditional_range,
    'cross_2pos': m_cross_2pos,
    # Group F: Cycle
    'cycle_auto': m_cycle_auto, 'cycle_exact': m_cycle_exact, 'cycle_fft': m_cycle_fft,
    # Group G: Gap/Overdue
    'gap_overdue': m_gap_overdue, 'gap_expected': m_gap_expected,
    # Group H: Lag
    'lag1': m_last, 'lag2': m_lag2, 'lag3': m_lag3, 'lag5': m_lag5,
    'lag7': m_lag7, 'lag10': m_lag10, 'best_lag': m_best_lag,
    # Group I: KNN
    'knn3': m_knn3, 'knn5': lambda h,p: m_knn(h,p,5), 'knn10': m_knn10, 'knn20': m_knn20,
    'knn_exact': m_knn_exact, 'knn_delta': m_knn_delta,
    # Group J: Modular
    'mod2': m_mod2, 'mod3': m_mod3, 'mod5': m_mod5, 'mod7': m_mod7,
    'digit_sum': m_digit_sum,
    # Group K: Streak/Direction
    'streak': m_streak, 'direction': m_direction,
    # Group L: Range
    'range_center': m_range_center, 'range_bounce': m_range_bounce,
    # Group M: Volatility
    'vol_regime': m_volatility_regime, 'regime_switch': m_regime_switch,
    # Group N: Ensemble
    'top2_trans': m_top2_transition, 'vote3': m_vote3, 'vote5': m_vote5,
    'weighted_vote': m_weighted_vote,
}

print(f"  Total methods: {len(METHODS)}\n")

# ================================================================
# BACKTEST ALL METHODS × 6 POSITIONS
# ================================================================
START = 100
TESTED = total - START - 1

# Exact hits, ±1, ±2, ±3
hits_exact = {m: [0]*6 for m in METHODS}
hits_pm1 = {m: [0]*6 for m in METHODS}
hits_pm2 = {m: [0]*6 for m in METHODS}
hits_pm3 = {m: [0]*6 for m in METHODS}

# Conditional accuracy: track which draws each method got right
method_correct = {m: [[] for _ in range(6)] for m in METHODS}

t0 = time.time()
for idx in range(START, total-1):
    history = sorted_draws[:idx+1]
    actual = sorted_draws[idx+1]

    for mname, mfunc in METHODS.items():
        for pos in range(6):
            try:
                pred = mfunc(history, pos)
                diff = abs(pred - actual[pos])
                if diff == 0:
                    hits_exact[mname][pos] += 1
                    method_correct[mname][pos].append(idx)
                if diff <= 1: hits_pm1[mname][pos] += 1
                if diff <= 2: hits_pm2[mname][pos] += 1
                if diff <= 3: hits_pm3[mname][pos] += 1
            except:
                pass

    done = idx - START + 1
    if done % 100 == 0:
        el = time.time() - t0
        eta = el / done * (TESTED - done) / 60
        print(f"  [{done}/{TESTED}] {el:.0f}s ETA={eta:.1f}m")
        sys.stdout.flush()

el = time.time() - t0
print(f"\n  Completed in {el:.0f}s ({el/60:.1f}m)\n")

# ================================================================
# RESULTS PER POSITION — FIND EACH COLUMN'S VULNERABILITY
# ================================================================
for pos in range(6):
    print(f"\n{'='*90}")
    print(f" POSITION {pos+1} {'(SMALLEST)' if pos==0 else '(LARGEST)' if pos==5 else ''}")
    print(f" Distribution: {len(set(sd[pos] for sd in sorted_draws))} unique values")
    vals = [sd[pos] for sd in sorted_draws]
    print(f" Range: {min(vals)}-{max(vals)}, Mean={np.mean(vals):.1f}, Std={np.std(vals):.1f}")
    top5 = Counter(vals).most_common(5)
    print(f" Top-5 values: {', '.join(f'{v}({c})' for v,c in top5)}")
    print(f"{'='*90}\n")

    # Rank all methods for this position
    ranked = []
    for mname in METHODS:
        exact = hits_exact[mname][pos] / TESTED * 100
        pm1 = hits_pm1[mname][pos] / TESTED * 100
        pm2 = hits_pm2[mname][pos] / TESTED * 100
        pm3 = hits_pm3[mname][pos] / TESTED * 100
        ranked.append((mname, exact, pm1, pm2, pm3))

    ranked.sort(key=lambda x: -x[1])  # Sort by exact accuracy

    print(f"  {'Rank':<5} {'Method':<20} {'Exact%':<10} {'±1%':<10} {'±2%':<10} {'±3%':<10}")
    print(f"  {'-'*65}")
    for i, (mname, exact, pm1, pm2, pm3) in enumerate(ranked[:20]):
        marker = " ◄ BEST" if i == 0 else ""
        print(f"  {i+1:<5} {mname:<20} {exact:<10.2f} {pm1:<10.2f} {pm2:<10.2f} {pm3:<10.2f}{marker}")

    # VULNERABILITY ANALYSIS
    best_name, best_exact = ranked[0][0], ranked[0][1]
    print(f"\n  ► VULNERABILITY: Best method = {best_name} at {best_exact:.2f}%")
    print(f"    {TESTED} draws tested, {hits_exact[ranked[0][0]][pos]} exact hits")

    # Check consecutive hits (streaks of correct predictions)
    correct_idxs = method_correct[best_name][pos]
    if correct_idxs:
        max_streak = 1
        cur_streak = 1
        for j in range(1, len(correct_idxs)):
            if correct_idxs[j] == correct_idxs[j-1] + 1:
                cur_streak += 1
                max_streak = max(max_streak, cur_streak)
            else:
                cur_streak = 1
        print(f"    Max consecutive hits streak: {max_streak}")

    # Category analysis: which VALUE is most predictable?
    val_hits = defaultdict(int)
    val_total = defaultdict(int)
    for j in range(START, total-1):
        actual_v = sorted_draws[j+1][pos]
        val_total[actual_v] += 1
    for j in correct_idxs:
        actual_v = sorted_draws[j+1][pos]
        val_hits[actual_v] += 1

    print(f"\n    Most predictable values at Pos {pos+1}:")
    val_accs = [(v, val_hits.get(v,0)/val_total[v]*100, val_total[v])
                for v in val_total if val_total[v] >= 10]
    val_accs.sort(key=lambda x: -x[1])
    for v, acc, cnt in val_accs[:5]:
        print(f"      Value {v:>2}: {acc:.1f}% ({val_hits.get(v,0)}/{cnt})")

    # Worst values
    print(f"    Hardest to predict values at Pos {pos+1}:")
    val_accs.sort(key=lambda x: x[1])
    for v, acc, cnt in val_accs[:5]:
        print(f"      Value {v:>2}: {acc:.1f}% ({val_hits.get(v,0)}/{cnt})")

# ================================================================
# GRAND SUMMARY — BEST METHOD PER COLUMN + JOINT
# ================================================================
print(f"\n{'='*90}")
print(f" GRAND SUMMARY — BEST SPECIALIST PER POSITION")
print(f"{'='*90}\n")

grand_methods = []
joint_exact = 1.0
joint_pm1 = 1.0
joint_pm2 = 1.0

for pos in range(6):
    best_name = max(METHODS, key=lambda m: hits_exact[m][pos])
    best_exact = hits_exact[best_name][pos] / TESTED * 100
    best_pm1 = hits_pm1[best_name][pos] / TESTED * 100
    best_pm2 = hits_pm2[best_name][pos] / TESTED * 100
    joint_exact *= best_exact / 100
    joint_pm1 *= best_pm1 / 100
    joint_pm2 *= best_pm2 / 100
    grand_methods.append((pos, best_name, best_exact))
    print(f"  Pos {pos+1}: {best_name:<20} = {best_exact:.2f}% exact "
          f"| ±1={best_pm1:.2f}% | ±2={best_pm2:.2f}%")

print(f"\n  JOINT EXACT 6/6 = {joint_exact*100:.6f}%")
print(f"  JOINT ±1 ALL 6  = {joint_pm1*100:.4f}%")
print(f"  JOINT ±2 ALL 6  = {joint_pm2*100:.4f}%")
print(f"  Expected 6/6 in {TESTED} draws = {joint_exact*TESTED:.3f}")

# ================================================================
# GAP TO 100%: WHAT WOULD IT TAKE?
# ================================================================
print(f"\n{'='*90}")
print(f" GAP ANALYSIS: HOW FAR FROM 100%?")
print(f"{'='*90}\n")

for pos, mname, acc in grand_methods:
    gap = 100 - acc
    factor = 100 / acc if acc > 0 else float('inf')
    print(f"  Pos {pos+1}: {acc:.2f}% → need {factor:.1f}x improvement to reach 100%")
    # What's the per-value ceiling?
    vals = [sd[pos] for sd in sorted_draws]
    top1_val = Counter(vals).most_common(1)[0]
    ceiling = top1_val[1] / total * 100
    print(f"    Mode ceiling (always guess {top1_val[0]}): {ceiling:.2f}%")
    n_unique = len(set(vals))
    random = 1/n_unique*100
    print(f"    Random baseline (1/{n_unique}): {random:.2f}%")
    print(f"    Current best is {acc/random:.1f}x better than random\n")

print(f"{'='*90}")
print(" DONE — Use per-position specialists to build optimal combos.")
print(f"{'='*90}")
