from typing import Any

import numpy as np
from numba import jit
from numpy.typing import ArrayLike


def simple_interval_reductor(t: float, a: float, b: float, *, pow=1) -> float:
    if t < 0:
        return 0
    elif t < a:
        return (t / a) ** pow
    elif t > b:
        return (b / t) ** pow
    else:
        return 1


def simple_interval_reductor_m(
    x: float, a: float, b: float, *, pow=1
) -> float:
    r = 1.0
    for t in x:
        if t == 0:
            continue
        if t < 0:
            r *= 0
        elif t < a:
            r *= (t / a) ** pow
        elif t > b:
            r *= (b / t) ** pow
        else:
            pass
    return r


def get_combined_reductor_1(r, a):
    return (1 - a)*sum(r)/len(r) + a*min(r)

def get_sharpe(dpnl: ArrayLike, *, resolution: float = 1) -> float:
    mu = np.mean(dpnl)
    s = np.std(dpnl)
    if s == 0:
        return 0
    return mu / s * np.sqrt(24 * 3600 / resolution)

def get_daily_sharpe_annualized(dpnl: ArrayLike) -> float:
    mu = np.mean(dpnl)
    s = np.std(dpnl)
    if s == 0:
        return 0
    return mu / s * np.sqrt(252)

def smoothen_positive(x):
    a = np.linalg.norm(x) / np.sqrt(x.shape[0])
    if a == 0:
        return x
    return np.tanh(x / a) * a


@jit(nopython=True)
def get_log_deltas(P):
    n = P.shape[0]
    y = np.zeros(n)
    for j in range(1, n):
        if P[j-1] > 0:
            y[j] = np.log(P[j] / P[j-1])
    return y


@jit(nopython=True)
def prolonged_action_positive(x, alpha):
    n = x.shape[0]
    y = np.zeros(n)
    h = 0
    q = 1 - alpha
    for j in range(n):
        x1 = x[j]
        h = h * q
        if x1 > h:
            h = x1
        y[j] = h
    return y


def calc_sharpe_m(
    dpnl, *, resolution: float = 1, allow_positive: bool = False
) -> ArrayLike:
    k = np.sqrt(24 * 3600 / resolution)
    m = dpnl.shape[1]
    sharpe = np.zeros(m)
    for i in range(m):
        y = dpnl[:, i]
        if allow_positive:
            x = smoothen_positive(y)
        else:
            x = y
        mu = np.mean(y)
        s = np.std(x)
        if s == 0:
            return np.zeros(m)
        sharpe[i] = mu / s * k
    return sharpe


@jit(nopython=True)
def avg_amp_l2(X):
    return np.linalg.norm(X) / np.sqrt(X.shape[0])


@jit(nopython=True)
def avg_abs_non_zero(X):
    S0, S1 = 0, 0
    n = X.shape[0]
    for j in range(n):
        x = X[j]
        if x != 0:
            S1 += np.abs(x)
            S0 += 1
    return S1 / S0 if S0 > 0 else 0


@jit(nopython=True)
def fibonacci_filter(X, h):
    n = X.shape[0]
    Y = np.zeros(n)
    h1 = h * (1 - (np.sqrt(5) - 1) / 2)
    A = 0
    for j in range(n):
        x = X[j]
        a = round(x / h)
        if j == 0:
            y = a * h
        else:
            y = a * h
            if a <= A - 2 or a >= A + 2:
                ...
            elif np.abs(x - y) < h1:
                ...
            else:
                y = Y[j - 1]
        Y[j] = y
        A = a
    return Y


@jit(nopython=True)
def calc_num_changes(X):
    n = X.shape[0]
    d = np.zeros(n)
    c = 0
    for j in range(1, n):
        if X[j] != X[j - 1]:
            c += 1
    return c


@jit(nopython=True)
def calc_diff(X) -> ArrayLike:
    n = X.shape[0]
    d = np.zeros(n)
    for j in range(1, n):
        d[j] = X[j] - X[j - 1]
    return d


@jit(nopython=True)
def calc_diff_s(X, s) -> ArrayLike:
    n = X.shape[0]
    d = np.zeros(n)
    for j in range(s, n):
        d[j] = X[j] - X[j - s]
    return d


@jit(nopython=True)
def calc_diff0(X) -> ArrayLike:
    n = X.shape[0]
    d = np.zeros(n)
    d[0] = X[0]
    for j in range(1, n):
        d[j] = X[j] - X[j - 1]
    return d


@jit(nopython=True)
def calc_diff_m(X) -> ArrayLike:
    n = X.shape[0]
    m = X.shape[1]
    d = np.zeros(X.shape)
    for i in range(m):
        for j in range(1, n):
            d[j, i] = X[j, i] - X[j - 1, i]
    return d


def normalize_m(X):
    n = X.shape[0]
    m = X.shape[1]
    Y = np.zeros(X.shape)
    for i in range(m):
        n1 = np.linalg.norm(X[:, i]) / np.sqrt(n)
        Y[:, i] = X[:, i] / n1
    return Y


@jit(nopython=True)
def encode_by_threshold(X, h) -> ArrayLike:
    n = X.shape[0]
    e = np.zeros(n)
    for j in range(n):
        x = X[j]
        if x >= h:
            e[j] = 1
        elif x <= -h:
            e[j] = -1
    return e


@jit(nopython=True)
def encode_by_threshold_2(X, h0, h1) -> ArrayLike:
    n = X.shape[0]
    e = np.zeros(n)
    for j in range(n):
        x = X[j]
        if x >= h1:
            e[j] = 1
        elif x <= -h0:
            e[j] = -1
    return e


@jit(nopython=True)
def make_risk_factor1(X, h0, h1) -> ArrayLike:
    n = X.shape[0]
    risk = np.zeros(n)
    R = 0
    for j in range(n):
        x = X[j]
        if R > 0:
            if x < h1:
                R = 0
        elif R < 0:
            if x > -h1:
                R = 0
        if x >= h0:
            R = 1
        elif x <= -h0:
            R = -1
        risk[j] = R
    return risk


@jit(nopython=True)
def apply_risk0(X, risk) -> ArrayLike:
    n = X.shape[0]
    Y = np.zeros(n)
    for j in range(n):
        if risk[j] == 0:
            Y[j] = X[j]
    return Y


@jit(nopython=True)
def apply_risk(X, risk) -> ArrayLike:
    n = X.shape[0]
    Y = np.zeros(n)
    for j in range(n):
        if risk[j] > 0:
            if X[j] > 0:
                Y[j] = X[j]
        elif risk[j] < 0:
            if X[j] < 0:
                Y[j] = X[j]
        else:
            Y[j] = X[j]
    return Y


@jit(nopython=True)
def calc_outlier_fraction(X, a) -> float:
    n = X.shape[0]
    s = 0
    for j in range(n):
        x = X[j]
        if x >= a or x <= -a:
            s += 1
    return s / n


@jit(nopython=True)
def calc_outlier_match(X, a, Y) -> float:
    n = X.shape[0]
    s0 = 0
    s1 = 0
    for j in range(n):
        x = X[j]
        if x >= a or x <= -a:
            s0 += 1
            if x * Y[j] >= 0:
                s1 += 1
    return s1 / s0


@jit(nopython=True)
def calc_max_profit(v, alpha, mode):
    n = int(v.shape[0])
    y = np.zeros(n)
    y0 = v[0]
    y[0] = y0
    for i in range(1, n):
        y[i] = y0
        y0 += alpha * (v[i] - y0)
    S = 0
    iV = 0
    cost = 0
    for i in range(1, n):
        if mode == 1:
            S += np.abs(y[i] - y[i - 1]) / y[i]
        else:
            if y[i] - y[i - 1] > 0:
                S += (y[i] - y[i - 1]) / y[i]
        if i > 1:
            if (y[i - 1] - y[i - 2]) * (y[i] - y[i - 1]) < 0:
                iV += 1
                cost += 0.001
    nMonth = n / (30 * 24 * 60)
    profit = S / nMonth
    return nMonth, S, cost, iV


@jit(nopython=True)
def diff(x):
    n = int(x.shape[0])
    y = np.zeros(n)
    for j in range(1, n):
        y[j] = x[j] - x[j - 1]
    return y


@jit(nopython=True)
def diffk(x, k):
    n = int(x.shape[0])
    y = np.zeros(n)
    for j in range(1, k):
        y[j] = x[j] - x[0]
    for j in range(k, n):
        y[j] = x[j] - x[j - k]
    return y


@jit(nopython=True)
def get_last_value(x, ast):
    n = x.shape[0]
    k = -1
    for j in range(n-1):
        if ast[j] > 0 and ast[j+1] == 0:
            k = j
            break
    if k < 0:
        k = n-1
    return x[k]


@jit(nopython=True)
def shift(x, k):
    n = int(x.shape[0])
    y = np.zeros(n)
    if k >= 0:
        for j in range(k, n):
            y[j] = x[j - k]
    else:
        for j in range(0, n + k):
            y[j] = x[j - k]
    return y


@jit(nopython=True)
def if_a_greater_b(a, b, c):
    n = int(a.shape[0])
    y = np.zeros(n)
    for j in range(n):
        x = a[j]
        x_abs = np.abs(x)
        if x > 0:
            if x_abs > b[j] * c:
                y[j] = 1
        elif x < 0:
            if x_abs > b[j] * c:
                y[j] = -1
    return y


@jit(nopython=True)
def mark_pos(x):
    n = int(x.shape[0])
    y = np.zeros(n)
    for j in range(n):
        if x[j] > 0:
            y[j] = 1
    return y


@jit(nopython=True)
def count_non_zero(x):
    n = int(x.shape[0])
    S = 0
    for j in range(n):
        if x[j] != 0:
            S += 1
    return S


@jit(nopython=True)
def count_non_zero_blocks(x):
    n = int(x.shape[0])
    S = 0
    j0 = -1
    for j in range(n):
        if x[j] != 0:
            if j0 == -1:
                j0 = j
        else:
            if j0 != -1:
                S += 1
                j0 = -1
    return S


@jit(nopython=True)
def avg_non_zero_blocks(x):
    M = count_non_zero_blocks(x)
    return count_non_zero(x) / M if M > 0 else 0





@jit(nopython=True)
def mark_neg(x):
    n = int(x.shape[0])
    y = np.zeros(n)
    for j in range(n):
        if x[j] < 0:
            y[j] = 1
    return y


@jit(nopython=True)
def DX0(x, alpha):
    y = calc_qema1(mark_pos(x), alpha)
    return y


@jit(nopython=True)
def DX1(x, alpha):
    y = calc_qema1(mark_neg(x), alpha)
    return y


@jit(nopython=True)
def DXI(x, alpha):
    n = x.shape[0]
    y = DX0(x, alpha)
    z = DX1(x, alpha)
    dxi = np.zeros(n)
    for j in range(n):
        if y[j] > 0 and z[j] > 0:
            dxi[j] = (z[j] - y[j]) / (z[j] + y[j])
    return y


@jit(nopython=True)
def if_a_less_b(a, b, c):
    n = int(a.shape[0])
    y = np.zeros(n)
    for j in range(n):
        x = a[j]
        x_abs = np.abs(x)
        if x > 0:
            if x_abs <= b[j] * c:
                y[j] = 1
        elif x < 0:
            if x_abs <= b[j] * c:
                y[j] = -1
    return y


@jit(nopython=True)
def shift_fill(x, k):
    n = int(x.shape[0])
    y = np.zeros(n)
    if k >= 0:
        for j in range(k):
            y[j] = x[0]
        for j in range(k, n):
            y[j] = x[j - k]
    else:
        for j in range(n + k, n):
            y[j] = x[-1]
        for j in range(0, n + k):
            y[j] = x[j - k]
    return y


@jit(nopython=True)
def num_mismatching(x, y):
    n = int(x.shape[0])
    N = 0
    for j in range(n):
        if y[j] != x[j]:
            N += 1
    return N


@jit(nopython=True)
def fix_quote(P):
    n = P.shape[0]
    P0 = np.zeros(n)
    p = P[0]
    P0[0] = p
    for j in range(1, n):
        if P[j] > 0:
            p = P[j]
        P0[j] = p
    return P0


@jit(nopython=True)
def calc_rsi(y, alpha, gain, loss):
    n = int(y.shape[0])
    rsi = np.zeros(n)
    # print(gain, loss)
    for j in range(1, n):
        dy = y[j] / y[j - 1] - 1
        if dy > 0:
            gain += alpha * (dy - gain)
        elif dy < 0:
            loss += alpha * (-dy - loss)
        if loss > 0:
            rsi[j] = 1 - 1 / (1 + gain / loss)
        else:
            rsi[j] = 1
        rsi[0] = rsi[1]
    return rsi, gain, loss


@jit(nopython=True)
def calc_rsi_r(y, alpha, resolution):
    n = int(y.shape[0])
    return v


@jit(nopython=True)
def calc_max_h(x, h):
    a = np.zeros(x.shape)
    n = x.shape[0]
    m = x.shape[1]
    for j in range(1, n):
        for i in range(m):
            a[j, i] = max(x[j, i], h)
    return a


@jit(nopython=True)
def calc_min_h(x, h):
    a = np.zeros(x.shape)
    n = x.shape[0]
    m = x.shape[1]
    for j in range(1, n):
        for i in range(m):
            a[j, i] = min(x[j, i], h)
    return a


@jit(nopython=True)
def calc_pos_areas(position):
    n = position.shape[0]
    m = position.shape[1]
    s0 = np.zeros(m)
    s1 = np.zeros(m)
    for j in range(1, n):
        for i in range(m):
            if position[j, i] > 0:
                s0[i] += 1
            elif position[j, i] < 0:
                s1[i] += 1
    return s0, s1


@jit(nopython=True)
def calc_pos_areas_pct(position):
    s0, s1 = calc_pos_areas(position)
    n = position.shape[0]
    m = position.shape[1]
    for i in range(m):
        s0[i] /= n
        s1[i] /= n
    return s0, s1


@jit(nopython=True)
def calc_volume(position):
    n = position.shape[0]
    volume = np.zeros(n)
    for j in range(1, n):
        dV = np.abs(position[j] - position[j - 1])
        volume[j] = dV
    return volume


@jit(nopython=True)
def apply_ast(p, ast):
    n = p.shape[0]
    size = int(sum(ast))
    a = np.zeros(size)
    i = 0
    for j in range(0, n):
        if ast[j] > 0:
            a[i] = p[j]
            i += 1
    return a


@jit(nopython=True)
def calc_num_contracts(position):
    n = position.shape[0]
    volume = np.zeros(n)
    for j in range(1, n):
        dp = position[j] - position[j - 1]
        if dp != 0:
            volume[j] += 1
    return volume


@jit(nopython=True)
def calc_num_contracts_m(position):
    n = position.shape[0]
    m = position.shape[1]
    volume = np.zeros(position.shape)
    for i in range(m):
        for j in range(1, n):
            dp = position[j, i] - position[j - 1, i]
            if dp != 0:
                volume[j, i] += 1
    return volume


@jit(nopython=True)
def calc_volume_m(position):
    volume = np.zeros(position.shape)
    n = position.shape[0]
    m = position.shape[1]
    for j in range(1, n):
        for i in range(m):
            dV = np.abs(position[j, i] - position[j - 1, i])
            volume[j, i] = volume[j - 1, i] + dV
    return volume


@jit(nopython=True)
def calc_tot_volume_and_intervals(position):
    volume = calc_volume(position)
    num_contracts = calc_num_contracts(position)
    tot_nc = np.sum(num_contracts)
    n = position.shape[0]
    intervals = 0
    if tot_nc > 0:
        intervals = n / tot_nc
    return np.sum(volume), intervals


@jit(nopython=True)
def calc_active_intervals(position):
    n = position.shape[0]
    I = 0
    N = 0
    j0 = -1
    x0 = 0
    for j in range(n):
        x = position[j]
        if j0 != -1 and x != x0:
            I += j - j0
            N += 1
            j0 = -1
        if x != 0 and j0 == -1:
            j0 = j
            x0 = x
    if N > 0:
        I /= N
    return I


@jit(nopython=True)
def calc_active_intervals_m(position):
    n = position.shape[0]
    m = position.shape[1]
    aI = np.zeros(m)
    for i in range(m):
        I = 0
        N = 0
        j0 = -1
        x0 = 0
        for j in range(n):
            x = position[j, i]
            if j0 != -1 and x != x0:
                I += j - j0
                N += 1
                j0 = -1
            if x != 0 and j0 == -1:
                j0 = j
                x0 = x
        if N > 0:
            I /= N
        aI[i] = I
    return aI


@jit(nopython=True)
def calc_tot_volume_and_intervals_m(position):
    volume = calc_volume_m(position)
    num_contracts = calc_num_contracts_m(position)
    m = position.shape[1]
    intervals = np.zeros(m)
    n = position.shape[0]
    for i in range(m):
        s = np.sum(num_contracts[:, i])
        if s > 0:
            intervals[i] = n / s
            # print(n, s, intervals[i])
    return volume[-1, :], intervals


@jit(nopython=True)
def calc_absolute_performance_m(position, data):
    n = position.shape[0]
    m = position.shape[1]
    performance = np.zeros(m)
    for i in range(m):
        enter_mid = 0
        S = 0
        for j in range(1, n):
            if position[j, i] != position[j - 1, i]:
                if enter_mid != 0:
                    if position[j - 1, i] != 0:
                        S += np.abs(data[j, i] - enter_mid)
            enter_mid = data[j, i]
        performance[i] = S
    return performance


@jit(nopython=True)
def calc_high_low_signal_m(data, thr):
    n = data.shape[0]
    m = data.shape[1]
    sig = np.zeros(data.shape)
    mn = np.zeros(data.shape)
    mx = np.zeros(data.shape)
    for i in range(m):
        for j in range(n):
            _mn = data[j, 0]
            _mx = data[j, 0]
            for k in range(m):
                if k == i:
                    continue
                x = data[j, k]
                if x < _mn:
                    _mn = x
                if x > _mx:
                    _mx = x
            mn[j, i] = _mn
            mx[j, i] = _mx
            # print(mn[j])
            # print(mx[j])
    for i in range(m):
        thr1 = thr[i]
        for j in range(n):
            x = data[j, i]
            if x >= mx[j, i] + thr1:
                sig[j, i] = 1
            elif x <= mn[j, i] - thr1:
                sig[j, i] = -1
    return sig


@jit(nopython=True)
def merge_signals_m(sig1, sig2):
    n = sig1.shape[0]
    m = sig1.shape[1]
    sig = np.zeros(sig1.shape)
    for i in range(m):
        for j in range(n):
            if sig2[j, i] > 0:
                if sig1[j, i] > 0:
                    sig[j, i] = sig1[j, i]
            elif sig2[j, i] < 0:
                if sig1[j, i] < 0:
                    sig[j, i] = sig1[j, i]
    return sig


@jit(nopython=True)
def merge_signals(sig1, sig2):
    n = sig1.shape[0]
    sig = np.zeros(sig1.shape)
    for j in range(n):
        if sig2[j] > 0:
            if sig1[j] > 0:
                sig[j] = sig1[j]
        elif sig2[j] < 0:
            if sig1[j] < 0:
                sig[j] = sig1[j]
    return sig


@jit(nopython=True)
def calc_matrix_ema(x, beta):
    a = np.zeros(x.shape)
    n = x.shape[0]
    m = x.shape[1]
    for i in range(m):
        a[0, i] = x[0, i]
    for j in range(1, n):
        for i in range(m):
            a[j, i] = (1 - beta[i]) * a[j - 1, i] + beta[i] * x[j, i]
    return a


@jit(nopython=True)
def calc_qema1(v, alpha):
    q = 1 - alpha
    _1_q2 = 1 - q ** 2
    _1_q_2 = alpha ** 2
    Y = v[0] / alpha
    Z = (v[0] - Y) / alpha
    y = np.zeros(v.shape[0])
    y[0] = v[0]
    for i in range(1, v.shape[0]):
        x = v[i]
        Y = q * Y + x
        Z = q * Z + x - Y
        y[i] += _1_q_2 * Z + _1_q2 * Y
    return y


@jit(nopython=True)
def calc_qema1_zero_start(v, alpha):
    q = 1 - alpha
    _1_q2 = 1 - q ** 2
    _1_q_2 = alpha ** 2
    Y = 0
    Z = 0
    y = np.zeros(v.shape[0])
    for i in range(0, v.shape[0]):
        x = v[i]
        Y = q * Y + x
        Z = q * Z + x - Y
        y[i] += _1_q_2 * Z + _1_q2 * Y
    return y


@jit(nopython=True)
def calc_qema_YZ(v, alpha):
    q = 1 - alpha
    _1_q2 = 1 - q ** 2
    _1_q_2 = alpha ** 2
    Y = v[0] / alpha
    Z = (v[0] - Y) / alpha
    y = np.zeros(v.shape[0])
    aY = np.zeros(v.shape[0])
    aZ = np.zeros(v.shape[0])
    y[0] = v[0]
    aY[0] = Y
    aZ[0] = Z
    for i in range(1, v.shape[0]):
        x = v[i]
        Y = q * Y + x
        Z = q * Z + x - Y
        y[i] += _1_q_2 * Z + _1_q2 * Y
        aY[i] = Y
        aZ[i] = Z
    return y, aY, aZ


@jit(nopython=True)
def calc_sa(v):
    S = 0.0
    a = np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        S += v[i]
        a[i] = S / (i + 1)
    return a


@jit(nopython=True)
def get_ast_j0_j1(ast):
    n = ast.shape[0]
    j0 = -1
    j1 = -1
    for j in range(n):
        if ast[j] > 0:
            if j0 == -1:
                j0 = j
        else:
            if j0 >= 0 and j1 == -1:
                j1 = j
    if j1 == -1:
        j1 = n
    return j0, j1


@jit(nopython=True)
def calc_safe_div(v, u):
    a = np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        if u[i] != 0:
            a[i] = v[i] / u[i]
    return a


@jit(nopython=True)
def normalize_rolling_m(x, beta):
    a = np.zeros(x.shape)
    m = x.shape[1]
    for i in range(m):
        _amp = x[:, i] * x[:, i]
        c = calc_sa(_amp)
        c = np.sqrt(c)
        a[:, i] = calc_safe_div(x[:, i], c)
        # for j in range(min(10, x.shape[0])):
        #     if j >= 2:
        #         a[j, i] = np.tanh(a[j, i])
        #     else:
        #         a[j, i] = 0
    return a


@jit(nopython=True)
def calc_rrank(g):
    rr = np.zeros(g.shape)
    n = g.shape[0]
    m = g.shape[1]
    for j in range(n):
        x = np.zeros(m)
        l = np.zeros(m)
        for i in range(m):
            x[i] = g[j, i]
            l[i] = i
        for k in range(m - 1):
            for p in range(m - k - 1):
                if x[p] > x[p + 1]:
                    t = x[p]
                    x[p] = x[p + 1]
                    x[p + 1] = t
                    t2 = l[p]
                    l[p] = l[p + 1]
                    l[p + 1] = t2
        for i in range(m):
            rr[j, int(l[i])] = i
    return rr


@jit(nopython=True)
def calc_rank_statistics(rk):
    n = rk.shape[0]
    m = rk.shape[1]
    stat = np.zeros(32 * m)
    y = np.zeros(n)
    for i in range(m):
        x = rk[:, i]
        for k in range(4):
            w = k + 1
            n_intervals = 0
            n_active_intervals = 0
            n_stable_intervals = 0
            S_active = 0
            mode = 0
            sig = np.zeros(n)
            signal = 0
            for j in range(n):
                _x = x[j]
                if signal > 0:
                    if _x >= w:
                        signal = 0
                elif signal < 0:
                    if _x < m - w:
                        signal = 0
                if signal == 0:
                    if _x < 1:
                        signal = 1
                    elif _x >= m - 1:
                        signal = -1
                sig[j] = signal

                if _x < w:
                    y[j] = 1
                    S_active += 1
                    if j > 0:
                        if y[j] != y[j-1]:
                            n_active_intervals += 1
                            n_intervals += 1
                        if mode != 1:
                            n_stable_intervals += 1
                            mode = 1
                elif _x >= m - w:
                    y[j] = -1
                    S_active += 1
                    if j > 0:
                        if y[j] != y[j-1]:
                            n_active_intervals += 1
                            n_intervals += 1
                        if mode != -1:
                            n_stable_intervals += 1
                            mode = -1
                else:
                    y[j] = 0
                    if j > 0:
                        if y[j] != y[j - 1]:
                            n_intervals += 1
            stat[32 * i + 8 * k] = S_active / n
            stat[32 * i + 8 * k + 1] = n_intervals
            stat[32 * i + 8 * k + 2] = n_active_intervals
            if n_active_intervals > 0:
                stat[32 * i + 8 * k + 3] = S_active / n_active_intervals
            stat[32 * i + 8 * k + 4] = n_stable_intervals
    return stat



@jit(nopython=True)
def calc_rrank_mask(g, mask):
    rr = np.zeros(g.shape)
    n = g.shape[0]
    m = g.shape[1]
    _next = [-1] * m
    for i in range(m - 1):
        if mask[i] == 0:
            continue
        for k in range(i + 1, m):
            if mask[k] > 0:
                _next[i] = k
                break
    for j in range(1, n):
        x = np.zeros(m)
        l = np.zeros(m)
        for i in range(m):
            if mask[i] == 0:
                continue
            x[i] = g[j, i]
            l[i] = i
        for k in range(m - 1):
            for p in range(m - k - 1):
                if mask[k] == 0 or mask[p] == 0:
                    continue
                p2 = _next[p]
                if p2 == -1:
                    print("! -1 !")
                if x[p] > x[p2]:
                    t = x[p]
                    x[p] = x[p2]
                    x[p2] = t
                    t2 = l[p]
                    l[p] = l[p2]
                    l[p2] = t2
        for i in range(m):
            if mask[i] == 0:
                continue
            rr[j, int(l[i])] = i
    return rr


@jit(nopython=True)
def calc_sectors_avg(g, sectors, n_sectors):
    n = g.shape[0]
    m = g.shape[1]
    a = np.zeros((n, n_sectors))
    b = np.zeros(n_sectors)
    for i in range(m):
        b[sectors[i]] += 1
    for j in range(n):
        for i in range(m):
            a[j, sectors[i]] += g[j, i]
        for k in range(n_sectors):
            a[j, k] /= b[k]
    return a


@jit(nopython=True)
def calc_mid_growth(mid, alpha):
    avgm = calc_matrix_ema(mid, alpha)
    g = np.zeros(mid.shape)
    n = avgm.shape[0]
    m = avgm.shape[1]
    for i in range(m):
        for j in range(1, n):
            dp = avgm[j, i] / avgm[j - 1, i] - 1
            g[j, i] = dp
    return g


@jit(nopython=True)
def calc_rel_mid_growth(mid, alpha, lin_a):
    g = calc_mid_growth(mid, alpha)
    rg = np.zeros(g.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    a = np.zeros(m)
    for i in range(m):
        for j in range(n):
            a[i] += lin_a[i] * (g[j, i] - a[i])
            rg[j, i] = g[j, i] - a[i]
    return rg


@jit(nopython=True)
def calc_portfolio_rsi(mid, alpha, lin_a, w, w_delta):
    delta = 200
    # print(delta)
    rg = calc_rel_mid_growth(mid, alpha, lin_a)
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        delta = w_delta[i]
        for j in range(1, n):
            r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < w:
                    rsi[j, i] = 1
                elif x >= m - w:
                    rsi[j, i] = -1
            elif r < 0:
                if x < w:
                    rsi[j, i] = 1
                elif x < m - w - delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
            else:
                if x >= m - w:
                    rsi[j, i] = -1
                elif x >= w + delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r

            # if r == 0:
            #     if x < w:
            #         rsi[j, i] = 1
            #     elif x >= m - w:
            #         rsi[j, i] = -1
            # elif r > 0:
            #     if x >= m - w:
            #         rsi[j, i] = -1
            #     elif x >= w + delta:
            #         print('1')
            #         # exit()
            #         rsi[j, i] = 0
            # else:  # r < 0
            #     if x < w:
            #         rsi[j, i] = 1
            #     elif x < m - w - delta:
            #         print('2')
            #         # exit()
            #         rsi[j, i] = 0

    return rsi


@jit(nopython=True)
def calc_portfolio_rsi0(mid, alpha, lin_a, w, w_delta):
    rg = calc_rel_mid_growth(mid, alpha, lin_a)
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        delta = w_delta[i]
        for j in range(1, n):
            # r = rsi[j-1, i]
            x = rr[j, i]
            if x < w:
                rsi[j, i] = 1
            elif x >= m - w:
                rsi[j, i] = -1

    return rsi


@jit(nopython=True)
def calc_rsi(dP, w):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(1, n):
            x = rr[j, i]
            if x < w:
                rsi[j, i] = 1
            elif x >= m - w:
                rsi[j, i] = -1

    return rsi


@jit(nopython=True)
def calc_smooth_rsi(dP):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(1, n):
            x = rr[j, i]
            rsi[j, i] = 1 - 2 * x / (m - 1)

    return rsi


@jit(nopython=True)
def calc_growth_filter(P, s, h0=0.25, h1=0.10, stop_on_loss=False):
    u0 = 1 - h0
    u1 = 1 - h1
    v0 = 1 / u0
    v1 = 1 / u1
    n = P.shape[0]
    filter = np.zeros(n)
    state = 0
    f_inva = 0
    for j in range(s, n):
        if stop_on_loss:
            _h2 = 0.60
            _h2i = 1 / _h2
            if P[j] < _h2 * P[j-390] or P[j] > _h2i * P[j-390]:
                f_inva = s
                break
        k = P[j] / P[j - s]
        if state == -1:
            if k >= u1:
                state = 0
        elif state == 1:
            if k <= v1:
                state = 0
        if k <= u0:
            state = -1
        elif k >= v0:
            state = 1
        filter[j] = state
    return filter, f_inva


@jit(nopython=True)
def apply_filter(sig, filter):
    n = sig.shape[0]
    sig2 = np.zeros(n)
    for j in range(n):
        x = sig[j]
        y = filter[j]
        if x > 0 and y < 0:
            x = 0
        if x < 0 and y > 0:
            x = 0
        sig2[j] = x
    return sig2


@jit(nopython=True)
def calc_classical_rsi(dP, w):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(1, n):
            x = rr[j, i]
            if x < w:
                rsi[j, i] = -1
            elif x >= m - w:
                rsi[j, i] = 1
    return rsi


@jit(nopython=True)
def calc_rsi_delayed(dP, w, delta):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(1, n):
            r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < w:
                    rsi[j, i] = 1
                elif x >= m - w:
                    rsi[j, i] = -1
            elif r < 0:
                if x < w:
                    rsi[j, i] = 1
                elif x < m - w - delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
            else:
                if x >= m - w:
                    rsi[j, i] = -1
                elif x >= w + delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
    return rsi


@jit(nopython=True)
def calc_rsi_delayed_p0(dP, w, delta, last_v):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(0, n):
            if j == 0:
                r = last_v[i]
            else:
                r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < w:
                    rsi[j, i] = 1
                elif x >= m - w:
                    rsi[j, i] = -1
            elif r < 0:
                if x < w:
                    rsi[j, i] = 1
                elif x < m - w - delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
            else:
                if x >= m - w:
                    rsi[j, i] = -1
                elif x >= w + delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
    return rsi


@jit(nopython=True)
def calc_rsi_bs(dP, boundary):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        for j in range(1, n):
            x = rr[j, i]
            if x < boundary:
                rsi[j, i] = 1 - x / boundary
            elif x >= m - boundary:
                rsi[j, i] = -1 + (m - 1 - x) / boundary  # x = m - boundary - 1: x = m - 1 - (m - boundary - 1) == b
    return rsi, rr[-1, :]


@jit(nopython=True)
def calc_rsi_g_p0(dP, group_size, groups):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    group0 = [0] * group_size
    group1 = [0] * group_size
    gs0 = [0] * group_size
    gs1 = [0] * group_size
    stat = {
        "new_top0": 0,
        "new_top1": 0,
    }
    j0 = 0
    if sum(groups) == -2 * group_size:
        j0 = 1
        k0 = 0
        k1 = 0
        for i in range(m):
            x = rr[0, i]
            # print("i", i, x)
            if k0 < group_size and x < group_size:
                group0[k0] = i
                k0 += 1
                # print(i, "> g0")
            if k1 < group_size and x >= m - group_size:
                group1[k1] = i
                k1 += 1
                # print(i, "> g1")
        # print("Init")
        # print(group0, group1, k0, k1)
        _a = [0] * m
        for _i in range(m):
            _a[int(rr[0, _i])] = _i
        # print(_a)
    else:
        for k in range(group_size):
            group0[k] = groups[k]
            group1[k] = groups[group_size + k]
    for j in range(j0, n):
        _a = [0] * m
        _u = [0] * m
        for _i in range(m):
            _a[int(rr[j, _i])] = _i
        for _k in range(group_size):
            _u[group0[_k]] = 1
            _u[group1[_k]] = 1
        top0 = -1
        top1 = -1
        for i in range(m):
            x = rr[j, i]
            if x == 0:
                top0 = i
            elif x == m - 1:
                top1 = i
        top0_new = 1
        top1_new = 1
        for k in range(group_size):
            if group0[k] == top0:
                top0_new = 0
            if group1[k] == top1:
                top1_new = 0
        if top0_new == 1:
            # print("groups", group0, group1)
            # print("_u", _u)
            stat["new_top0"] += 1
            for k in range(0, group_size - 1):
                group0[k] = group0[k + 1]
            group0[group_size - 1] = top0
            # checking the opposite group
            o1 = -1
            for k in range(group_size):
                if group1[k] == top0:
                    o1 = k
                    break
            if o1 != -1:
                # print("o1", o1, _u, group0, group1)
                for _r in range(m):
                    if _u[_a[m - 1 - _r]] == 0 and _a[m - 1 - _r] != top1:
                        group1[o1] = _a[m - 1 - _r]
                        break
            # print("new 0:", top0)
            # print(group0, group1, top0, top1)
            # print(_a)
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        if top1_new == 1:
            # print("groups", group0, group1)
            # print("_u", _u)
            stat["new_top1"] += 1
            for k in range(0, group_size - 1):
                group1[k] = group1[k + 1]
            group1[group_size - 1] = top1
            # checking the opposite group
            o0 = -1
            for k in range(group_size):
                if group0[k] == top1:
                    o0 = k
                    break
            if o0 != -1:
                # print("o0", o0, _u, group0, group1)
                for _r in range(m):
                    if _u[_a[_r]] == 0 and _a[_r] != top0:
                        group0[o0] = _a[_r]
                        break
            # print("new 1:", top1)
            # print(group0, group1, top0, top1)
            # print(_a)
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        # _r0 = group_size
        # _r1 = m - 1 - group_size
        # group0_c = group0.copy()
        # group1_c = group1.copy()
        # for _k in range(group_size):
        #     if int(rr[j, group0[_k]]) >= 2*group_size:
        #         while _u[_a[_r0]] > 0:
        #             _r0 += 1
        #         group0[_k] = _a[_r0]
        #         _u[group0[_k]] = 1
        #         # print(">", group0_c, group1_c, "    k =", _k, "    group0")
        #         # print("~", group0, group1)
        #         # return None, None, None
        #     if int(rr[j, group1[_k]]) < m - 2*group_size:
        #         while _u[_a[_r1]] > 0:
        #             _r1 -= 1
        #         group1[_k] = _a[_r1]
        #         _u[group1[_k]] = 1
        #         # print(">", group0_c, group1_c, "    k =", _k, "    group1")
        #         # print("~", group0, group1)
        #         # return None, None, None
        # for _k1 in range(group_size-1):
        #     for _k2 in range(_k1+1, group_size):
        #         if group0[_k1] == group0[_k2] or group1[_k1] == group1[_k2]:
        #             return None, None, None
        out_of_group_n = 0
        for _k in range(0, group_size):
            gs0[_k] = 0
            if int(rr[j, group0[_k]]) >= group_size:
                gs0[_k] = 1
                out_of_group_n += 1
            gs1[_k] = 0
            if int(rr[j, group1[_k]]) < m - group_size:
                gs1[_k] = 1
                out_of_group_n += 1

        # if out_of_group_n > 1 or True:
        #     # print("rebuilding...    ", group0, group1)
        #     k0 = 0
        #     k1 = 0
        #     for i in range(m):
        #         x = round(rr[j, i])
        #         if k0 < group_size and x < group_size:
        #             group0[k0] = i
        #             k0 += 1
        #         if k1 < group_size and x >= m - group_size:
        #             group1[k1] = i
        #             k1 += 1
        for k in range(group_size):
            rsi[j, group0[k]] = 1
            rsi[j, group1[k]] = -1
    _groups = [0] * (2 * group_size)
    for k in range(group_size):
        _groups[k] = group0[k]
        _groups[group_size + k] = group1[k]
    # print(group0, group1)
    return rsi, _groups, rr[-1, :], stat


@jit(nopython=True)
def calc_rsi_g1_p0(dP, group_size, groups):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    group0 = [0] * group_size
    group1 = [0] * group_size
    j0 = 0
    if sum(groups) == -2 * group_size:
        j0 = 1
        k0 = 0
        k1 = 0
        for i in range(m):
            x = rr[0, i]
            # print("i", i, x)
            if k0 < group_size and x < group_size:
                group0[k0] = i
                k0 += 1
                # print("> g0")
            if k1 < group_size and x >= m - group_size:
                group1[k1] = i
                k1 += 1
                # print("> g1")
        # print("Init")
        # print(group0, group1, k0, k1)
        _a = [0] * m
        for _i in range(m):
            _a[int(rr[0, _i])] = _i
        # print(_a)
    else:
        for k in range(group_size):
            group0[k] = groups[k]
            group1[k] = groups[group_size + k]
    for j in range(j0, n):
        _a = [0] * m
        _u = [0] * m
        for _i in range(m):
            _a[int(rr[j, _i])] = _i
        for _k in range(group_size):
            _u[group0[_k]] = 1
            _u[group1[_k]] = 1
        top0 = -1
        top1 = -1
        for i in range(m):
            x = rr[j, i]
            if x == 0:
                top0 = i
            elif x == m - 1:
                top1 = i
        top0_new = 1
        top1_new = 1
        for k in range(group_size):
            if group0[k] == top0:
                top0_new = 0
            if group1[k] == top1:
                top1_new = 0
        if top0_new == 1:
            # print("groups", group0, group1)
            # print("_u", _u)
            for k in range(0, group_size - 1):
                group0[k] = group0[k + 1]
            group0[group_size - 1] = top0
            # checking the opposite group
            o1 = -1
            for k in range(group_size):
                if group1[k] == top0:
                    o1 = k
                    break
            if o1 != -1:
                # print("o1", o1)
                for _r in range(m):
                    if _u[_a[m - 1 - _r]] == 0:
                        group1[o1] = _a[m - 1 - _r]
                        break
            # print("new 0:", top0)
            # print(group0, group1, top0, top1)
            # print(_a)
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        if top1_new == 1:
            # print("groups", group0, group1)
            # print("_u", _u)
            for k in range(0, group_size - 1):
                group1[k] = group1[k + 1]
            group1[group_size - 1] = top1
            # checking the opposite group
            o0 = -1
            for k in range(group_size):
                if group0[k] == top1:
                    o0 = k
                    break
            if o0 != -1:
                # print("o0", o0)
                for _r in range(m):
                    if _u[_a[_r]] == 0:
                        group0[o0] = _a[_r]
                        break
            # print("new 1:", top1)
            # print(group0, group1, top0, top1)
            # print(_a)
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        _i0 = group_size
        _i1 = m - group_size - 1
        for k in range(group_size):
            if group0[k] >= m - group_size:
                while _u[_i0] > 0:
                    _i0 += 1
                group0[k] = _a[_i0]
            if group1[k] < group_size:
                while _u[_i1] > 0:
                    _i1 += 1
                group1[k] = _a[_i1]
        for k in range(group_size):
            rsi[j, group0[k]] = 1
            rsi[j, group1[k]] = -1
    _groups = [0] * (2 * group_size)
    for k in range(group_size):
        _groups[k] = group0[k]
        _groups[group_size + k] = group1[k]
    return rsi, _groups, rr[-1, :]


@jit(nopython=True)
def calc_rsi_gA_p0(dP, group_size, groups):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    group0 = [0] * group_size
    group1 = [0] * group_size
    gs0 = [0] * group_size
    gs1 = [0] * group_size
    stat = {
        "new_top0": 0,
        "new_top1": 0,
    }
    j0 = 0
    if sum(groups) == -2 * group_size:
        j0 = 1
        k0 = 0
        k1 = 0
        for i in range(m):
            x = rr[0, i]
            if k0 < group_size and x < group_size:
                group0[k0] = i
                k0 += 1
            if k1 < group_size and x >= m - group_size:
                group1[k1] = i
                k1 += 1
        _a = [0] * m
        for _i in range(m):
            _a[int(rr[0, _i])] = _i
    else:
        for k in range(group_size):
            group0[k] = groups[k]
            group1[k] = groups[group_size + k]
    for j in range(j0, n):
        _a = [0] * m
        _u = [0] * m
        for _i in range(m):
            _a[int(rr[j, _i])] = _i     # _a: rank --> i (symbol)
        for _k in range(group_size):
            _u[group0[_k]] = 1          # _u: i (symbol) --> 0|1, if "i" is used
            _u[group1[_k]] = 1

        # finding new tops
        top0 = -1
        top1 = -1
        for i in range(m):
            x = rr[j, i]
            if x == 0:
                top0 = i
            elif x == m - 1:
                top1 = i
        top0_new = 1
        top1_new = 1
        top0_k = -1
        top1_k = -1
        for k in range(group_size):
            if group0[k] == top0:
                top0_new = 0
                top0_k = k
            if group1[k] == top1:
                top1_new = 0
                top1_k = k

        # pushing tops if already in the corresponding group
        # if top0 != group0[-1]:
        #     for k in range(top0_k, group_size-1):
        #         group0[k] = group0[k+1]
        #     group0[-1] = top0
        # if top1 != group1[-1]:
        #     for k in range(top1_k, group_size-1):
        #         group1[k] = group1[k+1]
        #     group1[-1] = top1

        # pushing all elements in order to insert a new top
        if top0_new == 1:
            stat["new_top0"] += 1
            for k in range(0, group_size - 1):
                group0[k] = group0[k + 1]
            group0[group_size - 1] = top0
            # checking the opposite group
            o1 = -1
            for k in range(group_size):
                if group1[k] == top0:
                    o1 = k
                    break
            if o1 != -1:
                for _r in range(m):
                    if _u[_a[m - 1 - _r]] == 0 and _a[m - 1 - _r] != top1:
                        group1[o1] = _a[m - 1 - _r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        if top1_new == 1:
            stat["new_top1"] += 1
            for k in range(0, group_size - 1):
                group1[k] = group1[k + 1]
            group1[group_size - 1] = top1
            # checking the opposite group
            o0 = -1
            for k in range(group_size):
                if group0[k] == top1:
                    o0 = k
                    break
            if o0 != -1:
                for _r in range(m):
                    if _u[_a[_r]] == 0 and _a[_r] != top0:
                        group0[o0] = _a[_r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1

        # out_of_group_n = 0
        # for _k in range(0, group_size):
        #     gs0[_k] = 0
        #     if int(rr[j, group0[_k]]) >= group_size:
        #         gs0[_k] = 1
        #         out_of_group_n += 1
        #     gs1[_k] = 0
        #     if int(rr[j, group1[_k]]) < m - group_size:
        #         gs1[_k] = 1
        #         out_of_group_n += 1

        # if out_of_group_n > 1 or True:
        #     # print("rebuilding...    ", group0, group1)
        #     k0 = 0
        #     k1 = 0
        #     for i in range(m):
        #         x = round(rr[j, i])
        #         if k0 < group_size and x < group_size:
        #             group0[k0] = i
        #             k0 += 1
        #         if k1 < group_size and x >= m - group_size:
        #             group1[k1] = i
        #             k1 += 1

        # fixing a signal
        for k in range(group_size):
            rsi[j, group0[k]] = 1
            rsi[j, group1[k]] = -1
    _groups = [0] * (2 * group_size)
    for k in range(group_size):
        _groups[k] = group0[k]
        _groups[group_size + k] = group1[k]
    # print(group0, group1)
    return rsi, _groups, rr[-1, :], stat


@jit(nopython=True)
def calc_rsi_gA_p0_dt(dP, P, dt, group_size, groups):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    group0 = [0] * group_size
    group1 = [0] * group_size
    # gs0 = [0] * group_size
    # gs1 = [0] * group_size
    stat = {
        "new_top0": 0,
        "new_top1": 0,
    }
    j0 = 0
    if sum(groups) == -2 * group_size:
        j0 = 1
        k0 = 0
        k1 = 0
        for i in range(m):
            x = rr[0, i]
            if k0 < group_size and x < group_size:
                group0[k0] = i
                k0 += 1
            if k1 < group_size and x >= m - group_size:
                group1[k1] = i
                k1 += 1
        _a = [0] * m
        for _i in range(m):
            _a[int(rr[0, _i])] = _i
    else:
        for k in range(group_size):
            group0[k] = groups[k]
            group1[k] = groups[group_size + k]
    top0_j = -1
    top1_j = -1
    top0_sym = -1
    top1_sym = -1
    for j in range(j0, n):
        _a = [0] * m
        _u = [0] * m
        for _i in range(m):
            _a[int(rr[j, _i])] = _i     # _a: rank --> i (symbol)
        for _k in range(group_size):
            _u[group0[_k]] = 1          # _u: i (symbol) --> 0|1, if "i" is used
            _u[group1[_k]] = 1

        # finding new tops
        top0 = -1
        top1 = -1
        for i in range(m):
            x = rr[j, i]
            if x == 0:
                top0 = i
            elif x == m - 1:
                top1 = i
        top0_new = 1
        top1_new = 1
        top0_k = -1
        top1_k = -1
        top0_in = 0
        top1_in = 0
        for k in range(group_size):
            if group0[k] == top0:
                top0_new = 0
                top0_k = k
            if group1[k] == top1:
                top1_new = 0
                top1_k = k
            if group0[k] == top0_sym:
                top0_in = 1
            if group1[k] == top1_sym:
                top1_in = 1

        # x0, x1 = np.min(P[j, :]), np.max(P[j, :])
        # z0, z1 = np.min(dP[j, :]), np.max(dP[j, :])
        # if dP[j, top0] != z0:
        #     print("?", z0, dP[j, top0])
        z0, z1 = -1, -1
        for _i in range(m):
            _z = dP[j, _i]
            if _i != top0:
                if z0 == -1:
                    z0 = _z
                elif _z < z0:
                    z0 = _z
            if _i != top1:
                if z1 == -1:
                    z1 = _z
                elif _z > z1:
                    z1 = _z

        if dP[j, top0] > z0 - dt:
            top0_new = 0
        if dP[j, top1] < z1 + dt:
            top1_new = 0

        # if top0_new == 1:
        #     top0_new = 0
        #     if top0_sym == top0:
        #         if j >= top0_j + dt:
        #             top0_new = 1
        #     else:
        #         top0_j, top0_sym = j, top0
        #         if j >= top0_j + dt:
        #             top0_new = 1
        # else:
        #     if top0_in:
        #         if j >= top0_j + dt:
        #             top0_new = 1
        #             top0 = top0_sym
        #     if top0_new == 0:
        #         top0_j, top0_sym = -1, -1
        # if top1_new == 1:
        #     top1_new = 0
        #     if top1_sym == top1:
        #         if j >= top1_j + dt:
        #             top1_new = 1
        #     else:
        #         top1_j, top1_sym = j, top1
        #         if j >= top1_j + dt:
        #             top1_new = 1
        # else:
        #     if top1_in:
        #         if j >= top1_j + dt:
        #             top1_new = 1
        #             top1 = top1_sym
        #     if top1_new == 0:
        #         top1_j, top1_sym = -1, -1

        # pushing all elements in order to insert a new top
        if top0_new == 1:
            stat["new_top0"] += 1
            for k in range(0, group_size - 1):
                group0[k] = group0[k + 1]
            group0[group_size - 1] = top0
            # checking the opposite group
            o1 = -1
            for k in range(group_size):
                if group1[k] == top0:
                    o1 = k
                    break
            if o1 != -1:
                for _r in range(m):
                    if _u[_a[m - 1 - _r]] == 0 and _a[m - 1 - _r] != top1:
                        group1[o1] = _a[m - 1 - _r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
            top0_j, top0_sym = -1, -1
        if top1_new == 1:
            stat["new_top1"] += 1
            for k in range(0, group_size - 1):
                group1[k] = group1[k + 1]
            group1[group_size - 1] = top1
            # checking the opposite group
            o0 = -1
            for k in range(group_size):
                if group0[k] == top1:
                    o0 = k
                    break
            if o0 != -1:
                for _r in range(m):
                    if _u[_a[_r]] == 0 and _a[_r] != top0:
                        group0[o0] = _a[_r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
            top1_j, top1_sym = -1, -1

        # fixing a signal
        for k in range(group_size):
            rsi[j, group0[k]] = 1
            rsi[j, group1[k]] = -1
    _groups = [0] * (2 * group_size)
    for k in range(group_size):
        _groups[k] = group0[k]
        _groups[group_size + k] = group1[k]
    # print(group0, group1)
    return rsi, _groups, rr[-1, :], stat


@jit(nopython=True)
def get_max(a, b):
    n = a.shape[0]
    mx = np.zeros(n)
    for j in range(n):
        mx[j] = max(a[j], b[j])
    return mx


@jit(nopython=True)
def calc_rsi_g2_p0(dP, group_size, w, groups):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    group0 = [0] * group_size
    group1 = [0] * group_size
    top0 = [0] * w
    top1 = [0] * w
    j0 = 0
    if sum(groups) == -2 * group_size:
        j0 = 1
        k0 = 0
        k1 = 0
        for i in range(m):
            x = rr[0, i]
            if k0 < group_size and x < group_size:
                group0[k0] = i
                k0 += 1
            if k1 < group_size and x >= m - group_size:
                group1[k1] = i
                k1 += 1
        _a = [0] * m
        for _i in range(m):
            _a[int(rr[0, _i])] = _i
    else:
        for k in range(group_size):
            group0[k] = groups[k]
            group1[k] = groups[group_size + k]
    for j in range(j0, n):
        _a = [0] * m
        _u = [0] * m
        for _i in range(m):
            _a[int(rr[j, _i])] = _i
        for _k in range(group_size):
            _u[group0[_k]] = 1
            _u[group1[_k]] = 1
        for v in range(w):
            top0[v] = -1
            top1[v] = -1
        for i in range(m):
            x = rr[j, i]
            if x < w:
                top0 = i
            elif x == m - 1:
                top1 = i
        top0_new = 1
        top1_new = 1
        for k in range(group_size):
            if group0[k] == top0:
                top0_new = 0
            if group1[k] == top1:
                top1_new = 0
        if top0_new == 1:
            for k in range(0, group_size - 1):
                group0[k] = group0[k + 1]
            group0[group_size - 1] = top0
            # checking the opposite group
            o1 = -1
            for k in range(group_size):
                if group1[k] == top0:
                    o1 = k
                    break
            if o1 != -1:
                for _r in range(m):
                    if _u[_a[m - 1 - _r]] == 0:
                        group1[o1] = _a[m - 1 - _r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        if top1_new == 1:
            for k in range(0, group_size - 1):
                group1[k] = group1[k + 1]
            group1[group_size - 1] = top1
            # checking the opposite group
            o0 = -1
            for k in range(group_size):
                if group0[k] == top1:
                    o0 = k
                    break
            if o0 != -1:
                for _r in range(m):
                    if _u[_a[_r]] == 0:
                        group0[o0] = _a[_r]
                        break
            for _i in range(m):
                _u[_i] = 0
            for _k in range(group_size):
                _u[group0[_k]] = 1
                _u[group1[_k]] = 1
        for k in range(group_size):
            rsi[j, group0[k]] = 1
            rsi[j, group1[k]] = -1
    _groups = [0] * (2 * group_size)
    for k in range(group_size):
        _groups[k] = group0[k]
        _groups[group_size + k] = group1[k]
    return rsi, _groups, rr[-1, :]


@jit(nopython=True)
def calc_rsi_delayed_t(dP, w, delta, thr, mask):
    rg = dP
    rr = calc_rrank_mask(rg, mask)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    mn = np.zeros(dP.shape)
    mx = np.zeros(dP.shape)
    p0 = 1.0
    low_rank = high_rank = None
    for i in range(m):
        if mask[i] == 0:
            continue
        if low_rank is None:
            low_rank = i
        high_rank = i
    print("low and high rank:", low_rank, high_rank)
    for i in range(m):
        if mask[i] == 0:
            continue
        for j in range(n):
            _mn = None
            _mx = None
            for k in range(m):
                if k == i or mask[k] == 0:
                    continue
                x = dP[j, k]
                if _mn is None:
                    _mn = x
                elif x < _mn:
                    _mn = x
                if _mx is None:
                    _mx = x
                elif x > _mx:
                    _mx = x
            mn[j, i] = _mn
            mx[j, i] = _mx
    for i in range(m):
        if mask[i] == 0:
            continue
        t = thr[i]
        i0 = -1
        for j in range(1, n):
            r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < low_rank + w:
                    # if dP[j, i] <= mn[j, i] - t and np.random.uniform() <= p0:  # and dP[j, i0] - dP[j-1, i0] < 0:
                    #     rsi[j, i] = 1
                    # else:
                    #     rsi[j, i] = 0
                    rsi[j, i] = 1
                elif x > high_rank - w:
                    # if dP[j, i] >= mx[j, i] + t and np.random.uniform() <= p0:  # and dP[j, i0] - dP[j-1, i0] > 0:
                    #     rsi[j, i] = -1
                    # else:
                    #     rsi[j, i] = 0
                    rsi[j, i] = -1
            elif r < 0:
                if x < low_rank + w:
                    # if dP[j, i] <= mn[j, i] - t and np.random.uniform() <= p0:  # and dP[j, i0] - dP[j-1, i0] < 0:
                    #     rsi[j, i] = 1
                    # else:
                    #     rsi[j, i] = 0
                    rsi[j, i] = 1
                elif x <= high_rank - w - delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
            else:
                if x > high_rank - w:
                    # if dP[j, i] >= mx[j, i] + t and np.random.uniform() <= p0:  # and dP[j, i0] - dP[j-1, i0] > 0:
                    #     rsi[j, i] = -1
                    # else:
                    #     rsi[j, i] = 0
                    rsi[j, i] = -1
                elif x >= low_rank + w + delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
    return rsi


@jit(nopython=True)
def inside_390(x, offset):
    n = x.shape[0]
    # n_days = n // 390
    a = np.zeros(n)
    for j in range(n):
        t = j % 390
        if offset <= t < 390 - offset:
            a[j] = x[j]
    return a


@jit(nopython=True)
def smart_enter_pos(sig, mid, h):
    Pos = 0
    enter_mid = 0
    enter_j = 0
    n = sig.shape[0]
    position = np.zeros(n)
    for j in range(1, n):
        s = sig[j]
        # exit position
        if Pos > 0:
            if s <= 0:
                Pos = 0
        elif Pos < 0:
            if s >= 0:
                Pos = 0
        # enter position
        if s > 0 >= sig[j - 1]:
            dP1 = mid[j] - mid[j - 1]
            enter_mid = mid[j] - h if dP1 > 0 else mid[j]
            enter_j = j
        if s < 0 <= sig[j - 1]:
            dP1 = mid[j] - mid[j - 1]
            enter_mid = mid[j] + h if dP1 > 0 else mid[j]
            enter_j = j
        if s == 0:
            enter_mid = 0
            enter_j = 0
        if Pos == 0:
            if s > 0:
                if mid[j] <= enter_mid:  # (mid[j] <= enter_mid - h and dP1 < 0) or (mid[j] <= enter_mid and dP1 >= 0):
                    Pos = s
            elif s < 0:
                if mid[j] <= enter_mid:  # (mid[j] >= enter_mid + h and dP1 > 0) or (mid[j] >= enter_mid and dP1 <= 0):
                    Pos = s
        position[j] = Pos
    # print(sum(position))
    return position


@jit(nopython=True)
def smart_enter_pos_p0(sig, mid, h, ini_Pos, ini_enter_mod, ini_enter_j):
    Pos = ini_Pos
    enter_mid = ini_enter_mod
    enter_j = ini_enter_j
    n = sig.shape[0]
    position = np.zeros(n)
    for j in range(1, n):
        s = sig[j]
        # exit position
        if Pos > 0:
            if s <= 0:
                Pos = 0
        elif Pos < 0:
            if s >= 0:
                Pos = 0
        # enter position
        if s > 0 >= sig[j - 1]:
            enter_mid = mid[j]
            enter_j = j
        if s < 0 <= sig[j - 1]:
            enter_mid = mid[j]
            enter_j = j
        if s == 0:
            enter_mid = 0
            enter_j = 0
        if Pos == 0:
            if s > 0:
                if mid[j] >= enter_mid + h:
                    Pos = 1
            elif s < 0:
                if mid[j] <= enter_mid - h:
                    Pos = -1
        position[j] = Pos
    # print(sum(position))
    return position, Pos, enter_mid, enter_j


@jit(nopython=True)
def round_to(sig, s):
    n = sig.shape[0]
    a = np.zeros(n)
    for j in range(n):
        x = sig[j]
        if x >= 0:
            a[j] = s * int(.5 + x/s)
        else:
            a[j] = - s * int(.5 - x/s)
    return a


@jit(nopython=True)
def calc_switchers(sig):
    n = sig.shape[0]
    m = sig.shape[1]
    a = np.zeros(sig.shape)
    state = 0
    for i in range(m):
        for j in range(n):
            x = sig[j, i]
            if x > 0:
                if state <= 0:
                    a[j, i] = 1
                    state = 1
            elif x < 0:
                if state >= 0:
                    a[j, i] = -1
                    state = -1
    return a


@jit(nopython=True)
def calc_switcher_volume(swt):
    n = swt.shape[0]
    m = swt.shape[1]
    volume = np.zeros(m)
    for i in range(m):
        V = 0
        for j in range(n):
            if swt[j, i] != 0:
                V += 1
        volume[i] = V
    return volume


@jit(nopython=True)
def expand_switchers(sig, max_len):
    n = sig.shape[0]
    m = sig.shape[1]
    a = np.zeros(sig.shape)
    state = 0
    state_j = 0
    for i in range(m):
        for j in range(n):
            x = sig[j, i]
            if x > 0:
                state = 1
                state_j = j
            elif x < 0:
                state = -1
                state_j = j
            if j <= state_j + max_len:
                a[j, i] = state
            else:
                a[j, i] = 0
    return a


@jit(nopython=True)
def get_order(s):
    m = s.shape[0]
    # print(m)
    if m == 3:
        x, y, z = s[:3]
        if x < y:
            if y < z:
                return 0
            elif x < z:
                return 1
            else:
                return 2
        else:
            if x < z:
                return 3
            elif y < z:
                return 4
            else:
                return 5
    elif m == 4:
        x, y, z, w = s[:4]
        if x < y:
            if y < z:  # xyz
                if z < w:
                    return 0
                elif y < w:
                    return 1
                elif x < w:
                    return 2
                else:
                    return 3
            elif x < z:  # xzy
                if y < w:
                    return 4
                elif z < w:
                    return 5
                elif x < w:
                    return 6
                else:
                    return 7
            else:  # zxy
                if y < w:
                    return 8
                elif x < w:
                    return 9
                elif z < w:
                    return 10
                else:
                    return 11
        else:
            if x < z:  # yxz
                if z < w:
                    return 12
                elif x < w:
                    return 13
                elif y < w:
                    return 14
                else:
                    return 15
            elif y < z:  # yzx
                if x < w:
                    return 16
                elif z < w:
                    return 17
                elif y < w:
                    return 18
                else:
                    return 19
            else:  # zyx
                if x < w:
                    return 20
                elif y < w:
                    return 21
                elif z < w:
                    return 22
                else:
                    return 23
    return 0


@jit(nopython=True)
def encode_by_sectors(sig, dP, n_sectors, components):
    add_sign_i = -1
    n = sig.shape[0]
    m = sig.shape[1]
    codes = np.zeros(n) - 1
    b = np.zeros(n_sectors)
    for i in range(m):
        b[components[i]] += 1
    s = np.zeros(n_sectors)
    # print(n_sectors)
    for j in range(n):
        f_encode = 0
        for i in range(m):
            if sig[j, i] != 0:
                f_encode = 1
                break
        if f_encode != 0:
            for k in range(n_sectors):
                s[k] = 0
            for i in range(m):
                k = components[i]
                s[k] += dP[j, i]
            for k in range(n_sectors):
                s[k] /= b[k]
            codes[j] = get_order(s)
            if add_sign_i >= 0:
                if n_sectors == 2:
                    if sig[j, add_sign_i] < 0:
                        codes[j] += 2
                elif n_sectors == 3:
                    if sig[j, add_sign_i] < 0:
                        codes[j] += 6
                elif n_sectors == 4:
                    if sig[j, add_sign_i] < 0:
                        codes[j] += 24
            # print(s, codes[j])
    return codes


@jit(nopython=True)
def calc_code_statistics(codes, n_codes):
    n = codes.shape[0]
    s = [0] * n_codes
    for j in range(n):
        code = int(codes[j])
        if code >= 0:
            s[code] += 1
    return s


@jit(nopython=True)
def apply_interval_statistics(sig, codes, n_codes, IY):
    sig2 = np.zeros(sig.shape)
    n = sig.shape[0]
    code = -1
    signal = 0
    for j in range(1, n):
        if sig[j] != 0:
            if sig[j] > 0 >= sig[j - 1] or sig[j] < 0 <= sig[j - 1]:
                code = int(codes[j])
                if sig[j] < 0:
                    code += n_codes
                if IY[code] > 0:
                    signal = 1
                elif IY[code] < 0:
                    signal = -1
                # signal = sig[j]
            if code != -1:
                sig2[j] = signal
    return sig2


@jit(nopython=True)
def calc_interval_statistics(mid, sig, codes, n_codes):
    n = mid.shape[0]
    IL = np.zeros(2 * n_codes)
    IN = np.zeros(2 * n_codes)
    IY = np.zeros(2 * n_codes)
    interval_sig = 0
    interval_code = -1
    interval_enter_mid = 0
    for j in range(n):
        # closing
        if interval_sig > 0 >= sig[j]:
            IY[interval_code] += mid[j] - interval_enter_mid
            IN[interval_code] += 1
            interval_code = -1
        elif interval_sig < 0 <= sig[j]:
            IY[interval_code] += mid[j] - interval_enter_mid
            IN[interval_code] += 1
            interval_code = -1
        # opening
        if sig[j] > 0:
            if j == 0 or sig[j - 1] <= 0:
                interval_code = int(codes[j])
                interval_sig = sig[j]
                interval_enter_mid = mid[j]
        elif sig[j] < 0:
            if j == 0 or sig[j - 1] >= 0:
                interval_code = int(codes[j]) + n_codes
                interval_sig = sig[j]
                interval_enter_mid = mid[j]
        if interval_code != -1:
            IL[interval_code] += 1

    for code in range(2 * n_codes):
        if IN[code] > 0:
            IY[code] /= IN[code]
    # print(IL)
    return IY


@jit(nopython=True)
def discretize_by_threshold(X, thr):
    n = X.shape[0]
    Y = np.zeros(n)
    for j in range(n):
        x = X[j]
        if x >= thr:
            Y[j] = 1
        elif x <= -thr:
            Y[j] = -1
    return Y


@jit(nopython=True)
def discretize_signal(
    X, enter_long_pos, exit_long_pos, enter_short_pos, exit_short_pos
):
    n = X.shape[0]
    sig = np.zeros(n)
    Pos = 0
    for j in range(n):
        x = X[j]
        if Pos > 0:
            if x >= -exit_long_pos:
                Pos = 0
        elif Pos < 0:
            if x <= exit_short_pos:
                Pos = 0
        if x <= -enter_long_pos:
            Pos = 1
        elif x >= enter_short_pos:
            Pos = -1
        sig[j] = Pos
    return sig


@jit(nopython=True)
def discretize_signal_p0(
    X, enter_long_pos, exit_long_pos, enter_short_pos, exit_short_pos, Pos0
):
    n = X.shape[0]
    sig = np.zeros(n)
    Pos = Pos0
    for j in range(n):
        x = X[j]
        if Pos > 0:
            if x >= -exit_long_pos:
                Pos = 0
        elif Pos < 0:
            if x <= exit_short_pos:
                Pos = 0
        if x <= -enter_long_pos:
            Pos = 1
        elif x >= enter_short_pos:
            Pos = -1
        sig[j] = Pos
    return sig


@jit(nopython=True)
def ffill_zeros(X):
    n = X.shape[0]
    Y = np.zeros(n)
    Y[0] = X[0]
    for j in range(1, n):
        x = X[j]
        if x == 0:
            Y[j] = Y[j-1]
        else:
            Y[j] = x
    return Y

def calc_median(X):
    return np.percentile(X, 50)


def get_daily_risk(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    day_dpnl = calc_diff0(day_pnl)
    return np.std(day_dpnl)


def get_daily_sharpe(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    # print(day_pnl)
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    # print(day_pnl)
    # print("daily sharpe", np.sum(dpnl), pnl[-1], day_pnl[-1])
    day_dpnl = calc_diff0(day_pnl)
    # print(day_dpnl)
    # print("sum of diffs", sum(day_dpnl))
    sharpe = np.mean(day_dpnl) / np.std(day_dpnl)
    return sharpe


def get_daily_sharpe_ida(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    pnl = calc_qema1(pnl, 1/day_ticks)
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    # print(day_pnl)
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    # print(day_pnl)
    # print("daily sharpe", np.sum(dpnl), pnl[-1], day_pnl[-1])
    day_dpnl = calc_diff0(day_pnl)
    # print(day_dpnl)
    # print("sum of diffs", sum(day_dpnl))
    sharpe = np.mean(day_dpnl) / np.std(day_dpnl)
    return sharpe


def get_daily_sharpe_sm(dpnl, day_ticks, day_alpha):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    day_dpnl = calc_diff0(day_pnl)
    avg_dpnl = calc_qema1(day_dpnl, day_alpha)
    y = day_dpnl - avg_dpnl
    s = np.linalg.norm(y) / np.sqrt(y.shape[0])
    sharpe = np.mean(day_dpnl) / s
    return sharpe


def get_sortino(X: Any, a: float, *, delta: float = 0):
    dX = X - a
    dXn = 0.5 * (np.abs(dX) - dX)
    sigma = np.linalg.norm(dXn) / np.sqrt(X.shape[0])
    return (np.mean(X) - delta) / sigma


def get_daily_sortino_0(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    day_dpnl = calc_diff0(day_pnl)
    sortino = get_sortino(day_dpnl, 0)
    return sortino


def get_daily_sortino_to_median(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    day_dpnl = calc_diff0(day_pnl)
    sortino = get_sortino(day_dpnl, calc_median(day_dpnl))
    return sortino


def get_daily_sortino_to_avg(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    day_dpnl = calc_diff0(day_pnl)
    sortino = get_sortino(day_dpnl, float(np.mean(day_dpnl)))
    return sortino


def get_daily_sortino_to_havg(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(day_ticks-1, pnl.shape[0], day_ticks)])
    if len(day_pnl) > 0:
        day_pnl[-1] = pnl[-1]
    day_dpnl = calc_diff0(day_pnl)
    sortino = get_sortino(day_dpnl, 0.5 * np.mean(day_dpnl))
    return sortino


def get_daily_dd(dpnl, day_ticks):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(0, pnl.shape[0], day_ticks)])
    day_dpnl = calc_diff(day_pnl)
    return np.min(day_dpnl)


@jit(nopython=True)
def get_drawdown(dpnl):
    pnl = dpnl.cumsum()
    n = pnl.shape[0]
    dd = np.zeros(n)
    mx = 0
    for j in range(n):
        x = pnl[j]
        if x > mx:
            mx = x
        dd[j] = mx - x
    return dd


@jit(nopython=True)
def get_max_drawdown(dpnl):
    pnl = dpnl.cumsum()
    n = pnl.shape[0]
    dd = 0
    mx = 0
    for j in range(n):
        x = pnl[j]
        if x > mx:
            mx = x
        if x < mx:
            if mx - x > dd:
                dd = mx - x
    return dd


def get_daily_sharpe_smooth(dpnl, day_ticks, alpha):
    pnl = dpnl.cumsum()
    day_pnl = np.array([pnl[j] for j in range(0, pnl.shape[0], day_ticks)])
    day_dpnl = calc_diff(day_pnl)
    pnl2 = calc_qema1(pnl, alpha)
    day_pnl2 = np.array([pnl2[j] for j in range(0, pnl2.shape[0], day_ticks)])
    day_dpnl2 = calc_diff(day_pnl2)
    sharpe = np.mean(day_dpnl) / np.std(day_dpnl2)
    return sharpe


@jit(nopython=True)
def get_long_short_imbalance(sig):
    n = sig.shape[0]
    n_long = n_short = 0
    for j in range(n):
        x = sig[j]
        if x > 0:
            n_long += 1
        elif x < 0:
            n_short += 1
    if n_long == 0 and n_short == 0:
        return 0
    return (n_long - n_short) / (n_long + n_short)


def restrict_with_ticks(dP1, ticks):
    P1 = dP1.cumsum()
    P1r = np.array([P1[j] for j in range(ticks-1, P1.shape[0], ticks)])
    dP1r = calc_diff0(P1r)
    return dP1r


def calc_corr_on_periods(dP1, dP2, ticks):
    dP1r = restrict_with_ticks(dP1, ticks)
    dP2r = restrict_with_ticks(dP2, ticks)
    corr = dP1r.dot(dP2r) / np.linalg.norm(dP1r) / np.linalg.norm(dP2r)
    return corr


@jit(nopython=True)
def get_holding_time_p(sig):
    n = sig.shape[0]
    N = 0
    for j in range(n):
        x = sig[j]
        if x != 0:
            N += 1
    return N / n


@jit(nopython=True)
def get_true_range(O, H, L, C):
    n = C.shape[0]
    TR = np.zeros(n)
    for j in range(n):
        TR[j] = max(H[j] - L[j], np.abs(H[j] - C[j]), np.abs(H[j] - C[j]))
    return TR


@jit(nopython=True)
def repeat_blocks(X, r):
    n = X.shape[0]
    Y = np.zeros(n * r)
    for i in range(n):
        for j in range(r):
            Y[i * r + j] = X[i]
    return Y


@jit(nopython=True)
def restrict_to_lattice(X, r, phase):
    n = X.shape[0]
    Y = np.zeros(n // r)
    for i in range(n // r):
        Y[i] = X[i * r + phase]
    return Y


@jit(nopython=True)
def fit_in_interval(X, a, b, n):
    n1 = b - a
    Y = np.zeros(n)
    for i in range(n1):
        Y[a+i] = X[i]
    return Y


@jit(nopython=True)
def const_on_interval(x, a, b, n):
    Y = np.zeros(n)
    for i in range(n):
        if a <= i < b:
            Y[i] = x
    return Y


@jit(nopython=True)
def linear_segment(x0, x1, n):
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = x0 + (x1 - x0) * i / n
    return Y


@jit(nopython=True)
def fillr_zeros(X):
    n = X.shape[0]
    Y = np.zeros(n)
    for i in range(n):
        j = n - 1 - i
        if X[j] == 0 and i > 0:
            Y[j] = Y[j + 1]
        else:
            Y[j] = X[j]
    for j in range(1, n):
        if Y[j] == 0:
            Y[j] = Y[j - 1]
    return Y


@jit(nopython=True)
def ema(X, alpha):
    n = X.shape[0]
    a = np.zeros(n)
    y = X[0]
    a[0] = y
    for j in range(1, n):
        y += alpha * (X[j] - y)
        a[j] = y
    return a


def get_atr_vol(O, H, L, C, alpha):
    TR = get_true_range(O, H, L, C)
    ATR = ema(TR, alpha)
    return ATR


def get_normalized_delta(O, H, L, C, averaged_mid, atr_alpha):
    return (C - averaged_mid) / get_atr_vol(O, H, L, C, atr_alpha)


def get_bars_from_q(mid, resolution):
    n = mid.shape[0]
    N = n // resolution
    O = np.zeros(N)
    H = np.zeros(N)
    L = np.zeros(N)
    C = np.zeros(N)
    for k in range(N - 1):
        O[k] = mid[k * resolution]
        C[k] = mid[(k + 1) * resolution]
        H[k] = np.max(mid[k * resolution : (k + 1) * resolution + 1])
        L[k] = np.min(mid[k * resolution : (k + 1) * resolution + 1])
    O[N - 1] = mid[(N - 1) * resolution]
    C[N - 1] = mid[n - 1]
    H[N - 1] = np.max(mid[(N - 1) * resolution : n])
    L[N - 1] = np.min(mid[(N - 1) * resolution : n])
    return O, H, L, C


@jit(nopython=True)
def calc_switcher_intervals(swt):
    volume = calc_switcher_volume(swt)
    n = swt.shape[0]
    m = swt.shape[1]
    intervals = np.zeros(m)
    for i in range(m):
        if volume[i] > 0:
            intervals[i] = n / volume[i]
    return intervals


@jit(nopython=True)
def calc_rsi1(dP, w, w_delta, stay):
    rg = dP
    rr = calc_rrank(rg)
    # rr = rg
    rsi = np.zeros(dP.shape)
    n = dP.shape[0]
    m = dP.shape[1]
    for i in range(m):
        delta = w_delta[i]
        for j in range(1, n):
            r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < w:
                    rsi[j, i] = 1
                elif x >= m - w:
                    rsi[j, i] = -1
            elif r < 0:
                if x < w:
                    rsi[j, i] = 1
                elif x < m - w - delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
            else:
                if x >= m - w:
                    rsi[j, i] = -1
                elif x >= w + delta:
                    rsi[j, i] = 0
                else:
                    rsi[j, i] = r
        if stay == 1:
            c = np.zeros(n)
            for j in range(1, n):
                if rsi[j - 1, i] == rsi[j, i]:
                    c[j] = rsi[j, i]
                else:
                    c[j] = c[j - 1]
            rsi[:, i] = c
        elif stay == 2:
            c = np.zeros(n)
            for j in range(2, n):
                if rsi[j - 1, i] == rsi[j, i] and rsi[j - 2, i] == rsi[j, i]:
                    c[j] = rsi[j, i]
                else:
                    c[j] = c[j - 1]
            rsi[:, i] = c

    return rsi


@jit(nopython=True)
def calc_ultra_rsi(dP):
    thr = 0.8
    n = dP.shape[0]
    m = dP.shape[1]
    rsi = np.zeros(dP.shape)
    for j in range(n):
        for i1 in range(m - 1):
            for i2 in range(i1 + 1, m):
                if dP[j, i1] > dP[j, i2] + thr:
                    rsi[j, i1] += 1
                elif dP[j, i1] < dP[j, i2] - thr:
                    rsi[j, i2] += 1

    return 1 - 2 * rsi / m


@jit(nopython=True)
def expand_nonzero_m(sig, l):
    a = np.zeros(sig.shape)
    n = sig.shape[0]
    m = sig.shape[1]
    for i in range(m):
        state_j = 0
        state_x = 0
        for j in range(0, n):
            x = sig[j, i]
            if x > 0:
                if state_x <= 0:
                    state_j = j
                    state_x = 1
                a[j, i] = 1
            elif x < 0:
                if state_x >= 0:
                    state_j = j
                    state_x = -1
                a[j, i] = -1
            else:
                if state_x != 0:
                    if j < state_j + l:
                        a[j, i] = state_x
                    else:
                        a[j, i] = 0
                        state_j = 0
                        state_x = 0
                else:
                    a[j, i] = 0
    return a


@jit(nopython=True)
def expand_signal_01_L(sig, period):
    n = sig.shape[0]
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = sig[i]
    i = 0
    while i < len(arr):
        if arr[i] == 1:
            for j in range(1, period):
                if i + j < len(arr):
                    arr[i + j] = 1
            i += period
        else:
            i += 1
    return arr


@jit(nopython=True)
def expand_signal_01(sig, s):
    a = np.zeros(sig.shape)
    n = sig.shape[0]
    state_j = -s - 1
    for j in range(0, n):
        x = sig[j]
        if x > 0:
            state_j = j
            a[j] = 1
        else:
            if j < state_j + s:
                a[j] = 1
            else:
                a[j] = 0
    return a


@jit(nopython=True)
def encode_signal_01_by_thld(sig, alpha, imb_thld_in=0.8, imb_thld_out=0.5):
    sig1 = calc_qema1(sig, alpha)
    n = sig.shape[0]
    sig2 = np.zeros(n)
    y = 0
    for j in range(0, n):
        x = sig1[j]
        if np.abs(x) < imb_thld_out:
            y = 0
        if x > imb_thld_in:
            y = 1
        elif x < -imb_thld_in:
            y = -1
        sig2[j] = y
    return sig2


@jit(nopython=True)
def resrict(X, m, phasw=0.8, imb_thld_out=0.5):
    sig1 = calc_qema1(sig, alpha)
    n = sig.shape[0]
    sig2 = np.zeros(n)
    y = 0
    for j in range(0, n):
        x = sig1[j]
        if np.abs(x) < imb_thld_out:
            y = 0
        if x > imb_thld_in:
            y = 1
        elif x < -imb_thld_in:
            y = -1
        sig2[j] = y
    return sig2


@jit(nopython=True)
def get_risk_est(dpnl, Y):
    n = dpnl.shape[0]
    a = np.zeros(n)
    for j in range(n-1):
        if Y[j] > 0:
            a[j+1] = dpnl[j+1]
    return a




@jit(nopython=True)
def calc_portfolio_rsi1(mid, alpha, lin_a, w, w_delta):
    rg = calc_rel_mid_growth(mid, alpha, lin_a)
    rr = calc_rrank(rg)
    rsi = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        for j in range(1, n):
            x = rr[j, i]
            r1 = 1 - 2 * x / (m - 1)
            if np.abs(r1) < 0.5:
                r1 = 0.0
            rsi[j, i] = r1

    return rsi


@jit(nopython=True)
def calc_portfolio_rsi2(mid, alpha, lin_a, w, w_delta):
    rg = calc_rel_mid_growth(mid, alpha, lin_a)
    rr = calc_rrank(rg)
    rsi = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        delta = w_delta[i]
        for j in range(1, n):
            r = rsi[j - 1, i]
            x = rr[j, i]
            if r == 0:
                if x < w:
                    rsi[j, i] = 1
                elif x >= m - w:
                    rsi[j, i] = -1
            elif r < 0:
                if x < w:
                    rsi[j, i] = 1
                elif x < 0.5 * (m - 1):
                    rsi[j, i] = 0
                else:
                    r1 = 1 - 2 * x / (m - 1)
                    if r1 > r:
                        rsi[j, i] = r1
                    else:
                        rsi[j, i] = r
            else:
                if x >= m - w:
                    rsi[j, i] = -1
                elif x > 0.5 * (m - 1):
                    rsi[j, i] = 0
                else:
                    r1 = 1 - 2 * x / (m - 1)
                    if r1 < r:
                        rsi[j, i] = r1
                    else:
                        rsi[j, i] = r

    return rsi


@jit(nopython=True)
def calc_limited_integral(sig, M, normalize=False):
    n = sig.shape[0]
    m = sig.shape[1]
    pos = np.zeros(sig.shape)
    for i in range(m):
        for j in range(1, n):
            if sig[j, i] != sig[j - 1, i] or True:
                p = pos[j - 1, i]
                v = p + sig[j, i]
                if v > M:
                    v = M
                elif v < -M:
                    v = -M
                pos[j, i] = v
            else:
                pos[j, i] = pos[j - 1, i]
    if normalize:
        pos /= M
    return pos


@jit(nopython=True)
def calc_n1_portfolio_rsi(mid, alpha, w, dt):
    rsi = calc_portfolio_rsi(mid, alpha, w)
    arsi = np.zeros(rsi.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    entered = np.zeros(m)
    for i in range(m):
        for j in range(1, n):
            last_pos = arsi[j - 1, i]
            pos = rsi[j, i]
            if pos > 0:
                if pos != last_pos:
                    entered[i] = j
            elif pos < 0:
                if pos != last_pos:
                    entered[i] = j
            else:  # pos == 0
                if last_pos != 0:
                    if j < entered[i] + dt:
                        pos = last_pos
                    else:
                        pos = 0
            arsi[j, i] = pos
    return arsi


@jit(nopython=True)
def calc_avg_portfolio_rsi(mid, alpha, w, beta):
    rsi = calc_portfolio_rsi(mid, alpha, w)
    arsi = calc_matrix_ema(rsi, beta)
    return arsi


@jit(nopython=True)
def remove_anomaly_spikes_m(mid, b):
    k = 1 + b
    n = mid.shape[0]
    m = mid.shape[1]
    a = mid.copy()
    for i in range(m):
        for s in range(1, 36, 3):
            for j in range(s, n - s):
                y = 0.5 * (a[j - s, i] + a[j + s, i])
                if a[j, i] > k * y:
                    a[j, i] = y
                elif a[j, i] < y / k:
                    a[j, i] = y
    return a


@jit(nopython=True)
def calc_simplified_portfolio_pnl_log(mid, sig):
    print("!!!")
    exit()
    n = mid.shape[0]
    dpnl = np.zeros(n)
    dV = np.zeros(n)
    diV = np.zeros(n)
    position = np.zeros(n)
    for j in range(1, n):
        if mid[j - 1] > 0:
            dpnl[j] = sig[j - 1] * (mid[j] / mid[j - 1] - 1)
        dpos = sig[j] - sig[j - 1]
        dV[j] = np.abs(dpos)
        diV[j] = 1
        position[j] = sig[j]
    return dpnl, dV, diV, position


@jit(nopython=True)
def calc_discrete_position(sig, k0, k1):
    H = avg_amp_l2(sig)
    state = 0
    h0 = k0 * H
    h1 = k1 * H
    n = sig.shape[0]
    sig2 = np.zeros(n)
    for j in range(n):
        if state == 1:
            if sig[j] <= h1:
                state = 0
        elif state == -1:
            if sig[j] >= -h1:
                state = 0
        if sig[j] >= h0:
            state = 1
        elif sig[j] <= -h0:
            state = -1
        sig2[j] = state
    return sig2 * (H / avg_amp_l2(sig2))


@jit(nopython=True)
def first_positive(sig):
    n = sig.shape[0]
    for j in range(n):
        if sig[j] > 0:
            return sig[j]
    return 0


@jit(nopython=True)
def first_positive_index(sig):
    n = sig.shape[0]
    for j in range(n):
        if sig[j] > 0:
            return j
    return 0


@jit(nopython=True)
def positive_part(sig):
    n = sig.shape[0]
    y = np.zeros(n)
    for j in range(n):
        if sig[j] > 0:
            y[j] = sig[j]
    return y


@jit(nopython=True)
def calc_staircase_position(sig, h):
    G = (np.sqrt(5) + 1) / 2
    dh = h / G
    Pos = 0
    n = sig.shape[0]
    sig2 = np.zeros(n)
    for j in range(n):
        x = sig[j]
        if x > Pos + dh or x < Pos - dh:
            Pos = h * np.round(x / h)
        sig2[j] = Pos
    return sig2


@jit(nopython=True)
def calc_simplified_portfolio_pnl(mid, sig):
    n = mid.shape[0]
    dpnl = np.zeros(n)
    dV = np.zeros(n)
    diV = np.zeros(n)
    position = np.zeros(n)
    position[0] = sig[0]
    for j in range(1, n):
        if mid[j - 1] > 0 and mid[j] > 0:
            dpnl[j] = sig[j - 1] * (mid[j] - mid[j - 1])
        dpos = sig[j] - sig[j - 1]
        dV[j] = np.abs(dpos)
        if dpos != 0:
            diV[j] = 1
        position[j] = sig[j]
    return dpnl, dV, diV, position


@jit(nopython=True)
def adjust_position1(mid, pos, A):
    n = pos.shape[0]
    apos = np.zeros(n)
    n_shares = round(A / mid[0])
    apos[0] = n_shares
    for j in range(1, n):
        if pos[j] != pos[j - 1]:
            n_shares = round(A / mid[j])
            apos[j] = n_shares
        else:
            apos[j] = apos[j - 1]
    return apos


# @jit(nopython=True)
# def adjust_to_aum(signal, aum, mid):
#     n = signal.shape[0]
#     y = np.zeros(n)
#     Pos = 0
#     for j in range(n):
#         x = signal[j]
#         if x > 0:
#             if Pos <= 0:
#                 Pos = aum / mid[j]
#         elif x < 0:
#             if Pos >= 0:
#                 Pos = -aum / mid[j]
#         else:
#             Pos = 0
#         y[j] = Pos
#     return y


@jit(nopython=True)
def adjust_to_aum_Y(signal, aum, mid, initial_Pos=0):
    n = signal.shape[0]
    y = np.zeros(n)
    Pos = initial_Pos
    for j in range(n):
        x = signal[j]
        if x > 0:
            if Pos <= 0:
                Pos = aum / mid[j]
        elif x < 0:
            if Pos >= 0:
                Pos = -aum / mid[j]
        else:
            Pos = 0
        y[j] = Pos
    return y


@jit(nopython=True)
def adjust_to_aum_X(signal, aum, mid, initial_Pos=0):
    n = signal.shape[0]
    y = np.zeros(n)
    Pos = initial_Pos
    for j in range(n):
        x = signal[j]
        if x > 0:
            if Pos == 0:
                Pos = aum / mid[j]
        elif x < 0:
            if Pos == 0:
                Pos = -aum / mid[j]
        else:
            Pos = 0
        y[j] = Pos
    return y


@jit(nopython=True)
def adjust_to_aum(signal, aum, mid, *, initial_Pos=0):
    return adjust_to_aum_X(signal, aum, mid, initial_Pos=initial_Pos)


@jit(nopython=True)
def adjust_c_to_aum_on_segments(aum, mid, period):
    n = mid.shape[0]
    position = np.zeros(n)
    Pos = 0
    for j in range(n):
        if j % period == 0:
            Pos = round(aum / mid[j])
        position[j] = Pos
    return position


@jit(nopython=True)
def calc_simplified_portfolio_pnl_adj1(mid, sig, A):
    pos = adjust_position1(mid, sig, A)
    pos = sig
    n = mid.shape[0]
    dpnl = np.zeros(n)
    dV = np.zeros(n)
    diV = np.zeros(n)
    for j in range(1, n):
        if mid[j - 1] > 0:
            dpnl[j] = pos[j - 1] * (mid[j] - mid[j - 1])
        dpos = pos[j] - pos[j - 1]
        dV[j] = np.abs(dpos)
        if dpos != 0:
            diV[j] = 1
    return dpnl, dV, diV, pos


@jit(nopython=True)
def calc_simplified_portfolio_pnl_m(mid, sig):
    dpnl = np.zeros(mid.shape)
    dV = np.zeros(mid.shape)
    diV = np.zeros(mid.shape)
    position = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        for j in range(1, n):
            if mid[j - 1, i] > 0:
                dpnl[j, i] = sig[j - 1, i] * (mid[j, i] - mid[j - 1, i])
            dpos = sig[j, i] - sig[j - 1, i]
            dV[j, i] = np.abs(dpos)
            if dpos != 0:
                diV[j, i] = 1
            position[j, i] = sig[j, i]

    return dpnl, dV, diV, position


@jit(nopython=True)
def calc_simplified_portfolio_pnl_log_m(mid, sig):
    print("!!!")
    exit()
    dpnl = np.zeros(mid.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    for i in range(m):
        for j in range(1, n):
            if mid[j - 1, i] > 0:
                dpnl[j, i] = sig[j - 1, i] * (mid[j, i] / mid[j - 1, i] - 1)
    return dpnl


@jit(nopython=True)
def calc_pos_e0(mid, sig, u):
    n = mid.shape[0]
    pos = np.zeros(n)
    pos[0] = sig[0]
    enter_mid = 0
    for j in range(n):
        if sig[j - 1] != sig[j]:
            enter_mid = mid[j]
        pos[j] = pos[j - 1]
        if pos[j - 1] != sig[j]:
            if sig[j] > 0:
                if mid[j] <= enter_mid - u:
                    pos[j] = sig[j]
            elif sig[j] < 0:
                if mid[j] >= enter_mid + u:
                    pos[j] = sig[j]
    return pos


@jit(nopython=True)
def calc_pos_e1(mid, sig, u):
    n = mid.shape[0]
    pos = np.zeros(n)
    enter_mid = 0
    for j in range(1, n):
        p = sig[j]
        last_pos = pos[j - 1]
        new_pos = last_pos
        dp = p - last_pos
        if sig[j - 1] != p:
            enter_mid = mid[j]
        if dp > 0:
            if enter_mid > 0:
                # print('1a')
                # if mid[j, i] != enter_mid:
                #     print('~~~~')
                if mid[j] <= enter_mid * (1 - u) or p > 0:
                    new_pos = p
        elif dp < 0:
            if enter_mid > 0:
                # print('2a')
                # if mid[j, i] != enter_mid:
                #     print('~~~~')
                if mid[j] >= enter_mid * (1 + u) or p < 0:
                    new_pos = p
        pos[j] = new_pos
    return pos


@jit(nopython=True)
def calc_pos_e1_m(mid, sig):
    pos = np.zeros(sig.shape)
    n = mid.shape[0]
    m = mid.shape[1]
    u = 0.0002
    for i in range(m):
        enter_mid = 0
        for j in range(1, n):
            p = sig[j, i]
            last_pos = pos[j - 1, i]
            new_pos = last_pos
            dp = p - last_pos
            if sig[j - 1, i] != p:
                enter_mid = mid[j, i]
            # else:
            #     if dp != 0:
            #         if enter_mid != 0:
            #             print('delta', j, mid[j, i] - enter_mid)
            #         else:
            #             print('skip', j)
            if dp > 0:
                if enter_mid > 0:
                    # print('1a')
                    # if mid[j, i] != enter_mid:
                    #     print('~~~~')
                    if mid[j, i] <= enter_mid * (1 - u) or p > 0:
                        new_pos = p
            elif dp < 0:
                if enter_mid > 0:
                    # print('2a')
                    # if mid[j, i] != enter_mid:
                    #     print('~~~~')
                    if mid[j, i] >= enter_mid * (1 + u) or p < 0:
                        new_pos = p
            pos[j, i] = new_pos
    return pos


@jit(nopython=True)
def calc_pos_e1_v(mid, sig):
    pos = np.zeros(sig.shape)
    n = mid.shape[0]
    u = 0.005
    enter_mid = 0
    for j in range(1, n):
        p = sig[j]
        last_pos = pos[j - 1]
        new_pos = last_pos
        dp = p - last_pos
        if sig[j - 1] != p:
            enter_mid = mid[j]
        else:
            if dp != 0:
                if enter_mid != 0:
                    print("delta", j, mid[j] - enter_mid)
                else:
                    print("skip", j)
        if dp > 0:
            if enter_mid > 0:
                print("1a")
                if mid[j] != enter_mid:
                    print("~~~~")
                if mid[j] <= enter_mid * (1 - u):
                    new_pos = p
        elif dp < 0:
            if enter_mid > 0:
                print("2a")
                if mid[j] != enter_mid:
                    print("~~~~")
                if mid[j] >= enter_mid * (1 + u):
                    new_pos = p
        pos[j] = new_pos
    return pos


@jit(nopython=True)
def apply_risk_m(sig, risk):
    sig2 = np.zeros(sig.shape)
    n = sig.shape[0]
    m = sig.shape[1]
    for i in range(m):
        for j in range(n):
            if risk[j] > 0:
                sig2[j, i] = max(0, sig[j, i])
            elif risk[j] < 0:
                sig2[j, i] = min(0, sig[j, i])
            else:
                sig2[j, i] = sig[j, i]
    return sig2


@jit(nopython=True)
def calc_max_ratio(a, b):
    n = a.shape[0]
    y = -1
    for j in range(n):
        if b[j] > 0:
            x = (a[j] + b[j]) / (2 * b[j])
            if x > y:
                y = x
    return y


@jit(nopython=True)
def calc_min_ratio(a, b):
    n = a.shape[0]
    y = 1
    for j in range(n):
        if b[j] > 0:
            x = (a[j] + b[j]) / (2 * b[j])
            if x < y:
                y = x
    return y

def get_n_items_q(a, q_list):
    r = []
    n = len(a)
    for q in q_list:
        m = -1
        for j in range(n):
            if a[j] < q:
                m = j
                break
        if m == -1:
            m = n
        r.append(m)
    return r

# @jit(nopython=True)
def calc_daily_oc_gaps(o, c) -> ArrayLike:
    n = len(o)
    gaps = np.zeros(n)
    for j in range(1, n):
        gaps[j] = o[j] - c[j - 1]
    return gaps

# @jit(nopython=True)
def calc_daily_oc_growth(o, c) -> ArrayLike:
    n = len(o)
    growth = np.zeros(n)
    for j in range(n):
        growth[j] = c[j] - o[j]
    return growth
