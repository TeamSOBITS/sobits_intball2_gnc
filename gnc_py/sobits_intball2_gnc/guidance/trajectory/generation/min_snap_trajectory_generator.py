"""
Minimum snap trajectory core (ROS-independent, numpy only).

参考: Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control
for Quadrotors", ICRA 2011.

1区間の軌道を，無次元時間 τ = t/T (0〜1) の (2k-1) 次多項式で表す．
    p(τ) = Σ_i c_i τ^i   (i = 0 .. 2k-1，c_i は x, y, z の3成分を持つ)
両端で 0〜k-1 階微分（位置・速度・加速度・jerk ...）を指定して
A c = b を解き，係数 c を求める．
k = 4 なら minimum snap（7次多項式），k = 3 なら minimum jerk（5次多項式）．

第2段階（多区間）: solve_min_snap が論文 (10)(11) の QP を解く．
    経由点では位置だけを決め，1〜k 階微分は「隣の区間とつながる」だけ（free）．
    コスト H と制約 A c = b から KKT 条件の連立方程式を立てて解く．
    引数の形は sobits の generate(waypoints, segment_times, v0) にそろえてある．

sobits_intball2_gnc への組み込み: 末尾の MinSnapTrajectoryGenerator が BaseTrajectoryGenerator の
契約を満たす（import は math と numpy だけで，ROS は不要）．移動モードとしては
guidance/trajectory/min_snap_trajectory.py の MinSnapTrajectory が plan_segment_times と
solve_min_snap を呼び，trajectory_tracking_mode=static_min_snap（motion_profile: min_snap）で使われる．
"""

import math

import numpy as np


def deriv_basis(n_coef, j, tau):
    """
    Return d^j/dτ^j of [1, τ, τ², ...] for each τ (shape: (len(τ), n_coef)).

    τ^i を j 回微分すると
        d^j/dτ^j τ^i = i!/(i-j)! · τ^(i-j)   (i >= j)
                     = 0                     (i <  j)
    この行と係数 c の積が j 階微分になる: p^(j)(τ) = basis · c
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    basis = np.zeros((tau.size, n_coef))
    for i in range(j, n_coef):
        basis[:, i] = math.factorial(i) / math.factorial(i - j) * tau ** (i - j)
    return basis


class PolySegment:
    """1区間の多項式軌道. coeffs は τ に関する係数 (shape: (n_coef, dim))."""

    def __init__(self, coeffs, duration):
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.duration = float(duration)

    def evaluate(self, t, order=0):
        """
        Return the order-th time derivative at time t (0: p, 1: v, 2: a).

        連鎖律: τ = t/T より dτ/dt = 1/T なので
            d^j p/dt^j = (1/T^j) · d^j p/dτ^j
        t は 0〜T の外なら端に丸める（開始前・終了後は端で静止）．
        """
        T = self.duration
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, T)
        tau = t_arr.ravel() / T
        basis = deriv_basis(self.coeffs.shape[0], order, tau)
        out = basis @ self.coeffs / T ** order
        return out[0] if t_arr.ndim == 0 else out


def solve_segment(start, end, duration, k=4):
    """
    Solve A c = b for one segment with boundary derivatives at both ends.

    start, end: shape (k, dim). 実時間 t での 0〜k-1 階微分
        start[0] = 位置, start[1] = 速度, start[2] = 加速度, start[3] = jerk ...
    k 階微分の2乗積分を最小にする解は (2k-1) 次多項式（係数 2k 個）．
    """
    start = np.asarray(start, dtype=float).reshape(k, -1)
    end = np.asarray(end, dtype=float).reshape(k, -1)
    T = float(duration)
    n_coef = 2 * k
    A = np.zeros((n_coef, n_coef))
    b = np.zeros((n_coef, start.shape[1]))
    for j in range(k):
        # 実時間の微分を τ の微分に直す: d^j p/dτ^j = T^j · d^j p/dt^j
        # τ = 0 の行: 列 j だけ j!（ほかは 0）
        A[j] = deriv_basis(n_coef, j, 0.0)[0]
        b[j] = start[j] * T ** j
        # τ = 1 の行: 列 i (i >= j) に i!/(i-j)!
        A[k + j] = deriv_basis(n_coef, j, 1.0)[0]
        b[k + j] = end[j] * T ** j
    coeffs = np.linalg.solve(A, b)
    return PolySegment(coeffs, T)


def line_trajectory(p0, p1, duration, k=4):
    """Rest-to-rest segment from p0 to p1 (両端で速度・加速度・... = 0)."""
    p0 = np.atleast_1d(np.asarray(p0, dtype=float))
    p1 = np.atleast_1d(np.asarray(p1, dtype=float))
    start = np.zeros((k, p0.size))
    end = np.zeros((k, p0.size))
    start[0] = p0
    end[0] = p1
    return solve_segment(start, end, duration, k)


def min_duration(p0, p1, v_max, a_max, k=4, n_samples=2001):
    """
    Shortest T for a rest-to-rest line satisfying |v| <= v_max and |a| <= a_max.

    直線では p = p0 + d·s(τ)（s: 0→1）なので
        |v| = |d|·|s'(τ)|/T   → T >= |d|·max|s'| / v_max
        |a| = |d|·|s''(τ)|/T² → T >= sqrt(|d|·max|s''| / a_max)
    """
    dist = np.linalg.norm(np.asarray(p1, float) - np.asarray(p0, float))
    unit = line_trajectory(0.0, 1.0, 1.0, k)   # s(τ)（T = 1 なので t = τ）
    tau = np.linspace(0.0, 1.0, n_samples)
    s1_max = np.max(np.abs(unit.evaluate(tau, 1)))
    s2_max = np.max(np.abs(unit.evaluate(tau, 2)))
    return max(dist * s1_max / v_max, math.sqrt(dist * s2_max / a_max))


# ======================================================================
# 第2段階: 多区間 minimum snap（論文 V 節 (9)(10)(11)）
# ======================================================================

class PiecewiseTrajectory:
    """区間をつないだ軌道（論文 (9) の区分多項式）. segments は PolySegment の列."""

    def __init__(self, segments, cost=None):
        self.segments = list(segments)
        self.durations = np.array([s.duration for s in self.segments])
        # 各区間の開始時刻 t_0, t_1, ..., t_{m-1}
        self.start_times = np.concatenate([[0.0], np.cumsum(self.durations)[:-1]])
        self.duration = float(self.durations.sum())
        self.cost = cost   # ∫‖d^k p/dt^k‖² dt（論文 (14) の f(T) に使う）

    def evaluate(self, t, order=0):
        """
        Return the order-th time derivative at global time t (0: p, 1: v, 2: a).

        t がどの区間に入るかを探し，その区間の PolySegment に区間内の時刻を渡す．
        0〜全体の時間の外は端に丸める．
        """
        t_arr = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        flat = t_arr.ravel()
        idx = np.searchsorted(self.start_times, flat, side='right') - 1
        idx = np.clip(idx, 0, len(self.segments) - 1)
        dim = self.segments[0].coeffs.shape[1]
        out = np.empty((flat.size, dim))
        for i, seg in enumerate(self.segments):
            mask = idx == i
            if mask.any():
                out[mask] = seg.evaluate(flat[mask] - self.start_times[i], order)
        return out[0] if t_arr.ndim == 0 else out

    def coeffs_in_seconds(self):
        """
        Return coefficients in local real time [s], shape (n_segments, dim, 2k).

        sobits の契約は「区間内の実時間 t (0〜T_i) の多項式」．
        τ = t/T なので c_i τ^i = (c_i / T^i) t^i → 係数を T^i で割る．
        """
        out = []
        for seg in self.segments:
            powers = seg.duration ** np.arange(seg.coeffs.shape[0])
            out.append((seg.coeffs / powers[:, None]).T)
        return np.array(out)


def _cost_matrix_unit(n_coef, k):
    """
    Return H with ∫_0^1 (d^k p/dτ^k)² dτ = cᵀ H c（論文 (11) の H，1区間・τ 版）.

    d^k τ^i/dτ^k = i!/(i-k)! τ^(i-k) なので
        H[i, j] = i!/(i-k)! · j!/(j-k)! · ∫_0^1 τ^(i+j-2k) dτ
                = i!/(i-k)! · j!/(j-k)! / (i+j-2k+1)   (i, j >= k)
    """
    H = np.zeros((n_coef, n_coef))
    for i in range(k, n_coef):
        for j in range(k, n_coef):
            fi = math.factorial(i) / math.factorial(i - k)
            fj = math.factorial(j) / math.factorial(j - k)
            H[i, j] = fi * fj / (i + j - 2 * k + 1)
    return H


def solve_min_snap(waypoints, segment_times, k=4, v0=None):
    """
    Solve the multi-segment minimum-derivative QP (論文 (10)(11)).

    waypoints: shape (n, dim)，n >= 2．segment_times: shape (n-1,)，すべて > 0 [s]．
    v0: 出発速度 shape (dim,)．None なら静止から出発．
    制約（論文 (10)）:
        両端      : 位置を指定，1〜k-1 階微分 = 0（出発の速度だけ v0 にできる），k 階は free
        途中の経由点: 位置を指定，1〜k 階微分が隣の区間とつながる（値は free）
    コスト: Σ_i ∫‖d^k p/dt^k‖² dt．x, y, z は独立（論文 V-A）なので同じ A, H で一度に解く．
    """
    waypoints = np.asarray(waypoints, dtype=float)
    segment_times = np.asarray(segment_times, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[0] < 2:
        raise ValueError('waypoints must have shape (n >= 2, dim)')
    n_seg = waypoints.shape[0] - 1
    dim = waypoints.shape[1]
    if segment_times.shape != (n_seg,):
        raise ValueError('segment_times must have shape (n_waypoints - 1,)')
    if np.any(segment_times <= 0.0):
        raise ValueError('segment_times must all be > 0')
    if v0 is not None:
        v0 = np.asarray(v0, dtype=float)
        if v0.shape != (dim,):
            raise ValueError('v0 must have shape (dim,)')

    n_coef = 2 * k               # 1区間の係数の数（(2k-1) 次多項式）
    n_var = n_coef * n_seg       # 全区間の係数の数
    T = segment_times

    def col(i):
        """区間 i の係数が並ぶ列の範囲."""
        return slice(i * n_coef, (i + 1) * n_coef)

    # --- コスト H（区間ごとのブロック対角）---
    # 実時間のコスト: ∫_0^T (d^k p/dt^k)² dt = (1/T^(2k-1)) ∫_0^1 (d^k p/dτ^k)² dτ
    # （論文 (13) の α^(2k-1) と同じ時間スケーリング）
    H_unit = _cost_matrix_unit(n_coef, k)
    H = np.zeros((n_var, n_var))
    for i in range(n_seg):
        H[col(i), col(i)] = H_unit / T[i] ** (2 * k - 1)

    # --- 制約 A c = b ---
    A_rows = []
    b_rows = []

    def add(row, value):
        A_rows.append(row)
        b_rows.append(value)

    # 出発（区間 0 の τ = 0）: d^j p/dτ^j = T^j · d^j p/dt^j
    for j in range(k):
        row = np.zeros(n_var)
        row[col(0)] = deriv_basis(n_coef, j, 0.0)[0]
        value = np.zeros(dim)
        if j == 0:
            value = waypoints[0]
        elif j == 1 and v0 is not None:
            value = v0 * T[0]
        add(row, value)

    # 到着（最後の区間の τ = 1）: 位置を指定，1〜k-1 階微分 = 0
    for j in range(k):
        row = np.zeros(n_var)
        row[col(n_seg - 1)] = deriv_basis(n_coef, j, 1.0)[0]
        add(row, waypoints[-1] if j == 0 else np.zeros(dim))

    # 途中の経由点 i（区間 i-1 の終わりと区間 i の始まり）
    for i in range(1, n_seg):
        # 位置: 両側の区間が経由点を通る
        row = np.zeros(n_var)
        row[col(i - 1)] = deriv_basis(n_coef, 0, 1.0)[0]
        add(row, waypoints[i])
        row = np.zeros(n_var)
        row[col(i)] = deriv_basis(n_coef, 0, 0.0)[0]
        add(row, waypoints[i])
        # 1〜k 階微分の連続性（実時間で一致）:
        #   (1/T_{i-1}^j) d^j p_{i-1}/dτ^j(1) = (1/T_i^j) d^j p_i/dτ^j(0)
        # 両辺に T_{i-1}^j を掛けて数値の大きさをそろえる
        for j in range(1, k + 1):
            row = np.zeros(n_var)
            row[col(i - 1)] = deriv_basis(n_coef, j, 1.0)[0]
            row[col(i)] = -(T[i - 1] / T[i]) ** j * deriv_basis(n_coef, j, 0.0)[0]
            add(row, np.zeros(dim))

    A = np.array(A_rows)
    b = np.array(b_rows)

    # --- KKT 条件: min cᵀHc s.t. A c = b ---
    #   ∂/∂c [cᵀHc + λᵀ(A c - b)] = 2Hc + Aᵀλ = 0,  A c = b
    #   → [2H  Aᵀ] [c]   [0]
    #     [A   0 ] [λ] = [b]
    n_con = A.shape[0]
    kkt = np.block([[2.0 * H, A.T], [A, np.zeros((n_con, n_con))]])
    rhs = np.vstack([np.zeros((n_var, dim)), b])
    coeffs = np.linalg.solve(kkt, rhs)[:n_var]          # shape (n_var, dim)

    cost = float(np.sum(coeffs * (H @ coeffs)))          # Σ_軸 cᵀHc
    segments = [PolySegment(coeffs[col(i)], T[i]) for i in range(n_seg)]
    return PiecewiseTrajectory(segments, cost)


def segment_times_by_distance(waypoints, avg_speed):
    """
    Return initial segment times proportional to segment length (T_i = |Δp_i| / v).

    論文 V-C の初期値と同じ「経由点の間を直線・一定速度で進む」とした時間．
    (14) の時間配分の最適化は，この値から始める予定．
    """
    waypoints = np.asarray(waypoints, dtype=float)
    dist = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    if np.any(dist <= 0.0):
        raise ValueError('consecutive waypoints must be different')
    return dist / float(avg_speed)


def optimize_segment_times(waypoints, segment_times, k=4, v0=None,
                           max_iter=50, tol=1e-6):
    """
    Optimize the ratio of segment times with the total fixed (論文 V-C (14)).

        min f(T)  s.t. Σ T_i = t_m,  T_i >= 0
    f(T) は solve_min_snap のコスト（論文 (10) の最小値）．
    手順（論文と同じ）:
      1. 方向 g_i（i 番目が 1，ほかが -1/(m-1)，Σ g_i = 0）で数値微分
             ∇_{g_i} f ≈ (f(T + h g_i) - f(T)) / h
         ※ 論文は -1/(m-2) と書いているが，区間が m 個のとき Σ g_i = 0 になるのは -1/(m-1)
      2. 下り方向 d = -Σ_i (∇_{g_i} f) g_i（Σ d = 0 なので合計時間は変わらない）
      3. バックトラッキング直線探索（歩幅を半分ずつ小さくする）で T を更新
    Returns:
        (segment_times, cost_history)．cost_history[0] は最初の配分のコスト
    """
    T = np.asarray(segment_times, dtype=float).copy()
    m = T.size

    def f(times):
        return solve_min_snap(waypoints, times, k, v0).cost

    cost = f(T)
    history = [cost]
    if m < 2:
        return T, history   # 区間が1つなら比を変える余地がない

    # 方向 g_i を行に並べた行列 G（G[i] = g_i）
    G = np.full((m, m), -1.0 / (m - 1))
    np.fill_diagonal(G, 1.0)

    for _ in range(max_iter):
        # 1. 方向微分（h は区間時間に対して小さい値）
        h = 1e-6 * T.min()
        dir_deriv = np.array([(f(T + h * G[i]) - cost) / h for i in range(m)])

        # 2. 下り方向（∇f·d = -Σ(∇_{g_i} f)² <= 0 なので必ず下り）
        d = -dir_deriv @ G
        slope = -np.sum(dir_deriv ** 2)
        if slope == 0.0:
            break

        # 3. バックトラッキング: T_i > 0 を保てる歩幅から始め，コストが十分下がるまで半分に
        shrinking = d < 0.0
        step = 0.5 * np.min(T[shrinking] / -d[shrinking]) if shrinking.any() else 1.0
        while step > 1e-12:
            T_new = T + step * d
            cost_new = f(T_new)
            if cost_new <= cost + 1e-4 * step * slope:   # Armijo 条件
                break
            step *= 0.5
        else:
            break   # どの歩幅でも下がらない → 収束

        improvement = (cost - cost_new) / cost
        T, cost = T_new, cost_new
        history.append(cost)
        if improvement < tol:
            break
    return T, history


def _max_vel_acc(traj, n_samples=200):
    """Return (max |v|, max |a|) sampled densely on every segment."""
    v_max = a_max = 0.0
    for t0, seg in zip(traj.start_times, traj.segments):
        t = np.linspace(0.0, seg.duration, n_samples)
        v_max = max(v_max, np.linalg.norm(seg.evaluate(t, 1), axis=1).max())
        a_max = max(a_max, np.linalg.norm(seg.evaluate(t, 2), axis=1).max())
    return v_max, a_max


def scale_segment_times(waypoints, segment_times, v_max, a_max, k=4, v0=None,
                        max_iter=10):
    """
    Scale all segment times by α so that max|v| <= v_max and max|a| <= a_max.

    論文 V-A-1) Temporal scaling: 時間を α 倍すると，形は同じまま
        v → v/α，a → a/α²
    なので，上限ちょうどになる最小の α は
        α = max( max|v| / v_max ,  sqrt(max|a| / a_max) )
    （α < 1 なら速くする方向に縮める）．比は変えないので (14) の最適な比が保たれる．
    v0 を与えたときは，出発速度だけは α 倍しても変わらないので，解き直して数回くり返す．
    """
    T = np.asarray(segment_times, dtype=float).copy()
    for _ in range(max_iter):
        traj = solve_min_snap(waypoints, T, k, v0)
        v_peak, a_peak = _max_vel_acc(traj)
        alpha = max(v_peak / v_max, math.sqrt(a_peak / a_max))
        T = T * alpha
        if abs(alpha - 1.0) < 1e-6 or v0 is None:
            break
    return T


def plan_segment_times(waypoints, v_max, a_max, k=4, v0=None, optimize=True):
    """
    Return segment times: 長さに比例 → (14) で比を最適化 → α で上限に合わせる.

    sobits の generate(waypoints, segment_times, v0) に渡す segment_times を作る．
    """
    T = segment_times_by_distance(waypoints, v_max)
    if optimize:
        T, _ = optimize_segment_times(waypoints, T, k, v0)
    return scale_segment_times(waypoints, T, v_max, a_max, k, v0)


# ======================================================================
# sobits_intball2_gnc との接続
# （BaseTrajectoryGenerator の契約を満たす薄いアダプタ）
# ======================================================================

N_CONTRACT_COEFFS = 8   # sobits の契約: 1区間あたり8個（7次多項式）の係数


class MinSnapTrajectoryGenerator:
    """
    Minimum-snap generator satisfying sobits' BaseTrajectoryGenerator contract.

    generate(waypoints, segment_times, v0) → coeffs, shape (n_segments, 3, 8)．
    係数は昇べき順で，区間内の実時間 t [s]（0〜T_i）の多項式（sobits の契約）．
    k < 4 のときは使わない高次の係数を 0 で埋める．
    """

    def __init__(self, k=4):
        if not 2 <= k <= N_CONTRACT_COEFFS // 2:
            raise ValueError('k must be 2, 3 or 4 (contract has 8 coefficients)')
        self.k = k

    def generate(self, waypoints, segment_times, v0=None):
        """Return coeffs (n_segments, 3, 8) through waypoints (sobits contract)."""
        waypoints = np.asarray(waypoints, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError('waypoints must have shape (n_waypoints, 3)')
        if v0 is not None and np.asarray(v0).shape != (3,):
            raise ValueError('v0 must have shape (3,)')
        # 残りの入力チェック（2点以上，区間時間の形と正の値）は solve_min_snap が行う
        traj = solve_min_snap(waypoints, segment_times, self.k, v0)
        coeffs = traj.coeffs_in_seconds()                 # (n_seg, 3, 2k)
        out = np.zeros((coeffs.shape[0], 3, N_CONTRACT_COEFFS))
        out[:, :, :coeffs.shape[2]] = coeffs
        return out


def _main():
    import matplotlib.pyplot as plt

    # --- 1. 係数の確認: 0 → 1 の s(τ) を k = 2, 3, 4 で解く ---
    for k in (2, 3, 4):
        c = line_trajectory(0.0, 1.0, 1.0, k).coeffs[:, 0]
        print(f'k={k}: c =', np.round(c, 6))
    print('  (k=4 の期待値: c4..c7 = 35, -84, 70, -20)')

    # --- 2. 3次元の直線軌道 ---
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 0.5, 0.3])
    T = 10.0
    seg = line_trajectory(p0, p1, T, k=4)

    t = np.linspace(0.0, T, 1001)
    p = seg.evaluate(t, 0)
    v = seg.evaluate(t, 1)
    a = seg.evaluate(t, 2)

    print('\n両端の値 (t=0, t=T):')
    for name, order in (('p', 0), ('v', 1), ('a', 2)):
        print(f'  {name}(0) = {np.round(seg.evaluate(0.0, order), 9)}, '
              f'{name}(T) = {np.round(seg.evaluate(T, order), 9)}')

    dist = np.linalg.norm(p1 - p0)
    v_norm = np.linalg.norm(v, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    print(f'\n|v| の最大: {v_norm.max():.6f} m/s '
          f'(式 35/16·|d|/T = {35 / 16 * dist / T:.6f}, t = {t[v_norm.argmax()]:.2f} s)')
    print(f'|a| の最大: {a_norm.max():.6f} m/s^2 '
          f'(式 7.51·|d|/T² ≈ {7.5131 * dist / T ** 2:.6f}, t = {t[a_norm.argmax()]:.2f} s)')

    # --- 3. 位置・速度・加速度の3段グラフ ---
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
    for data, ax, label in zip((p, v, a), axes, ('p [m]', 'v [m/s]', 'a [m/s²]')):
        for i, axis_name in enumerate('xyz'):
            ax.plot(t, data[:, i], label=axis_name)
        ax.set_ylabel(label)
        ax.grid(True)
    axes[0].legend()
    axes[0].set_title(f'minimum snap (k=4): p0={p0.tolist()} → p1={p1.tolist()}, T={T} s')
    axes[-1].set_xlabel('t [s]')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _main()
