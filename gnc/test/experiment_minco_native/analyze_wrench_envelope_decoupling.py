"""wrench envelope（config/wrench_envelope.csv、1901面）を force/torque独立の
箱/球で保守的に近似した場合、実際の可動域をどれだけ失うかを見積もる。

MINCO側のenvelopeチェック（F_ENV @ wrench <= G_ENV、1901面の内積）を
「force用のr_F」「torque用のr_T」のような単純な独立制約に置き換えられれば
1点あたりO(1901)がO(1)になり大幅に速くなるはずだが、そのために失う
可動域がどれくらいかを事前に見積もる（実装前のFeasibilityチェック）。

結論（2026-09-01時点の計測）: 単純な独立分離は可動域を大きく失いすぎる
（球モデルはforce/torqueの同時使用可能量がpure-axis限界の1割程度まで
落ちる、箱モデルでも全軸一律で28%程度しか使えない）ため、この案は
見送りが妥当と判断した。詳細はdocs参照。
"""

import numpy as np

PATH = "../../../minco_native_py/config/wrench_envelope.csv"


def load_envelope(path):
    with open(path) as f:
        rows, cols = map(int, f.readline().split())
        data = np.loadtxt(f)
    assert data.shape == (rows, cols + 1)
    return data[:, :6], data[:, 6]


def per_axis_limits(F, G):
    limits = {}
    for name, idx in [("Fx", 0), ("Fy", 1), ("Fz", 2), ("Tx", 3), ("Ty", 4), ("Tz", 5)]:
        e = np.zeros(6)
        e[idx] = 1.0
        coef = F @ e
        pos = coef > 1e-9
        tmax = (G[pos] / coef[pos]).min() if pos.any() else np.inf
        neg = coef < -1e-9
        tmin = (G[neg] / coef[neg]).max() if neg.any() else -np.inf
        limits[name] = (tmax, -tmin if np.isfinite(tmin) else tmin)
    return limits


def ball_pareto_frontier(F, G, axis_f_ref, axis_t_ref, fracs):
    aF = np.linalg.norm(F[:, :3], axis=1)
    aT = np.linalg.norm(F[:, 3:], axis=1)
    rows = []
    for frac in fracs:
        rF = frac * axis_f_ref
        slack = G - rF * aF
        if np.any(slack < 0):
            rows.append((rF, None))
            continue
        pos = aT > 1e-12
        rT = (slack[pos] / aT[pos]).min()
        rows.append((rF, rT))
    return rows


def box_uniform_shrink(F, G, box_ref):
    worst = (np.abs(F) @ box_ref) / G
    k = 1.0 / worst.max()
    return k, box_ref * k


if __name__ == "__main__":
    F, G = load_envelope(PATH)
    limits = per_axis_limits(F, G)
    print("per-axis independent limits (N or N*m):")
    for name, (pos, neg) in limits.items():
        print(f"  {name}: +{pos:.6f} / {neg:.6f}")

    axis_f_ref = min(limits["Fx"][0], limits["Fy"][0], limits["Fz"][0])
    axis_t_ref = min(limits["Tx"][0], limits["Ty"][0], limits["Tz"][0])
    print(f"\nball pareto frontier (axis_f_ref={axis_f_ref:.5f}, axis_t_ref={axis_t_ref:.6f}):")
    for rF, rT in ball_pareto_frontier(F, G, axis_f_ref, axis_t_ref, np.linspace(0, 1.0, 11)):
        if rT is None:
            print(f"  r_F={rF:.5f}: infeasible")
        else:
            print(f"  r_F={rF:.5f} ({rF/axis_f_ref:.0%} of axis) -> "
                  f"r_T={rT:.6f} ({rT/axis_t_ref:.0%} of axis)")

    box_ref = np.array([limits["Fx"][0], limits["Fy"][0], limits["Fz"][0],
                         limits["Tx"][0], limits["Ty"][0], limits["Tz"][0]])
    k, box = box_uniform_shrink(F, G, box_ref)
    print(f"\nbox uniform shrink k={k:.3f} (i.e. only {k:.0%} of per-axis independent "
          f"limits usable simultaneously in the worst-case corner)")
    print("  shrunk box:", box)
