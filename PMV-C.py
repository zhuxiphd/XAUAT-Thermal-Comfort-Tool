# -*- coding: utf-8 -*-
"""
PMV 原始实现 + PMV-C(儿童)改动插入版
要求：
- 不改动 PMV 原始函数签名（pmv_core / pmv_with_components / pmv_ppd / pmv_ppd_with_components）
- 只新增：pmv_with_components_auto / pmv_ppd_auto 作为“带 age 的入口”
"""

import numpy as np
from numba import float64, vectorize
from typing import Union, Dict

MET_TO_W_M2 = 58.15
CHILD_ACTIVITY_COEFFICIENTS = {
    "seated_rest": (1.994, 0.0549),
    "reading_writing": (2.0245, 0.051),
    "standing_rest": (2.029, 0.0495),
    "walk_3kmh": (3.3829, 0.0313),
}


# =========================
# ===== 原始 PMV.py ========
# =========================
@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, float64)],
    cache=False,  # Avoid Numba cache issues on some Python/numba combos (e.g., "<dynamic>" rebuild errors).
)
def pmv_core(tdb, tr, vr, rh, met, clo, wme):
    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo

    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w

    if icl <= 0.078:
        f_cl = 1.0 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    hcf = 12.1 * np.sqrt(vr)
    hc = hcf

    taa = tdb + 273.0
    tra = tr + 273.0

    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)

    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015
    n = 0

    while np.abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * np.abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return pmv


def pmv_with_components(tdb, tr, vr, rh, met, clo, wme=0.0):
    tdb = float(tdb); tr = float(tr); vr = float(vr); rh = float(rh)
    met = float(met); clo = float(clo); wme = float(wme)

    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo

    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w

    if icl <= 0.078:
        f_cl = 1.0 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    hcf = 12.1 * np.sqrt(vr)
    hc = hcf

    taa = tdb + 273.0
    tra = tr + 273.0

    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)

    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015
    n = 0

    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)

    q_total = hl1 + hl2 + hl3 + hl4 + hl5 + hl6
    q_sens = hl4 + hl5 + hl6
    q_sensible = hl5 + hl6
    c_res = hl4
    q_lat = hl1 + hl2 + hl3
    e_skin = hl1 + hl2
    e_res = hl3
    e_rsw = hl2
    e_diff = hl1

    return {
        "ok": True,
        "pmv": float(pmv),
        "ppd": float(ppd),
        "hl1": float(hl1),
        "hl2": float(hl2),
        "hl3": float(hl3),
        "hl4": float(hl4),
        "hl5": float(hl5),
        "hl6": float(hl6),
        "mw": float(mw),
        "q_total": float(q_total),
        "q_sens": float(q_sens),
        "q_sensible": float(q_sensible),
        "c_res": float(c_res),
        "q_lat": float(q_lat),
        "e_skin": float(e_skin),
        "e_res": float(e_res),
        "e_rsw": float(e_rsw),
        "e_diff": float(e_diff),
    }


def pmv_ppd(tdb, tr, vr, rh, met, clo, wme=0.0):
    tdb = np.asarray(tdb, dtype=float)
    tr = np.asarray(tr, dtype=float)
    vr = np.asarray(vr, dtype=float)
    rh = np.asarray(rh, dtype=float)
    met = np.asarray(met, dtype=float)
    clo = np.asarray(clo, dtype=float)
    wme = np.asarray(wme, dtype=float)

    pmv = pmv_core(tdb, tr, vr, rh, met, clo, wme)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)

    if pmv.size == 1:
        return float(pmv), float(ppd)
    return pmv, ppd


def pmv_ppd_with_components(tdb, tr, vr, rh, met, clo, wme=0.0):
    res = pmv_with_components(tdb, tr, vr, rh, met, clo, wme)
    return (
        res["pmv"], res["ppd"],
        res["hl1"], res["hl2"], res["hl3"], res["hl4"], res["hl5"], res["hl6"]
    )


# ======================================
# ====== 以下为 PMV-C 插入新增 ==========
# （不改动原始函数，只新增“带 age 的入口”）
# ======================================
def _sanitize_children(tdb, tr, vr, rh, met, clo, wme):
    rh = np.clip(rh, 0.0, 100.0)
    vr = np.maximum(vr, 0.0)
    met = np.maximum(met, 0.0)
    clo = np.maximum(clo, 0.0)
    return float(tdb), float(tr), float(vr), float(rh), float(met), float(clo), float(wme)


def _is_child_mode(age: Union[float, int, None]) -> bool:
    if age is None:
        return False
    try:
        a = float(age)
    except Exception:
        return False
    if not np.isfinite(a):
        return False
    if abs(a - np.floor(a)) > 1e-12:
        return False
    ia = int(np.floor(a))
    return 8 <= ia <= 18


def _resolve_child_met(
    age: Union[float, int],
    child_activity: Union[str, None],
    fallback_met: Union[float, int, None],
) -> float:
    if child_activity is None:
        if fallback_met is None:
            raise ValueError("儿童 PMV 需要提供 child_activity 或 met。")
        return max(float(fallback_met), 0.0)
    key = str(child_activity).strip().lower()
    coeffs = CHILD_ACTIVITY_COEFFICIENTS.get(key)
    if coeffs is None:
        valid_keys = ", ".join(CHILD_ACTIVITY_COEFFICIENTS.keys())
        raise ValueError(f"无效的儿童活动类型：{child_activity}。可选值：{valid_keys}")
    intercept, slope = coeffs
    return max(float(intercept - slope * float(age)), 0.0)


def _thermal_sensation_level(pmv: float) -> int:
    if pmv <= -2.5:
        return -3
    elif pmv <= -1.5:
        return -2
    elif pmv <= -0.5:
        return -1
    elif pmv < 0.5:
        return 0
    elif pmv < 1.5:
        return 1
    elif pmv < 2.5:
        return 2
    else:
        return 3


def pmv_with_components_auto(
    tdb, tr, vr, rh, met, clo, wme=0.0,
    age: Union[float, int, None] = None,
    height: Union[float, None] = None,
    weight: Union[float, None] = None,
    child_activity: Union[str, None] = None,
) -> Dict[str, float]:
    """
    自动选择标准/儿童：
    - 非儿童：直接调用原始 pmv_with_components（标准）
    - 儿童：按 PMV-C 改动点计算
    """

    # ---------- 标准分支：不改动原始函数 ----------
    if not _is_child_mode(age):
        res = pmv_with_components(tdb, tr, vr, rh, met, clo, wme)
        res["mode"] = "standard"
        res["sensation_level"] = _thermal_sensation_level(float(res["pmv"]))
        return res

    # ---------- 儿童分支 ----------
    met = _resolve_child_met(age, child_activity, met)
    tdb, tr, vr, rh, met, clo, wme = _sanitize_children(tdb, tr, vr, rh, met, clo, wme)
    ia = int(float(age))

    if height is None or weight is None:
        raise ValueError("儿童 PMV 需要 height(米) 与 weight(kg)。")
    height = float(height)
    weight = float(weight)
    if height <= 0 or weight <= 0:
        raise ValueError("height/weight 必须为正数。")

    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo

    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w

    if icl <= 0.078:
        f_cl = 1.0 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    # PMV-C：hcf 替换
    f_eff = 0.84
    bsa = 0.202 * (weight**0.425) * (height**0.725)
    X = bsa / weight
    hcf = 42.71 * (vr**0.59) * (X**0.37)
    hc = hcf

    taa = tdb + 273.0
    tra = tr + 273.0

    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa

    # PMV-C：p5 替换
    T_skin = 36.45 - 0.0126 * mw - 0.126 * ia
    p5 = (T_skin + 273.0) + (p2 * (tra / 100.0) ** 4)

    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015
    n = 0

    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        # PMV-C：hcn 替换
        hcn = 4.87 * abs(100.0 * xf - taa) ** 0.25 * (X**0.27)
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * (xf ** 4)) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0

    # hl1 先按原始计算
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)

    # PMV-C：hl2 替换 + HL1 并入 HL2，输出 HL1=0
    hl2_raw = -18.9 + 0.26 * mw + 1.1 * ia
    hl2 = max(hl2_raw, hl1)
    hl1_out = 0.0

    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)

    # PMV-C：hl5 替换
    hl5 = 5.595 * f_eff * f_cl * (xn**4 - (tra / 100.0) ** 4)

    hl6 = f_cl * hc * (tcl - tdb)

    # PMV-C：ts 用 mw
    ts = 0.303 * np.exp(-0.036 * mw) + 0.028

    pmv = ts * (mw - hl1_out - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)

    q_total = hl1_out + hl2 + hl3 + hl4 + hl5 + hl6
    q_sens = hl4 + hl5 + hl6
    q_sensible = hl5 + hl6
    c_res = hl4
    q_lat = hl1_out + hl2 + hl3
    e_skin = hl1_out + hl2
    e_res = hl3
    e_rsw = hl2
    e_diff = hl1_out

    return {
        "ok": True,
        "pmv": float(pmv),
        "ppd": float(ppd),
        "hl1": float(hl1_out),
        "hl2": float(hl2),
        "hl3": float(hl3),
        "hl4": float(hl4),
        "hl5": float(hl5),
        "hl6": float(hl6),
        "mw": float(mw),
        "q_total": float(q_total),
        "q_sens": float(q_sens),
        "q_sensible": float(q_sensible),
        "c_res": float(c_res),
        "q_lat": float(q_lat),
        "e_skin": float(e_skin),
        "e_res": float(e_res),
        "e_rsw": float(e_rsw),
        "e_diff": float(e_diff),
        "mode": "children(6-18)",
        "sensation_level": int(_thermal_sensation_level(float(pmv))),
    }


def pmv_ppd_auto(
    tdb, tr, vr, rh, met, clo, wme=0.0,
    age: Union[float, int, None] = None,
    height: Union[float, None] = None,
    weight: Union[float, None] = None
):
    """自动选择标准/儿童（标准支持数组，儿童仅支持标量）"""
    if not _is_child_mode(age):
        return pmv_ppd(tdb, tr, vr, rh, met, clo, wme)
    res = pmv_with_components_auto(tdb, tr, vr, rh, met, clo, wme, age=age, height=height, weight=weight)
    return res["pmv"], res["ppd"]


if __name__ == "__main__":
    # ========= 原始标准 PMV 的验证方式：print =========
    # 标准算例（不传 age！因为原函数不支持 age）
    res_std = pmv_with_components(
        tdb=26, tr=26, vr=0.1, rh=50,
        met=1.0, clo=0.6, wme=0.0
    )
    print("\n=== 标准 PMV（原始函数 pmv_with_components）===")
    print("PMV:", res_std["pmv"], "PPD:", res_std["ppd"])
    print("hl1~hl6:", res_std["hl1"], res_std["hl2"], res_std["hl3"], res_std["hl4"], res_std["hl5"], res_std["hl6"])

    # ========= 新增入口：带 age 自动选择 =========
    res_std_auto = pmv_with_components_auto(
        tdb=26, tr=26, vr=0.1, rh=50,
        met=1.0, clo=0.6, wme=0.0,
        age=35
    )
    print("\n=== 自动入口（age=35 -> standard）===")
    print("mode:", res_std_auto["mode"], "PMV:", res_std_auto["pmv"], "PPD:", res_std_auto["ppd"])

    res_child_auto = pmv_with_components_auto(
        tdb=26, tr=26, vr=0.1, rh=50,
        met=1.0, clo=0.6, wme=0.0,
        age=10, height=1.3, weight=30
    )
    print("\n=== 自动入口（age=10 -> children）===")
    print("mode:", res_child_auto["mode"], "PMV:", res_child_auto["pmv"], "PPD:", res_child_auto["ppd"])
    print("hl1(应为0):", res_child_auto["hl1"], "hl2:", res_child_auto["hl2"])
