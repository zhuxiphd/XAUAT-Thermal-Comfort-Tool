"""
Standalone PMV / PPD implementation (Fanger / ISO 7730 style)

完全独立版本：
- 不依赖 pythermalcomfort
- 只依赖 numpy 和 numba（加速），如果你不想用 numba，可以去掉 @vectorize 装饰器。

函数：
- pmv_core(...)              : 计算 PMV（支持标量或数组，依赖 numba.vectorize）
- pmv_ppd(...)               : 计算 PMV 和 PPD，返回 (pmv, ppd)
- pmv_with_components(...)   : 标量版，返回 PMV、PPD 和 Fanger 的 6 个散热项 hl1~hl6
- pmv_ppd_with_components(...) : 标量封装，直接返回 (pmv, ppd, hl1, ..., hl6)

单位约定：
- tdb : 干球温度 [°C]
- tr  : 平均辐射温度 [°C]
- vr  : 相对风速 [m/s]
- rh  : 相对湿度 [%]（0–100）
- met : 代谢率 [met]
- clo : 服装热阻 [clo]
- wme : 外部功 [met]
"""

import numpy as np
from numba import float64, vectorize

# 1 met = 58.15 W/m²（人体表面积上的功率密度）
MET_TO_W_M2 = 58.15


@vectorize(
    [
        float64(
            float64,  # tdb
            float64,  # tr
            float64,  # vr
            float64,  # rh
            float64,  # met
            float64,  # clo
            float64,  # wme
        )
    ],
    cache=True,
)
def pmv_core(tdb, tr, vr, rh, met, clo, wme):
    """
    核心 PMV 计算函数（向量化 ufunc）
    输入可以是标量，也可以是 numpy 数组，会自动广播。

    实现等价于 _pmv_ppd_optimized，只是：
    - 常量 met_to_w_m2 换成了模块内的 MET_TO_W_M2
    - 函数名改为 pmv_core
    """
    # 水汽分压 [Pa] 的近似公式
    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))

    # 服装热阻 [m²·K/W]
    icl = 0.155 * clo

    # 代谢率、外部功 [W/m²]
    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w  # 体内产热

    # 服装面积系数 f_cl
    if icl <= 0.078:
        f_cl = 1.0 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    # 强迫对流换热系数
    hcf = 12.1 * np.sqrt(vr)
    hc = hcf  # 初始化

    taa = tdb + 273.0
    tra = tr + 273.0

    # 初始衣服表面温度估计
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
    # 迭代求衣服表面温度 tcl
    while np.abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * np.abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            # 正常情况下不会触发，只是防止死循环
            raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0

    # 各种散热项（Fanger 模型）
    # 1) 皮肤水分扩散（潜热）
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    # 2) 出汗散热（代谢率较高时，潜热）
    hl2 = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0
    # 3) 呼吸潜热
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    # 4) 呼吸显热
    hl4 = 0.0014 * m * (34.0 - tdb)
    # 5) 辐射显热
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    # 6) 对流显热
    hl6 = f_cl * hc * (tcl - tdb)

    # 热感觉传递系数
    ts = 0.303 * np.exp(-0.036 * m) + 0.028

    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return pmv


def pmv_with_components(tdb, tr, vr, rh, met, clo, wme=0.0):
    """
    标量版 PMV 计算，把 Fanger 模型里的 6 个散热分量一起返回。
    返回:
        {
          "ok": True,
          "pmv": ...,
          "ppd": ...,
          "hl1": ...,
          "hl2": ...,
          "hl3": ...,
          "hl4": ...,
          "hl5": ...,
          "hl6": ...,
          "mw": ...,
          "q_total": ...,
          "q_sens": ...,
          "q_sensible": ...,
          "c_res": ...,
          "q_lat": ...,
          "e_skin": ...,
          "e_res": ...,
          "e_rsw": ...,
          "e_diff": ...
        }
    """
    # 转成标量 float，避免奇怪的 numpy 类型
    tdb = float(tdb)
    tr = float(tr)
    vr = float(vr)
    rh = float(rh)
    met = float(met)
    clo = float(clo)
    wme = float(wme)

    # 水汽分压 [Pa]
    pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))

    # 服装热阻
    icl = 0.155 * clo

    # 代谢率、外部功 [W/m²]
    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w

    # 服装面积系数 f_cl
    if icl <= 0.078:
        f_cl = 1.0 + 1.29 * icl
    else:
        f_cl = 1.05 + 0.645 * icl

    # 强迫对流换热系数
    hcf = 12.1 * np.sqrt(vr)
    hc = hcf

    taa = tdb + 273.0
    tra = tr + 273.0

    # 初始衣服表面温度估计
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
    # 迭代求 tcl
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0

    # 6 个散热项
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    hl2 = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    # PPD 按原公式算
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)

    # 组合输出（全部为 W/m²）
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
    """
    计算 PMV 和 PPD 的简单封装。

    参数可以是标量或者 array-like：
    - 如果传标量，返回标量 pmv, ppd
    - 如果传数组，返回 numpy.ndarray（可广播）
    """
    # 转成 numpy 数组（支持 Python list / 标量混合）
    tdb = np.asarray(tdb, dtype=float)
    tr = np.asarray(tr, dtype=float)
    vr = np.asarray(vr, dtype=float)
    rh = np.asarray(rh, dtype=float)
    met = np.asarray(met, dtype=float)
    clo = np.asarray(clo, dtype=float)
    wme = np.asarray(wme, dtype=float)

    # 调用核心 PMV 计算（向量化）
    pmv = pmv_core(tdb, tr, vr, rh, met, clo, wme)

    # 按 ISO 7730 / Fanger 的公式计算 PPD
    ppd = 100.0 - 95.0 * np.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)

    # 如果输入都是标量，那就把输出也转成标量，方便打印 / 下游逻辑
    if pmv.size == 1:
        return float(pmv), float(ppd)
    return pmv, ppd


def pmv_ppd_with_components(tdb, tr, vr, rh, met, clo, wme=0.0):
    """
    标量版封装：
    直接返回 PMV、PPD 和 6 个散热项 hl1~hl6。

    返回顺序：
        pmv, ppd, hl1, hl2, hl3, hl4, hl5, hl6
    """
    res = pmv_with_components(tdb, tr, vr, rh, met, clo, wme)
    return (
        res["pmv"],
        res["ppd"],
        res["hl1"],
        res["hl2"],
        res["hl3"],
        res["hl4"],
        res["hl5"],
        res["hl6"],
    )


if __name__ == "__main__":
    # 单点测试参数（你可以按需改成自己工况）
    tdb = 30.0
    tr = 40.0
    vr = 0.25
    rh = 60.0
    met = 1.72
    clo = 0.6
    wme = 0.0

    # 1) 原来只要 PMV、PPD：
    pmv_value, ppd_value = pmv_ppd(tdb, tr, vr, rh, met, clo, wme)
    print("=== Simple PMV/PPD ===")
    print("PMV:", pmv_value)
    print("PPD:", ppd_value)

    # 2) 新：一次性拿到 6 个散热项
    pmv_c, ppd_c, hl1, hl2, hl3, hl4, hl5, hl6 = pmv_ppd_with_components(
        tdb, tr, vr, rh, met, clo, wme
    )
    print("\n=== Components (Fanger 6 heat loss terms) ===")
    print("PMV:", pmv_c, "PPD:", ppd_c)
    print("hl1 (皮肤水分扩散):", hl1)
    print("hl2 (出汗显著蒸发):", hl2)
    print("hl3 (呼吸潜热):    ", hl3)
    print("hl4 (呼吸显热):    ", hl4)
    print("hl5 (辐射):        ", hl5)
    print("hl6 (对流):        ", hl6)

    # 3) 也保留原来的数组输入示例：
    tdb_array = [22.0, 25.0, 28.0]
    pmv_arr, ppd_arr = pmv_ppd(tdb_array, tr, vr, rh, met, clo, wme)
    print("\n=== Array input demo ===")
    print("PMV array:", pmv_arr)
    print("PPD array:", ppd_arr)
