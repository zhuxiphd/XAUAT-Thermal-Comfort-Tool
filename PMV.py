"""
Standalone PMV / PPD implementation (Fanger / ISO 7730 style)

完全独立版本：
- 不依赖 pythermalcomfort
- 只依赖 numpy 和 numba（加速），如果你不想用 numba，可以去掉 @vectorize 装饰器。

函数：
- pmv_core(...)  : 计算 PMV（支持标量或数组，依赖 numba.vectorize）
- pmv_ppd(...)   : 计算 PMV 和 PPD，返回 (pmv, ppd)

单位约定（和你贴的源码保持一致）：
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

    这里的实现等价于你贴出来的 _pmv_ppd_optimized，只是：
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
    # 1) 皮肤水分扩散
    hl1 = 3.05 * 0.001 * (5733.0 - (6.99 * mw) - pa)
    # 2) 出汗散热（代谢率较高时）
    hl2 = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0
    # 3) 呼吸潜热
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    # 4) 呼吸显热
    hl4 = 0.0014 * m * (34.0 - tdb)
    # 5) 辐射散热
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    # 6) 对流散热
    hl6 = f_cl * hc * (tcl - tdb)

    # 热感觉传递系数
    ts = 0.303 * np.exp(-0.036 * m) + 0.028

    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return pmv


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


if __name__ == "__main__":
    # 简单自测
    tdb = 25.0
    tr = 25.0
    vr = 0.1
    rh = 50.0
    met = 1
    clo = 0.5
    wme = 0.0

    pmv_value, ppd_value = pmv_ppd(tdb, tr, vr, rh, met, clo, wme)
    print("PMV:", pmv_value)
    print("PPD:", ppd_value)

    # 也支持数组输入：
    tdb_array = [22.0, 25.0, 28.0]
    pmv_arr, ppd_arr = pmv_ppd(tdb_array, tr, vr, rh, met, clo, wme)
    print("PMV array:", pmv_arr)
    print("PPD array:", ppd_arr)
