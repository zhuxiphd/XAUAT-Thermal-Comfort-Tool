"""
Standalone SET implementation (Gagge two-node / ASHRAE 55 style)

完全独立版本：
- 不依赖 pythermalcomfort
- 只依赖 numpy 和 numba（可选加速）
- 如果你不想装 numba，可以把 try/except 里的 dummy njit 留着，
  或者直接删掉 @njit 装饰器也行。

核心函数：
- _set_single(...) : 标量版本，内部用 Gagge 两节点模型 + 标准环境热平衡求 SET
- set_tmp(...)     : 对外接口，支持标量和数组，会自动广播

单位约定：
- tdb : 干球温度 [°C]
- tr  : 平均辐射温度 [°C]
- v   : 风速 [m/s]
- rh  : 相对湿度 [%]（0–100）
- met : 代谢率 [met]
- clo : 服装热阻 [clo]
- wme : 外部功 [met]
- body_surface_area : 体表面积 [m²]，默认 1.8258
- p_atm : 大气压 [Pa]，默认 101325
- position : "standing" 或 "sitting"
"""

import math
import numpy as np

# --- 可选 numba 加速 ---
try:
    from numba import njit
except ImportError:
    # 如果没装 numba，就用一个空装饰器占位
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


# 1 met = 58.15 W/m²
MET_TO_W_M2 = 58.15


@njit(cache=True)
def _sat_vapor_pressure_torr(temp_c: float) -> float:
    """Gagge / ASHRAE 中常用的饱和水汽压近似公式，单位：torr"""
    return math.exp(18.6686 - 4030.183 / (temp_c + 235.0))


@njit(cache=True)
def _set_single(
    tdb: float,
    tr: float,
    v: float,
    rh: float,
    met: float,
    clo: float,
    wme: float,
    body_surface_area: float,
    p_atm: float,
    posture_code: int,
) -> float:
    """
    标量版 SET 计算（单个人体 / 单工况）。

    posture_code:
        0 -> standing
        1 -> sitting
    """
    # 至少按 0.1 m/s 处理（SET 定义里就是轻微气流）
    air_speed = v if v >= 0.1 else 0.1

    # ---- 一堆模型常数（来自 Gagge 两节点模型 / ASHRAE 55）----
    k_clo = 0.25
    body_weight = 70.0          # [kg]
    met_factor = MET_TO_W_M2    # met -> W/m²
    sbc = 5.6697e-8             # Stefan-Boltzmann [W/m²/K⁴]
    c_sw = 170.0                # 出汗调节系数
    c_dil = 120.0               # 血管扩张系数
    c_str = 0.5                 # 血管收缩系数
    temp_skin_neutral = 33.7    # 中性皮肤温度
    temp_core_neutral = 36.8    # 中性核心温度
    alpha = 0.1                 # 皮肤质量分数
    temp_body_neutral = (
        alpha * temp_skin_neutral + (1.0 - alpha) * temp_core_neutral
    )
    skin_blood_flow_neutral = 6.3  # [kg/h/m²]

    # ---- 初始状态 ----
    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    skin_blood_flow = skin_blood_flow_neutral

    e_skin = 0.1 * met  # 总蒸发散热初值 [W/m²]
    q_dry = 0.0         # 显热散热（对流 + 辐射）
    w = 0.0             # 皮肤润湿度
    e_rsw = 0.0         # 出汗蒸发散热
    e_diff = 0.0        # 通过皮肤扩散的水汽
    q_res = 0.0         # 呼吸潜热
    c_res = 0.0         # 呼吸显热

    pressure_atm = p_atm / 101325.0
    minutes_sim = 60    # 模拟 60 分钟的调节

    # ---- 服装和环境参数 ----
    r_cl = 0.155 * clo
    f_cl = 1.0 + 0.15 * clo      # 着衣面积系数
    lewis = 2.2 / pressure_atm   # Lewis 比

    # 代谢功率
    m = met * met_factor
    ext_work = wme * met_factor
    rm = (met - wme) * met_factor

    # “舒适”工况下的蒸发散热（后面没用到，只是完整保留）
    e_comfort = 0.42 * (rm - met_factor)
    if e_comfort < 0.0:
        e_comfort = 0.0

    # 水汽透过率
    i_cl = 0.45 if clo > 0.0 else 1.0

    # 皮肤最大可行润湿度
    if clo <= 0.0:
        w_limit = 0.38 * air_speed ** (-0.29)
    else:
        w_limit = 0.59 * air_speed ** (-0.08)

    # 实际环境下的对流换热系数
    h_c_nat = 3.0 * pressure_atm ** 0.53
    h_c_forced = 8.6 * (air_speed * pressure_atm) ** 0.53
    h_c = h_c_nat if h_c_nat > h_c_forced else h_c_forced
    if met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        if h_c_met > h_c:
            h_c = h_c_met

    # 初始辐射换热系数
    h_r = 4.7
    h_t = h_r + h_c
    r_a = 1.0 / (f_cl * h_t)
    t_op = (h_r * tr + h_c * tdb) / h_t

    # 空气水汽分压 [torr]
    p_v = rh / 100.0 * _sat_vapor_pressure_torr(tdb)

    # 呼吸散热
    q_res = 0.0023 * m * (44.0 - p_v)     # 潜热
    c_res = 0.0014 * m * (34.0 - tdb)     # 显热

    # ---- 时间步进（两节点人体调节）----
    for _ in range(minutes_sim):
        # 1) 迭代算衣服外表面温度 t_cl
        t_cl = (r_a * t_skin + r_cl * t_op) / (r_a + r_cl)
        for _iter in range(150):
            # 姿态影响辐射面积系数
            rad_area = 0.7 if posture_code == 1 else 0.73
            emissivity = 0.95
            h_r = (
                4.0
                * emissivity
                * sbc
                * ((t_cl + tr) * 0.5 + 273.15) ** 3.0
                * rad_area
            )
            h_t = h_r + h_c
            r_a = 1.0 / (f_cl * h_t)
            t_op = (h_r * tr + h_c * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_cl * t_op) / (r_a + r_cl)
            if abs(t_cl_new - t_cl) <= 0.01:
                t_cl = t_cl_new
                break
            t_cl = t_cl_new

        # 2) 显热散热（皮肤 -> 空气+围护）
        q_dry = (t_skin - t_op) / (r_a + r_cl)

        # 3) 核心 -> 皮肤的导热 / 血流换热
        q_tissue = (t_core - t_skin) * (5.28 + 1.163 * skin_blood_flow)

        # 4) 核心和皮肤热储量
        s_core = m - q_tissue - q_res - c_res - ext_work
        s_skin = q_tissue - q_dry - e_skin

        # 5) 热容
        c_skin = 0.97 * alpha * body_weight
        c_core = 0.97 * (1.0 - alpha) * body_weight

        # 6) 每分钟温度变化
        d_t_skin = (s_skin * body_surface_area) / (c_skin * 60.0)
        d_t_core = (s_core * body_surface_area) / (c_core * 60.0)

        t_skin += d_t_skin
        t_core += d_t_core

        # 7) 平均体温
        t_body = alpha * t_skin + (1.0 - alpha) * t_core

        # 8) 体温反馈信号
        skin_signal = t_skin - temp_skin_neutral
        warm_skin = skin_signal if skin_signal > 0.0 else 0.0
        cold_skin = -skin_signal if skin_signal < 0.0 else 0.0

        core_signal = t_core - temp_core_neutral
        warm_core = core_signal if core_signal > 0.0 else 0.0
        cold_core = -core_signal if core_signal < 0.0 else 0.0

        body_signal = t_body - temp_body_neutral
        warm_body = body_signal if body_signal > 0.0 else 0.0

        # 9) 皮肤血流调节
        skin_blood_flow = (skin_blood_flow_neutral + c_dil * warm_core) / (
            1.0 + c_str * cold_skin
        )
        if skin_blood_flow > 90.0:
            skin_blood_flow = 90.0
        if skin_blood_flow < 0.5:
            skin_blood_flow = 0.5

        # 10) 出汗调节
        m_rsw = c_sw * warm_body * math.exp(warm_skin / 10.7)
        if m_rsw > 500.0:
            m_rsw = 500.0
        e_rsw = 0.68 * m_rsw  # [W/m²]

        # 11) 蒸发阻力
        r_ea = 1.0 / (lewis * f_cl * h_c)     # 空气层
        r_ecl = r_cl / (lewis * i_cl)        # 服装层

        # 蒸发平衡所需散热（这里没直接用到，只是完整保留）
        e_req = rm - q_res - c_res - q_dry

        # 最大蒸发能力
        p_sat_skin = _sat_vapor_pressure_torr(t_skin)
        e_max = (p_sat_skin - p_v) / (r_ea + r_ecl)
        if e_max == 0.0:
            e_max = 1e-3

        if e_max > 0.0:
            p_rsw = e_rsw / e_max
        else:
            p_rsw = 0.0

        w = 0.06 + 0.94 * p_rsw
        e_diff = w * e_max - e_rsw

        # 限制最大润湿度
        if w > w_limit:
            w = w_limit
            p_rsw = w_limit / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max

        if e_max < 0.0:
            e_diff = 0.0
            e_rsw = 0.0
            w = w_limit

        # 总蒸发散热
        e_skin = e_rsw + e_diff
        if e_rsw > 0:
            m_rsw = e_rsw / 0.68
        else:
            m_rsw = 0.0

        # 12) 发抖代谢（寒冷时）
        met_shivering = 19.4 * cold_skin * cold_core
        m = rm + met_shivering

        # 13) 动态更新皮肤质量分数
        alpha = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417)

    # ---- 时间积分结束，得到最终皮肤热平衡 ----
    q_skin = q_dry + e_skin
    p_sat_skin = _sat_vapor_pressure_torr(t_skin)

    # ========== 标准环境（SET 定义的那个虚拟环境） ==========
    # rh = 50%,  v ≈ 0.1 m/s,  tr = tdb

    h_r_std = h_r
    h_c_std = 3.0 * pressure_atm ** 0.53
    if met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        if h_c_met > h_c_std:
            h_c_std = h_c_met
    if h_c_std < 3.0:
        h_c_std = 3.0

    h_t_std = h_c_std + h_r_std

    # 标准化服装热阻（随代谢率调整）
    r_clo_std = 1.52 / ((met - wme) + 0.6944) - 0.1835   # [clo]
    if r_clo_std < 0.0:
        r_clo_std = 0.0
    r_cl_std = 0.155 * r_clo_std                          # [m²·K/W]

    f_a_cl_std = 1.0 + k_clo * r_clo_std

    # 着衣表面积与裸体表面积之比
    f_cl_ratio_std = 1.0 / (1.0 + 0.155 * f_a_cl_std * h_t_std * r_clo_std)

    i_m_std = 0.45
    denom = h_c_std / h_t_std - f_cl_ratio_std * i_m_std
    if denom == 0.0:
        denom = 1e-6
    i_cl_std = i_m_std * h_c_std / h_t_std * (1.0 - f_cl_ratio_std) / denom

    r_a_std = 1.0 / (f_a_cl_std * h_t_std)
    r_ea_std = 1.0 / (lewis * f_a_cl_std * h_c_std)
    r_ecl_std = r_cl_std / (lewis * i_cl_std)

    h_d_std = 1.0 / (r_a_std + r_cl_std)      # 干热传热系数
    h_e_std = 1.0 / (r_ea_std + r_ecl_std)    # 蒸发传热系数

    # ---- 在标准环境里，用牛顿迭代求解 SET ----
    delta = 1e-4
    set_old = t_skin - q_skin / h_d_std  # 初始猜测
    dx = 100.0
    max_iter = 200
    n_iter = 0

    while abs(dx) > 0.01 and n_iter < max_iter:
        # 标准环境空气水汽分压：50% 饱和
        p_room = 0.5 * _sat_vapor_pressure_torr(set_old)

        # 当前 SET 下的热平衡误差
        err1 = q_skin - h_d_std * (t_skin - set_old) - w * h_e_std * (
            p_sat_skin - p_room
        )

        # 用一个很小的扰动近似导数
        t2 = set_old + delta
        p_room2 = 0.5 * _sat_vapor_pressure_torr(t2)
        err2 = q_skin - h_d_std * (t_skin - t2) - w * h_e_std * (p_sat_skin - p_room2)

        denom = err2 - err1
        if denom == 0.0:
            break

        new_set = set_old - delta * err1 / denom
        dx = new_set - set_old
        set_old = new_set
        n_iter += 1

    return set_old


def set_tmp(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    wme=0.0,
    body_surface_area=1.8258,
    p_atm=101325.0,
    position: str = "standing",
):
    """
    计算 Standard Effective Temperature (SET)。

    参数可以是标量或 array-like，会自动广播。
    position: "standing" 或 "sitting"
    返回值：
        - 如果所有输入都是标量 -> float
        - 否则 -> numpy.ndarray
    """
    tdb = np.asarray(tdb, dtype=float)
    tr = np.asarray(tr, dtype=float)
    v = np.asarray(v, dtype=float)
    rh = np.asarray(rh, dtype=float)
    met = np.asarray(met, dtype=float)
    clo = np.asarray(clo, dtype=float)
    wme = np.asarray(wme, dtype=float)
    body_surface_area = np.asarray(body_surface_area, dtype=float)
    p_atm = np.asarray(p_atm, dtype=float)

    posture_code = 1 if position == "sitting" else 0

    # 用 numpy.vectorize 把标量核广播到数组（标量核被 numba JIT 加速）
    vec = np.vectorize(_set_single)
    set_val = vec(tdb, tr, v, rh, met, clo, wme, body_surface_area, p_atm, posture_code)

    if np.size(set_val) == 1:
        return float(set_val)
    return set_val


if __name__ == "__main__":
    # 简单自测（和 pythermalcomfort 文档例子对比）
    tdb = 25.0
    tr = 25.0
    v = 0.1
    rh = 50.0
    met = 1.2
    clo = 0.5

    set_value = set_tmp(tdb, tr, v, rh, met, clo)
    print("SET (single):", set_value)

    # 数组输入示例
    tdb_array = [22.0, 25.0, 28.0]
    set_arr = set_tmp(tdb_array, tr, v, rh, met, clo)
    print("SET array:", set_arr)
