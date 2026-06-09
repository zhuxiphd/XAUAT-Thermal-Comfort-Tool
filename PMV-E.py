# -*- coding: utf-8 -*-
"""
PMV-E: 老年人修正 PMV 模型（comfort-Tskin + comfort-Eskin + radiation-feff 版）

修正逻辑：
1) 保留 Fanger/ISO 7730 PMV 的热平衡骨架；
2) 对流换热系数按课题组 Kang 2025 的 18–70 岁整体公式替换：
   - 自然对流: hcn,E = 1.87 * |Tcl - Ta|^0.25
   - 强制对流: hcf,E = 10.833 * v^0.616
   - hc,E = max(hcn,E, hcf,E)
3) 辐射散热项按 Zuo 2026 的有效辐射面积系数进行姿势化修正：
   - 静坐: f_eff,E = 0.79
   - 站立/步行: f_eff,E = 0.84
   - hl5_E = 5.595 * f_eff,E * f_cl * [(Tcl/100)^4 - (Tr/100)^4]
   - 同步用于 Tcl 迭代中的辐射项，避免只改最终输出 R 而 Tcl 仍按原式求解。
4) 皮肤温度与皮肤蒸发散热按“偏好温度舒适稳态”拟合公式替换：
   - 舒适皮肤温度: Tsk,E = 33.46 - 0.003 * M_E；
   - 皮肤净蒸发散热: Eskin,E = 0.552 * M_E - 14.54 (R²=0.70)；
   - measured / literature / mass_g 模式仍保留为敏感性分析或缺失值兜底入口。
5) 以偏好温度状态作为老年人舒适锚点，拟合 TL0,E；
6) 输出 PMV_original 与 PMV_E，便于对照。

单位约定：
- tdb, tr, tskin: °C
- vr: m/s
- rh: %
- met, wme: met
- clo: clo
- height: m 或 cm（函数会自动识别）
- weight: kg
- heat terms, TL, M: W/m²

作者说明：
本代码重点用于论文/模型开发阶段。若 Excel 列名调整，请在 compute_elderly_preferred_rows()
中的列名映射处同步修改。
"""

from __future__ import annotations

import csv
import math
import warnings
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from zipfile import ZipFile
import xml.etree.ElementTree as ET

try:
    import numpy as np
except Exception:  # 拟合功能需要 numpy；单点计算不强制依赖
    np = None

Number = Union[int, float]
MET_TO_W_M2 = 58.15
LATENT_HEAT_J_PER_G = 2430.0  # 约 30°C 附近水汽化潜热，J/g


# =========================================================
# 1. 默认 PMV-E 系数
#    这些系数来自当前上传的老年人偏好温度实验表，并使用：
#    - Kang 2025 18–70 岁自然/强制对流公式；
#    - 偏好温度实验末 10 min 稳态舒适皮温拟合 Tsk,E = 33.46 - 0.003*M_E；
#    - 偏好温度实验末 10 min 净蒸发散热拟合 Eskin,E = 0.552*M_E - 14.54；
#    - 偏好温度状态作为 TSV≈0 的舒适锚点。
# =========================================================
@dataclass
class PMVEConfig:
    # TL0,E = d0 + d_M*M_Wm2
    # 已使用 metMET课题组代谢率数据2026.6.8(2).xlsx 中
    # “偏好温度和热反应实验”子表 A列=“老年人偏好温度实验”的 120 行数据重新拟合。
    # 拟合时使用主模型 comfort_model：
    # Tsk,E = 33.46 - 0.003*M_E；Eskin,E = 0.552*M_E - 14.54。
    # 按你的最新逻辑，TL0,E 只作为代谢率 M_E 的函数，不再单独加入工况哑变量。
    # 使用 AW 列 M(W/m2) / AX 列 met 对应的 M_E，n=120，R²=0.7781474674。
    tl0_d0: float = -15.3129604058
    tl0_d_M: float = 0.2559914488
    # 保留这两个字段仅为兼容旧接口；主模型 tl0_elderly() 不再使用工况项。
    tl0_d_walk3: float = 0.0000000000
    tl0_d_walk5: float = 0.0000000000
    tl0_calibrated: bool = True
    tl0_source: str = "calibrated from elderly preferred-temperature rows; comfort_model Tsk/Eskin; TL0_E=f(M_E) only; n=120; R2=0.7634876299"

    # PMV_E = slope * (TL_E - TL0_E)
    # 该斜率保留文献 Top–TSV/TL 映射的默认值；后续若有完整老年人 TSV 实测，可重新拟合。
    tsv_tl_slope: float = 0.0888454767

    # 主模型：本研究老年人偏好温度实验末 10 min 舒适稳态拟合
    # M_Wm2 为老年人代谢产热，单位 W/m²。
    # Tsk,E = 33.46 - 0.003*M_Wm2
    # Eskin,E = 0.552*M_Wm2 - 14.54, R²=0.70
    tsk_pref_intercept: float = 33.46
    tsk_pref_slope_M: float = -0.003
    tsk_pref_r2: float = 0.07
    eskin_pref_intercept: float = -14.54
    eskin_pref_slope_M: float = 0.552
    eskin_pref_r2: float = 0.70

    # 旧版 Excel-Eskin 模型：仅保留为 eskin_mode="excel_model" 的敏感性分析入口；
    # 主模型不再默认使用该式，以避免高代谢网页单点被外推到不合理蒸发散热。
    # Eskin,E = e0 + e_M*M_Wm2 + e_walk3*I_walk3 + e_walk5*I_walk5
    eskin_e0: float = 18.18532484
    eskin_e_M: float = 2.70152526
    eskin_e_walk3: float = 294.14970941
    eskin_e_walk5: float = 582.97122425

    # 对流公式：18–70 岁整体自然/强制对流
    hcn_A: float = 1.87
    hcn_B: float = 0.25
    hcf_A: float = 10.833
    hcf_B: float = 0.616

    # 辐射公式：Zuo 2026 有效辐射面积系数 feff
    # 该研究显示年龄/性别对 feff 和 fp 的影响较小，姿势影响显著；
    # 因此 PMV-E 只做姿势化修正，不做“老年年龄系数”修正。
    feff_sitting: float = 0.79
    feff_standing: float = 0.84
    radiation_base_coeff: float = 5.595

    # 缺失实测皮温时的 fallback：优先用 Xiong Top–Tsk 稳态关系近似
    # Xiong Fig.4: Top [21,25,28,31], Tsk [31.4,32.7,33.9,34.8]
    # 线性拟合约 Tsk = 0.3421*Top + 24.2526
    tsk_top_intercept: float = 24.2526
    tsk_top_slope: float = 0.3421

    # 缺失实测蒸发散热时的 fallback：Tsuzuki EWL 换算后对 Top 的二次拟合
    # 仅作为兜底，不作为本研究主模型。
    eskin_c0: float = 295.61
    eskin_c_top: float = -21.97285714
    eskin_c_top2: float = 0.42857143

    # 是否使用自由截距版本；默认 False，保证偏好温度舒适锚点处 PMV-E≈0
    use_free_intercept: bool = False
    free_intercept: float = 0.38927522
    free_slope: float = 0.09350479


DEFAULT_CONFIG = PMVEConfig()

# Tsuzuki & Ohfuku 2002 Fig.4：老年人 EWL 从 g/(m²·h) 换算为 W/m² 后的数字化结果。
# 该数据用于 PMV-E 主模型的皮肤净蒸发散热 Eskin,E 文献输入。
TSUZUKI_ELDERLY_EWL_TOP_C = (23.0, 25.0, 27.0, 29.0, 31.0)
TSUZUKI_ELDERLY_EWL_WM2 = (16.5, 14.8, 15.5, 17.2, 27.0)


def interp_clamped(x: float, xp, fp) -> float:
    """一维线性插值；超出文献范围时采用端点值，避免外推导致不合理蒸发散热。"""
    x = float(x)
    xp = list(map(float, xp))
    fp = list(map(float, fp))
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            r = (x - xp[i]) / (xp[i + 1] - xp[i])
            return fp[i] + r * (fp[i + 1] - fp[i])
    return fp[-1]



# =========================================================
# 2. 基础工具函数
# =========================================================
def finite_or_none(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def normalize_height_m(height: Optional[Number]) -> Optional[float]:
    h = finite_or_none(height)
    if h is None or h <= 0:
        return None
    # 大于 3 基本可判断为 cm
    return h / 100.0 if h > 3.0 else h


def bsa_m2(height: Optional[Number], weight: Optional[Number]) -> Optional[float]:
    """体表面积，m²。公式与 PMV-C 中使用的 0.202*m^0.725*kg^0.425 一致。"""
    h = normalize_height_m(height)
    w = finite_or_none(weight)
    if h is None or w is None or w <= 0:
        return None
    return 0.202 * (w ** 0.425) * (h ** 0.725)


def normalize_activity(activity: Optional[str]) -> str:
    s = "" if activity is None else str(activity).strip()
    slow = s.lower().replace(" ", "")
    if "5" in slow and ("km" in slow or "走步" in s or "步行" in s):
        return "walk5"
    if "3" in slow and ("km" in slow or "走步" in s or "步行" in s):
        return "walk3"
    if "4" in slow and ("km" in slow or "走步" in s or "步行" in s):
        return "walk4"
    if "2" in slow and ("km" in slow or "走步" in s or "步行" in s):
        return "walk2"
    if "静坐" in s or "坐" in s or s in {"看电视", "玩手机", "棋牌活动"}:
        return "seated"
    return s or "unknown"


def activity_dummies(activity: Optional[str]) -> Tuple[float, float]:
    group = normalize_activity(activity)
    return (1.0 if group == "walk3" else 0.0, 1.0 if group == "walk5" else 0.0)


def ppd_fanger(pmv: float) -> float:
    return 100.0 - 95.0 * math.exp(-0.03353 * pmv**4.0 - 0.2179 * pmv**2.0)


# =========================================================
# 3. 老年人特异参数：Tskin,E、Eskin,E、hc,E、TL0,E
# =========================================================
def tskin_elderly(
    top_or_tdb: Number,
    M_Wm2: Optional[Number] = None,
    measured_tskin: Optional[Number] = None,
    tskin_mode: str = "comfort_model",
    config: PMVEConfig = DEFAULT_CONFIG,
) -> float:
    """
    老年人舒适/稳态平均皮肤温度 Tsk,E。

    tskin_mode：
    - "comfort_model"：默认。使用本研究老年人偏好温度实验末 10 min 舒适稳态拟合式：
      Tsk,E = 33.46 - 0.003*M_Wm2。
      该式用于网页/论文主模型，使 PMV-E 具有可泛化的公式输入。
    - "measured"：逐行使用输入的实测皮温，适合复算实验原始行。
    - "top_relation" / "xiong"：使用 Xiong 2019 Top–Tsk 稳态关系：
      Tsk,E = 24.2526 + 0.3421*Top。
    - "auto"：优先 measured；缺失时 comfort_model；再缺失时 top_relation。
    """
    mode = str(tskin_mode).lower().strip()

    def _clip_tsk(x: float) -> float:
        return max(28.0, min(37.5, float(x)))

    if mode in {"measured", "row_measured"}:
        v = finite_or_none(measured_tskin)
        if v is None or not (20.0 <= v <= 42.0):
            raise ValueError("tskin_mode='measured' 但 measured_tskin 缺失或超出合理范围。")
        return _clip_tsk(v)

    if mode in {"auto", "measured_first"}:
        v = finite_or_none(measured_tskin)
        if v is not None and 20.0 <= v <= 42.0:
            return _clip_tsk(v)
        if M_Wm2 is not None:
            return _clip_tsk(config.tsk_pref_intercept + config.tsk_pref_slope_M * float(M_Wm2))
        top = float(top_or_tdb)
        return _clip_tsk(config.tsk_top_intercept + config.tsk_top_slope * top)

    if mode in {"comfort_model", "preferred_model", "met_model", "main"}:
        if M_Wm2 is None:
            raise ValueError("tskin_mode='comfort_model' 时必须提供 M_Wm2。")
        return _clip_tsk(config.tsk_pref_intercept + config.tsk_pref_slope_M * float(M_Wm2))

    if mode in {"top_relation", "xiong", "fallback"}:
        top = float(top_or_tdb)
        return _clip_tsk(config.tsk_top_intercept + config.tsk_top_slope * top)

    raise ValueError(f"未知 tskin_mode: {tskin_mode}")

def eskin_from_evaporation_input(
    value: Optional[Number],
    unit: str = "W/m2",
    height: Optional[Number] = None,
    weight: Optional[Number] = None,
    duration_s: float = 600.0,
    latent_heat_j_per_g: float = LATENT_HEAT_J_PER_G,
    subtract_e_res_Wm2: float = 0.0,
) -> Optional[float]:
    """
    将蒸发散热输入统一为 W/m²。

    unit 可选：
    - "W/m2" / "wm2"：输入已经是 W/m²，例如 Excel 中“净蒸发散热”；
    - "g_10min" / "g"：输入是 10 min 体重变化或蒸发量 g，需用 BSA 换算；
    - "g_10min_net"：输入为总质量损失 g，换算为总蒸发散热后减去呼吸潜热，得到皮肤净蒸发散热。

    注意：当前课题组 Excel 同时有“体重变化g”“蒸发散热量W/m2”“净蒸发散热”。
    默认建议直接读取“净蒸发散热”（W/m²），而不是再从 g 重算。
    """
    x = finite_or_none(value)
    if x is None:
        return None
    u = str(unit).lower().replace(" ", "")
    if u in {"w/m2", "wm2", "w·m-2", "w_m2", "w/m²"}:
        return max(0.0, x)
    if u in {"g", "g_10min", "g/10min", "g10min", "g_10min_total", "g_10min_net"}:
        area = bsa_m2(height, weight)
        if area is None or area <= 0:
            raise ValueError("蒸发量为 g 时必须提供 height 和 weight，用于换算 BSA。")
        total_wm2 = x * latent_heat_j_per_g / (duration_s * area)
        if u == "g_10min_net":
            total_wm2 -= float(subtract_e_res_Wm2 or 0.0)
        return max(0.0, total_wm2)
    raise ValueError(f"未知蒸发散热单位: {unit}")


def eskin_elderly(
    top_or_tdb: Number,
    activity: Optional[str] = None,
    M_Wm2: Optional[Number] = None,
    measured_eskin: Optional[Number] = None,
    eskin_unit: str = "W/m2",
    eskin_mode: str = "comfort_model",
    height: Optional[Number] = None,
    weight: Optional[Number] = None,
    duration_s: float = 600.0,
    e_res_Wm2: float = 0.0,
    config: PMVEConfig = DEFAULT_CONFIG,
) -> float:
    """
    老年人皮肤净蒸发散热 Eskin,E。

    eskin_mode：
    - "comfort_model"：默认。使用本研究老年人偏好温度实验末 10 min 净蒸发散热拟合式：
      Eskin,E = 0.552*M_Wm2 - 14.54, R²=0.70；
    - "literature"：使用 Tsuzuki & Ohfuku 2002 Fig.4 老年人 EWL 换算 W/m² 曲线；
    - "measured"：逐行使用输入的实测净蒸发散热；
    - "mass_g"：输入为 10 min 体重变化 g，自动换算为 W/m²；
    - "excel_model"：旧版。使用本研究偏好温度实验拟合 Eskin,E = f(M, activity)，仅用于敏感性分析；
    - "fanger"：不替换皮肤蒸发散热，回退到原 PMV 的 hl1 + hl2。

    推荐：网页/论文主模型用 "comfort_model"；实验复算可用 "measured" 或 "mass_g"；
    文献对照可用 "literature"。
    """
    mode = str(eskin_mode).lower().strip()

    if mode in {"comfort_model", "preferred_model", "met_model", "main"}:
        if M_Wm2 is None:
            raise ValueError("eskin_mode='comfort_model' 时必须提供 M_Wm2。")
        val = config.eskin_pref_intercept + config.eskin_pref_slope_M * float(M_Wm2)
        return max(0.0, val)

    if mode in {"literature", "tsuzuki", "ewl"}:
        # 文献数据为固定温度暴露下的老年人 EWL，按 Top/tdb 插值；
        # 超出 23–31°C 时采用端点值，避免网页任意输入时外推失控。
        return max(0.0, interp_clamped(float(top_or_tdb), TSUZUKI_ELDERLY_EWL_TOP_C, TSUZUKI_ELDERLY_EWL_WM2))

    if mode in {"excel_model", "model"}:
        if M_Wm2 is None:
            raise ValueError("eskin_mode='excel_model' 时必须提供 M_Wm2。")
        I3, I5 = activity_dummies(activity)
        val = config.eskin_e0 + config.eskin_e_M * float(M_Wm2) + config.eskin_e_walk3 * I3 + config.eskin_e_walk5 * I5
        return max(0.0, val)

    if mode == "fanger":
        # 返回 NaN 作为标记，pmv_e_with_components 中将不覆盖 hl1+hl2。
        return float("nan")

    if mode in {"measured", "row_measured"}:
        val = eskin_from_evaporation_input(
            measured_eskin,
            unit=eskin_unit,
            height=height,
            weight=weight,
            duration_s=duration_s,
            subtract_e_res_Wm2=e_res_Wm2,
        )
        if val is None:
            raise ValueError("eskin_mode='measured' 但 measured_eskin 缺失。")
        return val

    if mode in {"mass_g", "g", "g_10min"}:
        val = eskin_from_evaporation_input(
            measured_eskin,
            unit="g_10min_net",
            height=height,
            weight=weight,
            duration_s=duration_s,
            subtract_e_res_Wm2=e_res_Wm2,
        )
        if val is None:
            raise ValueError("eskin_mode='mass_g' 但 measured_eskin 缺失。")
        return val

    if mode == "fallback":
        return max(0.0, interp_clamped(float(top_or_tdb), TSUZUKI_ELDERLY_EWL_TOP_C, TSUZUKI_ELDERLY_EWL_WM2))

    raise ValueError(f"未知 eskin_mode: {eskin_mode}")

def feff_elderly(activity: Optional[str] = None, config: PMVEConfig = DEFAULT_CONFIG) -> float:
    """
    老年人 PMV-E 的有效辐射面积系数。

    依据 Zuo et al. 2026：年龄和性别对 feff/fp 影响较小，姿势影响显著；
    标准/紧凑站姿约 0.84，标准/紧凑坐姿约 0.79。

    因此：
    - 静坐/坐姿：0.79
    - 步行/站立：0.84
    """
    group = normalize_activity(activity)
    raw = "" if activity is None else str(activity).lower()
    if group in {"walk2", "walk3", "walk4", "walk5"} or "walk" in raw or "standing" in raw or "stand" in raw or "步行" in str(activity) or "站" in str(activity):
        return float(config.feff_standing)
    return float(config.feff_sitting)


def radiation_coeff_elderly(activity: Optional[str] = None, config: PMVEConfig = DEFAULT_CONFIG) -> float:
    """PMV-E 辐射散热系数：5.595 * f_eff,E。"""
    return float(config.radiation_base_coeff) * feff_elderly(activity, config=config)


def hc_kang_18_70(tcl: Number, tdb: Number, vr: Number, config: PMVEConfig = DEFAULT_CONFIG) -> Tuple[float, float, float]:
    """
    Kang 2025 课题组 18–70 岁整体对流换热公式。
    返回: hc, hcn, hcf
    """
    dt = abs(float(tcl) - float(tdb))
    v = max(0.0, float(vr))
    hcn = config.hcn_A * (dt ** config.hcn_B) if dt > 0 else 0.0
    hcf = config.hcf_A * (v ** config.hcf_B) if v > 0 else 0.0
    return max(hcn, hcf), hcn, hcf


def tl0_elderly(M_Wm2: Number, activity: Optional[str] = None, config: PMVEConfig = DEFAULT_CONFIG) -> float:
    """
    老年人舒适热负荷锚点 TL0,E。

    最新定版逻辑：TL0,E 只作为老年人代谢产热 M_E 的函数，
    不再单独加入静坐/3 km/h/5 km/h 工况哑变量。
    activity 参数保留是为了兼容旧调用接口，但此处不参与计算。
    """
    return config.tl0_d0 + config.tl0_d_M * float(M_Wm2)


# =========================================================
# 4. PMV 热平衡核心：原始 PMV 与 PMV-E 共用
# =========================================================
def heat_balance_core(
    tdb: Number,
    tr: Number,
    vr: Number,
    rh: Number,
    met: Number,
    clo: Number,
    wme: Number = 0.0,
    tskin_override: Optional[Number] = None,
    eskin_override_Wm2: Optional[Number] = None,
    use_kang_hc: bool = False,
    radiation_coeff_override: Optional[Number] = None,
    config: PMVEConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    """
    Fanger 热平衡核心。
    - use_kang_hc=False: 原始 Fanger/ISO 7730 hc；
    - use_kang_hc=True: PMV-E 使用 Kang 18–70 岁 hcn/hcf；
    - tskin_override: 替换 Fanger 舒适皮温基准；
    - eskin_override_Wm2: 整体替换 hl1 + hl2；
    - radiation_coeff_override: 若提供，则用该系数替换原 PMV 辐射常数 3.96，
      并同步进入 Tcl 迭代和最终 hl5。
    """
    tdb = float(tdb)
    tr = float(tr)
    vr = max(0.0, float(vr))
    rh = max(0.0, min(100.0, float(rh)))
    met = max(0.0, float(met))
    clo = max(0.0, float(clo))
    wme = float(wme or 0.0)

    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo
    m = met * MET_TO_W_M2
    w = wme * MET_TO_W_M2
    mw = m - w

    f_cl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    taa = tdb + 273.0
    tra = tr + 273.0

    # 初始衣服表面温度估计
    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)
    p1 = icl * f_cl

    # 辐射散热系数：原 PMV 为 3.96；PMV-E 可传入 5.595*f_eff,E。
    # 注意：该系数不仅用于最终 hl5，也同步用于 Tcl 迭代中的辐射项。
    rad_coeff = 3.96 if radiation_coeff_override is None else float(radiation_coeff_override)
    p2 = p1 * rad_coeff
    p3 = p1 * 100.0
    p4 = p1 * taa

    if tskin_override is None:
        tsk_ref = 35.7 - 0.028 * mw
        p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    else:
        tsk_ref = float(tskin_override)
        p5 = (tsk_ref + 273.0) + (p2 * (tra / 100.0) ** 4)

    xn = t_cla / 100.0
    xf = t_cla / 50.0
    eps = 0.00015
    hc = 0.0
    hcn = 0.0
    hcf = 0.0

    for n in range(151):
        if abs(xn - xf) <= eps:
            break
        xf = (xf + xn) / 2.0
        tcl_tmp = 100.0 * xf - 273.0
        if use_kang_hc:
            hc, hcn, hcf = hc_kang_18_70(tcl_tmp, tdb, vr, config=config)
        else:
            hcn = 2.38 * abs(tcl_tmp - tdb) ** 0.25
            hcf = 12.1 * math.sqrt(vr)
            hc = max(hcn, hcf)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
    else:
        raise StopIteration("PMV iteration did not converge within 150 steps")

    tcl = 100.0 * xn - 273.0
    if use_kang_hc:
        hc, hcn, hcf = hc_kang_18_70(tcl, tdb, vr, config=config)
    else:
        hcn = 2.38 * abs(tcl - tdb) ** 0.25
        hcf = 12.1 * math.sqrt(vr)
        hc = max(hcn, hcf)

    # 原始 Fanger 皮肤潜热项
    hl1_raw = 3.05e-3 * (5733.0 - 6.99 * mw - pa)  # 皮肤水分扩散
    hl2_raw = 0.42 * (mw - MET_TO_W_M2) if mw > MET_TO_W_M2 else 0.0  # 显著出汗

    if eskin_override_Wm2 is None:
        hl1 = hl1_raw
        hl2 = hl2_raw
    else:
        # 按 PMV-C 思路：把皮肤潜热项作为整体替换。
        # 为了保持 hl1~hl6 输出格式，hl1 置 0，hl2 承载 Eskin,E。
        hl1 = 0.0
        hl2 = max(0.0, float(eskin_override_Wm2))

    hl3 = 1.7e-5 * m * (5867.0 - pa)       # 呼吸潜热
    hl4 = 0.0014 * m * (34.0 - tdb)        # 呼吸显热
    hl5 = rad_coeff * f_cl * (xn**4 - (tra / 100.0) ** 4)  # 辐射
    hl6 = f_cl * hc * (tcl - tdb)           # 对流

    TL = mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * TL

    return {
        "ok": True,
        "pmv": float(pmv),
        "ppd": float(ppd_fanger(pmv)),
        "TL": float(TL),
        "ts_coefficient": float(ts),
        "pa": float(pa),
        "m": float(m),
        "mw": float(mw),
        "met": float(met),
        "wme": float(wme),
        "f_cl": float(f_cl),
        "rad_coeff": float(rad_coeff),
        "hc": float(hc),
        "hcn": float(hcn),
        "hcf": float(hcf),
        "tcl": float(tcl),
        "tsk_ref": float(tsk_ref),
        "hl1": float(hl1),
        "hl2": float(hl2),
        "hl3": float(hl3),
        "hl4": float(hl4),
        "hl5": float(hl5),
        "hl6": float(hl6),
        "e_skin": float(hl1 + hl2),
        "e_res": float(hl3),
        "c_res": float(hl4),
        "q_sensible": float(hl5 + hl6),
        "q_total": float(hl1 + hl2 + hl3 + hl4 + hl5 + hl6),
    }


def pmv_standard_with_components(
    tdb: Number,
    tr: Optional[Number] = None,
    vr: Number = 0.1,
    rh: Number = 50.0,
    met: Number = 1.0,
    clo: Number = 0.6,
    wme: Number = 0.0,
) -> Dict[str, float]:
    tr = tdb if tr is None else tr
    return heat_balance_core(tdb, tr, vr, rh, met, clo, wme, use_kang_hc=False)


def pmv_e_with_components(
    tdb: Number,
    tr: Optional[Number] = None,
    vr: Number = 0.1,
    rh: Number = 50.0,
    met: Number = 1.0,
    clo: Number = 0.6,
    wme: Number = 0.0,
    activity: Optional[str] = "静坐",
    age: Optional[Number] = None,
    height: Optional[Number] = None,
    weight: Optional[Number] = None,
    tskin: Optional[Number] = None,
    tskin_mode: str = "comfort_model",
    eskin: Optional[Number] = None,
    eskin_unit: str = "W/m2",
    eskin_mode: str = "comfort_model",
    eskin_duration_s: float = 600.0,
    config: PMVEConfig = DEFAULT_CONFIG,
) -> Dict[str, float]:
    """
    老年人 PMV-E 主入口。

    参数 tskin：
    - 主模型默认不直接使用逐行 tskin，而是使用 tskin_mode="comfort_model" 的拟合公式；
    - 若要复算实验原始行，可设 tskin_mode="measured"。

    参数 eskin：
    - 主模型默认使用 eskin_mode="comfort_model" 的净蒸发散热拟合公式；
    - 若来自 Excel 的“净蒸发散热”，使用 eskin_mode="measured", eskin_unit="W/m2"；
    - 若来自 10 min 体重变化 g，使用 eskin_mode="mass_g"，并提供 height/weight。
    """
    tdb = float(tdb)
    tr = tdb if tr is None else float(tr)
    met = float(met)
    M_Wm2 = met * MET_TO_W_M2

    original = pmv_standard_with_components(tdb, tr, vr, rh, met, clo, wme)

    # 先算呼吸潜热，用于 g_10min_net 需要扣除 Eres 的情况。
    e_res_for_net = original["hl3"]
    tsk_e = tskin_elderly(
        top_or_tdb=tdb,
        M_Wm2=M_Wm2,
        measured_tskin=tskin,
        tskin_mode=tskin_mode,
        config=config,
    )
    esk_e = eskin_elderly(
        top_or_tdb=tdb,
        activity=activity,
        M_Wm2=M_Wm2,
        measured_eskin=eskin,
        eskin_unit=eskin_unit,
        eskin_mode=eskin_mode,
        height=height,
        weight=weight,
        duration_s=eskin_duration_s,
        e_res_Wm2=e_res_for_net,
        config=config,
    )

    eskin_override = None if (isinstance(esk_e, float) and math.isnan(esk_e)) else esk_e

    f_eff_e = feff_elderly(activity, config=config)
    rad_coeff_e = float(config.radiation_base_coeff) * f_eff_e

    elderly = heat_balance_core(
        tdb, tr, vr, rh, met, clo, wme,
        tskin_override=tsk_e,
        eskin_override_Wm2=eskin_override,
        use_kang_hc=True,
        radiation_coeff_override=rad_coeff_e,
        config=config,
    )

    TL_e = elderly["TL"]
    if not bool(config.tl0_calibrated):
        warnings.warn(
            "TL0,E is using legacy placeholder coefficients. For final PMV-E results, "
            "run fit_tl0_from_preferred_excel(..., tskin_mode='comfort_model', eskin_mode='comfort_model') "
            "and use the returned calibrated config.",
            RuntimeWarning,
            stacklevel=2,
        )
    TL0_e = tl0_elderly(M_Wm2, activity, config=config)
    dTL_e = TL_e - TL0_e

    if config.use_free_intercept:
        pmv_e = config.free_intercept + config.free_slope * dTL_e
    else:
        pmv_e = config.tsv_tl_slope * dTL_e

    return {
        "ok": True,
        "mode": "PMV-E elderly",
        "activity_group": normalize_activity(activity),
        "age": None if age is None else float(age),
        "M_Wm2": float(M_Wm2),
        "met_e": float(met),
        "pmv_original": original["pmv"],
        "ppd_original": original["ppd"],
        "TL_original": original["TL"],
        "pmv_e": float(pmv_e),
        "ppd_e_fanger": float(ppd_fanger(pmv_e)),
        "TL_e": float(TL_e),
        "TL0_e": float(TL0_e),
        "TL0_calibrated": bool(config.tl0_calibrated),
        "TL0_source": str(config.tl0_source),
        "dTL_e": float(dTL_e),
        "tsk_e": float(tsk_e),
        "tskin_mode_used": tskin_mode,
        "eskin_e": float(elderly["e_skin"] if (isinstance(esk_e, float) and math.isnan(esk_e)) else esk_e),
        "eskin_unit_used": eskin_unit,
        "eskin_mode_used": eskin_mode,
        "feff_e": float(f_eff_e),
        "rad_coeff_e": float(rad_coeff_e),
        # PMV-E 分项
        "hc_e": elderly["hc"],
        "hcn_e": elderly["hcn"],
        "hcf_e": elderly["hcf"],
        "tcl_e": elderly["tcl"],
        "rad_coeff_original": original["rad_coeff"],
        "rad_coeff_e_used": elderly["rad_coeff"],
        "hl1_e": elderly["hl1"],
        "hl2_e": elderly["hl2"],
        "hl3_e": elderly["hl3"],
        "hl4_e": elderly["hl4"],
        "hl5_e": elderly["hl5"],
        "hl6_e": elderly["hl6"],
        # 原始 PMV 分项
        "hc_original": original["hc"],
        "hcn_original": original["hcn"],
        "hcf_original": original["hcf"],
        "tcl_original": original["tcl"],
        "hl1_original": original["hl1"],
        "hl2_original": original["hl2"],
        "hl3_original": original["hl3"],
        "hl4_original": original["hl4"],
        "hl5_original": original["hl5"],
        "hl6_original": original["hl6"],
    }


# =========================================================
# 5. xlsx 读取：使用标准库直接读取，不依赖 openpyxl/pandas
# =========================================================
def _col_to_idx(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def _load_shared_strings(z: ZipFile) -> List[str]:
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    return ["".join((t.text or "") for t in si.iter(ns + "t")) for si in root.findall(ns + "si")]


def _parse_xlsx_sheet(z: ZipFile, sheet_path: str) -> List[List[object]]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = _load_shared_strings(z)
    rows: List[List[object]] = []
    with z.open(sheet_path) as f:
        for event, row in ET.iterparse(f, events=("end",)):
            if row.tag != ns + "row":
                continue
            vals: Dict[int, object] = {}
            for c in list(row):
                if c.tag != ns + "c":
                    continue
                ref = c.attrib.get("r")
                if not ref:
                    continue
                idx = _col_to_idx(ref)
                typ = c.attrib.get("t")
                v = c.find(ns + "v")
                value = None
                if v is not None and v.text is not None:
                    text = v.text
                    if typ == "s":
                        value = shared[int(text)] if shared else text
                    else:
                        try:
                            value = float(text) if re.search(r"[.Ee]", text) else int(text)
                        except Exception:
                            value = text
                vals[idx] = value
            if vals:
                rows.append([vals.get(i) for i in range(1, max(vals) + 1)])
            row.clear()
    return rows


def _unique_headers(headers: List[object]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, h in enumerate(headers):
        name = f"col_{i+1}" if h is None else str(h).replace("\n", "").strip()
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def read_xlsx_named_sheets(xlsx_path: Union[str, Path]) -> Dict[str, List[Dict[str, object]]]:
    xlsx_path = Path(xlsx_path)
    with ZipFile(xlsx_path) as z:
        wb_root = ET.fromstring(z.read("xl/workbook.xml"))
        rel_root = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rels = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}
        ns_main = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        ns_rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
        out: Dict[str, List[Dict[str, object]]] = {}
        for sh in wb_root.iter(ns_main + "sheet"):
            name = sh.attrib["name"]
            rid = sh.attrib[ns_rel + "id"]
            target = rels[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target
            rows = _parse_xlsx_sheet(z, target)
            if not rows:
                out[name] = []
                continue
            headers = _unique_headers(rows[0])
            records: List[Dict[str, object]] = []
            for r in rows[1:]:
                r2 = r + [None] * (len(headers) - len(r))
                records.append(dict(zip(headers, r2[: len(headers)])))
            out[name] = records
    return out



def _first_existing(row: Dict[str, object], keys, default=None):
    """按候选列名顺序取第一个非空值，兼容不同版本 Excel 列名。"""
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return default


def _num(row: Dict[str, object], key: str, default: Optional[float] = None) -> Optional[float]:
    v = finite_or_none(row.get(key))
    return default if v is None else v


def _num_any(row: Dict[str, object], keys, default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        v = finite_or_none(row.get(key))
        if v is not None:
            return v
    return default


def _height_m_from_row(row: Dict[str, object]) -> Optional[float]:
    h_m = _num_any(row, ["Height（m）", "Height(m)", "height_m", "身高m", "身高（m）"])
    if h_m is not None:
        return normalize_height_m(h_m)
    h_cm = _num_any(row, ["身高cm", "身高(cm)", "Height（cm）", "Height(cm)"])
    if h_cm is not None:
        return normalize_height_m(h_cm)
    return None


def _weight_kg_from_row(row: Dict[str, object]) -> Optional[float]:
    return _num_any(row, ["Weight（kg）", "Weight(kg)", "weight_kg", "体重kg", "体重(kg)", "体重（kg）"])


def _activity_from_row(row: Dict[str, object]) -> str:
    return str(_first_existing(row, ["Activity", "活动", "测试方法"], "") or "")


def _is_elderly_row(row: Dict[str, object]) -> bool:
    group = str(_first_existing(row, ["Population groups", "分组", "年龄段"], "") or "")
    return "老年" in group or ">60" in group or "elder" in group.lower()


def _has_preferred_temperature(row: Dict[str, object]) -> bool:
    return _num_any(row, ["偏好温度(℃)", "偏好空气温度Ta_pref(℃)tdb_used", "Ta_pref", "Tpref_C"]) is not None


def _is_preferred_experiment_row(row: Dict[str, object]) -> bool:
    """兼容两类 Excel：
    1) 合并表：有“实验波次=老年人偏好温度实验”；
    2) 全年龄段汇总表：无实验波次，但有“分组=老年人”和偏好温度列。
    """
    if not _is_elderly_row(row):
        return False
    wave = str(row.get("实验波次") or "")
    if wave:
        return "偏好温度" in wave and "老年" in wave
    return _has_preferred_temperature(row)


def _extract_preferred_row_values(row: Dict[str, object]) -> Optional[Dict[str, object]]:
    """把不同 Excel 版本的列名统一为 PMV-E 计算需要的字段。"""
    top = _num_any(row, ["偏好温度(℃)", "偏好空气温度Ta_pref(℃)tdb_used", "Ta_pref", "Tpref_C"])
    met = _num_any(row, ["met", "MET"])
    # 部分表只有代谢率 W/m²，没有 met，则反算 met。
    M_wm2 = _num_any(row, ["M_Wm2", "M(W/m2)", "代谢率（W/m2）", "代谢率(W/m2)"])
    if met is None and M_wm2 is not None:
        met = M_wm2 / MET_TO_W_M2
    if top is None or met is None:
        return None
    if not (10.0 <= top <= 40.0) or not (0.3 <= met <= 8.0):
        return None

    tr = _num_any(row, ["辐射温度", "平均辐射温度Tr_used", "tr_C"], top) or top
    rh = _num_any(row, ["相对湿度RH_used", "rh_pct", "RH"], 50.0) or 50.0
    vr = _num_any(row, ["相对风速vr", "vr_m_s", "相对风速", "Va"], 0.1) or 0.1
    clo = _num_any(row, ["服装热阻(clo)", "服装热阻clo", "clo"], 0.6) or 0.6
    wme = _num_any(row, ["外部做功wme", "wme"], 0.0) or 0.0
    tsk = _num_any(row, ["皮肤温度(℃)", "Tskin_input_C", "Tskin_used_C"])
    esk = _num_any(row, ["净蒸发散热", "Eskin_input", "Eskin_used_Wm2"])
    age = _num_any(row, ["Age", "年龄"])
    height = _height_m_from_row(row)
    weight = _weight_kg_from_row(row)
    activity = _activity_from_row(row)

    return {
        "top": float(top),
        "tr": float(tr),
        "rh": float(rh),
        "vr": float(vr),
        "clo": float(clo),
        "wme": float(wme),
        "met": float(met),
        "M_Wm2_input": None if M_wm2 is None else float(M_wm2),
        "tsk": tsk,
        "esk": esk,
        "age": age,
        "height": height,
        "weight": weight,
        "activity": activity,
    }


# =========================================================
# 6. 批量计算和 TL0 重新拟合
# =========================================================
def iter_elderly_preferred_records(xlsx_path: Union[str, Path]) -> List[Dict[str, object]]:
    """读取老年人偏好温度记录，兼容当前课题组合并表和全龄段汇总表。"""
    sheets = read_xlsx_named_sheets(xlsx_path)
    records: List[Dict[str, object]] = []
    for sheet_name, rows in sheets.items():
        if not ("偏好" in sheet_name or "热反应" in sheet_name or "数据汇总" in sheet_name):
            continue
        for row in rows:
            if not _is_preferred_experiment_row(row):
                continue
            vals = _extract_preferred_row_values(row)
            if vals is None:
                continue
            row2 = dict(row)
            row2["__sheet_name"] = sheet_name
            records.append(row2)
    if not records:
        raise ValueError(
            "未找到有效的老年人偏好温度记录。请确认表格包含老年人分组、偏好温度、met或M(W/m2)等列。"
        )
    return records


def fit_tl0_from_preferred_excel(
    xlsx_path: Union[str, Path],
    base_config: PMVEConfig = DEFAULT_CONFIG,
    trim_quantile: float = 0.0,
    eskin_column: str = "净蒸发散热",
    eskin_unit: str = "W/m2",
    eskin_mode: str = "comfort_model",
    tskin_mode: str = "comfort_model",
) -> Tuple[PMVEConfig, Dict[str, float]]:
    """
    用老年人偏好温度实验重新拟合 TL0,E。

    默认使用主模型：
    - Tsk,E = 33.46 - 0.003*M_E；
    - Eskin,E = 0.552*M_E - 14.54。

    拟合目标：在偏好温度舒适状态下，使 TL0,E 拟合由新热平衡算出的 TL_E。
    最新定版逻辑：TL0,E = d0 + d_M*M_E，只用代谢率作为自变量；
    不再加入 walk3 / walk5 工况哑变量。
    """
    if np is None:
        raise ImportError("fit_tl0_from_preferred_excel 需要 numpy。")

    records = iter_elderly_preferred_records(xlsx_path)
    data = []
    skipped = 0
    for row in records:
        vals = _extract_preferred_row_values(row)
        if vals is None:
            skipped += 1
            continue
        if str(tskin_mode).lower().strip() in {"measured", "row_measured"} and vals["tsk"] is None:
            skipped += 1
            continue
        if str(eskin_mode).lower().strip() in {"measured", "row_measured", "mass_g", "g", "g_10min"} and vals["esk"] is None:
            skipped += 1
            continue
        res = pmv_e_with_components(
            tdb=vals["top"],
            tr=vals["tr"],
            vr=vals["vr"],
            rh=vals["rh"],
            met=vals["met"],
            clo=vals["clo"],
            wme=vals["wme"],
            activity=vals["activity"],
            age=vals["age"],
            height=vals["height"],
            weight=vals["weight"],
            tskin=vals["tsk"],
            tskin_mode=tskin_mode,
            eskin=vals["esk"],
            eskin_unit=eskin_unit,
            eskin_mode=eskin_mode,
            config=base_config,
        )
        data.append({
            "TL": res["TL_e"],
            "M": res["M_Wm2"],
            "activity": vals["activity"],
        })

    if not data:
        raise ValueError("没有可用于拟合 TL0,E 的有效老年人偏好温度行。")

    TL = np.array([d["TL"] for d in data], dtype=float)
    use = np.ones(len(data), dtype=bool)
    if 0 < trim_quantile < 0.2 and len(data) > 20:
        lo, hi = np.quantile(TL, [trim_quantile, 1 - trim_quantile])
        use = (TL >= lo) & (TL <= hi)

    # 最新定版：只用 M_E 作为 TL0,E 的自变量，不再加入工况项。
    X = []
    y = []
    for flag, d in zip(use, data):
        if not flag:
            continue
        X.append([1.0, d["M"]])
        y.append(d["TL"])
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ coef
    r2 = 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else float("nan")

    new_config = replace(
        base_config,
        tl0_d0=float(coef[0]),
        tl0_d_M=float(coef[1]),
        tl0_d_walk3=0.0,
        tl0_d_walk5=0.0,
        tl0_calibrated=True,
        tl0_source=f"refit_tl0_from_preferred_excel_M_only(tskin_mode={tskin_mode}, eskin_mode={eskin_mode}, file={Path(xlsx_path).name})",
    )
    info = {
        "n_total": float(len(data) + skipped),
        "n_used": float(len(y)),
        "n_skipped": float(skipped),
        "r2": float(r2),
        "tl0_d0": float(coef[0]),
        "tl0_d_M": float(coef[1]),
        "tl0_d_walk3": 0.0,
        "tl0_d_walk5": 0.0,
        "tl0_formula": f"TL0_E = {coef[0]:.8f} + {coef[1]:.8f}*M_E",
        "tskin_mode": str(tskin_mode),
        "eskin_mode": str(eskin_mode),
        "tl0_model": "M_only",
    }
    return new_config, info


def format_pmve_config_block(config: PMVEConfig) -> str:
    """输出可粘贴回 PMVEConfig 的 TL0 系数块。"""
    return (
        f"tl0_d0: float = {config.tl0_d0:.10f}\n"
        f"tl0_d_M: float = {config.tl0_d_M:.10f}\n"
        f"tl0_d_walk3: float = {config.tl0_d_walk3:.10f}\n"
        f"tl0_d_walk5: float = {config.tl0_d_walk5:.10f}\n"
        f"tl0_calibrated: bool = True\n"
        f"tl0_source: str = {config.tl0_source!r}\n"
    )


def compute_elderly_preferred_rows(
    input_xlsx: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    config: PMVEConfig = DEFAULT_CONFIG,
    refit_tl0: bool = False,
    eskin_column: str = "净蒸发散热",
    eskin_unit: str = "W/m2",
    eskin_mode: str = "comfort_model",
    tskin_mode: str = "comfort_model",
) -> List[Dict[str, object]]:
    """批量读取老年人偏好温度实验并计算 PMV_original 和 PMV-E。"""
    if refit_tl0:
        config, info = fit_tl0_from_preferred_excel(
            input_xlsx,
            base_config=config,
            eskin_column=eskin_column,
            eskin_unit=eskin_unit,
            eskin_mode=eskin_mode,
            tskin_mode=tskin_mode,
        )
        print("Refitted TL0,E:", info)
        print("\nPaste this TL0 block back into PMVEConfig if you want fixed calibrated defaults:\n")
        print(format_pmve_config_block(config))

    records = iter_elderly_preferred_records(input_xlsx)
    out_rows: List[Dict[str, object]] = []
    for row in records:
        vals = _extract_preferred_row_values(row)
        if vals is None:
            continue
        res = pmv_e_with_components(
            tdb=vals["top"],
            tr=vals["tr"],
            vr=vals["vr"],
            rh=vals["rh"],
            met=vals["met"],
            clo=vals["clo"],
            wme=vals["wme"],
            activity=vals["activity"],
            age=vals["age"],
            height=vals["height"],
            weight=vals["weight"],
            tskin=vals["tsk"],
            tskin_mode=tskin_mode,
            eskin=vals["esk"],
            eskin_unit=eskin_unit,
            eskin_mode=eskin_mode,
            config=config,
        )
        out_rows.append({
            "sheet": row.get("__sheet_name"),
            "ID": row.get("ID") or row.get("序号"),
            "姓名": row.get("姓名"),
            "Age": vals["age"],
            "Gender_1M2F": row.get("Gender（1男2女）") or row.get("Gender（1男2女）_0") or row.get("性别"),
            "Activity": vals["activity"],
            "activity_group": res["activity_group"],
            "Tpref_C": vals["top"],
            "tr_C": vals["tr"],
            "rh_pct": vals["rh"],
            "vr_m_s": vals["vr"],
            "clo": vals["clo"],
            "met": vals["met"],
            "M_Wm2": res["M_Wm2"],
            "height_m": vals["height"],
            "weight_kg": vals["weight"],
            "BSA_m2_calc": bsa_m2(vals["height"], vals["weight"]),
            "Tskin_input_C": vals["tsk"],
            "Tskin_used_C": res["tsk_e"],
            "Tskin_mode": res["tskin_mode_used"],
            "Eskin_input": vals["esk"],
            "Eskin_unit": eskin_unit,
            "Eskin_mode": eskin_mode,
            "Eskin_used_Wm2": res["eskin_e"],
            "PMV_original": res["pmv_original"],
            "PPD_original": res["ppd_original"],
            "PMV_E": res["pmv_e"],
            "PPD_E_Fanger": res["ppd_e_fanger"],
            "TL_original": res["TL_original"],
            "TL_E": res["TL_e"],
            "TL0_E": res["TL0_e"],
            "TL0_calibrated": res["TL0_calibrated"],
            "TL0_source": res["TL0_source"],
            "dTL_E": res["dTL_e"],
            "hc_original": res["hc_original"],
            "hcn_original": res["hcn_original"],
            "hcf_original": res["hcf_original"],
            "hc_E_Kang": res["hc_e"],
            "hcn_E_Kang": res["hcn_e"],
            "hcf_E_Kang": res["hcf_e"],
            "feff_E_Zuo2026": res["feff_e"],
            "rad_coeff_E": res["rad_coeff_e"],
            "hl5_E_Radiation": res["hl5_e"],
            "hl2_E_Eskin": res["hl2_e"],
        })

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
            if out_rows:
                writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
                writer.writeheader()
                writer.writerows(out_rows)
    return out_rows


if __name__ == "__main__":
    # 单点示例：默认使用舒适皮温/净蒸发散热拟合公式，并使用已校准 TL0,E
    sample = pmv_e_with_components(
        tdb=26.0,
        tr=26.0,
        vr=0.1,
        rh=50.0,
        met=0.95,
        clo=0.6,
        activity="静坐",
        age=70,
        height=1.65,
        weight=65,
        tskin=33.2,
        eskin=160.0,
        eskin_unit="W/m2",
        eskin_mode="comfort_model",  # 主模型使用 E_SW = 0.552*M_E - 14.54；改成 "measured" 可逐行使用实测净蒸发散热
    )
    print("=== PMV-E single point ===")
    for k in ["pmv_original", "pmv_e", "ppd_original", "ppd_e_fanger", "TL_e", "TL0_e", "dTL_e", "hc_original", "hc_e", "feff_e", "rad_coeff_e", "hl5_e", "eskin_e"]:
        print(f"{k}: {sample[k]:.4f}")

    # 批量示例：读取当前目录下的课题组 Excel，重新拟合 TL0,E 并输出 CSV
    default_xlsx = Path("/mnt/data/metMET课题组代谢率数据2026.6.8(2).xlsx")
    if default_xlsx.exists():
        out_csv = Path("/mnt/data/PMV_E_elderly_results_Monly_TL0.csv")
        rows = compute_elderly_preferred_rows(
            default_xlsx,
            output_csv=out_csv,
            refit_tl0=True,
            eskin_column="净蒸发散热",
            eskin_unit="W/m2",
            tskin_mode="comfort_model",
            eskin_mode="comfort_model",
        )
        print(f"\nSaved {len(rows)} rows to {out_csv}")
