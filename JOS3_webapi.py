from typing import List, Literal, Optional, Union, Dict, Tuple
import pathlib
import traceback
import math
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
JOS3_LOCAL_SRC = BASE_DIR / "jos3_original" / "src"
if JOS3_LOCAL_SRC.exists():
    # Ensure the vendored jos3 package is imported instead of an arbitrary pip release.
    sys.path.insert(0, str(JOS3_LOCAL_SRC))

try:
    import jos3  # noqa: E402  # pylint: disable=wrong-import-position
except ImportError as err:
    raise ImportError(
        f"无法从 {JOS3_LOCAL_SRC} 导入本地 jos3 源码，请确认仓库包含原始 JOS3 代码。"
    ) from err
from PMV import pmv_ppd, pmv_with_components
from SET import set_tmp  # SET 计算函数
from two_node_excel import calculate_comfort_parameters
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ========= FastAPI 基本设置 =========

app = FastAPI(
    title="JOS-3 Thermoregulation Web API",
    description="Web API wrapping the JOS-3 human thermoregulation model.",
)

# 允许前端（包括 Live Server 的 5500 端口）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # 开发阶段直接放开，省心；以后可改成具体域名
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"

# 静态文件（前端资源）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """返回前端主页面 index.html"""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ========= 一些通用类型 =========

ScalarOrArray = Union[float, List[float]]
RawValue = Union[float, int, str, None]

# ========= PMV 计算接口 =========


class PMVInput(BaseModel):
    ta: float               # 空气温度
    tr: Optional[float] = None  # 平均辐射温度，可不填
    vel: float              # 风速
    rh: float               # 相对湿度
    met: float              # 代谢率
    clo: float              # 服装
    wme: float = 0.0        # 外部功（met），默认 0


class PMVOutput(BaseModel):
    ok: bool
    pmv: float
    ppd: float
    hl1: float   # 皮肤水分扩散
    hl2: float   # 出汗
    hl3: float   # 呼吸潜热
    hl4: float   # 呼吸显热
    hl5: float   # 辐射散热
    hl6: float   # 对流散热
    mw: Optional[float] = None
    q_total: Optional[float] = None
    q_sens: Optional[float] = None
    q_sensible: Optional[float] = None
    c_res: Optional[float] = None
    q_lat: Optional[float] = None
    e_skin: Optional[float] = None
    e_res: Optional[float] = None
    e_rsw: Optional[float] = None
    e_diff: Optional[float] = None


@app.post("/api/pmv", response_model=PMVOutput)
def api_pmv(payload: PMVInput):
    """调用 PMV.py 里的 pmv_with_components，返回 PMV / PPD + HL1~HL6。"""

    # 如果没给 tr，就用 ta
    tr = payload.tr if payload.tr is not None else payload.ta

    res = pmv_with_components(
        payload.ta,   # tdb
        tr,           # tr
        payload.vel,  # vr
        payload.rh,   # rh
        payload.met,  # met
        payload.clo,  # clo
        payload.wme,  # wme
    )

    return PMVOutput(
        ok=res["ok"],
        pmv=res["pmv"],
        ppd=res["ppd"],
        hl1=res["hl1"],
        hl2=res["hl2"],
        hl3=res["hl3"],
        hl4=res["hl4"],
        hl5=res["hl5"],
        hl6=res["hl6"],
        mw=res.get("mw"),
        q_total=res.get("q_total"),
        q_sens=res.get("q_sens"),
        q_sensible=res.get("q_sensible"),
        c_res=res.get("c_res"),
        q_lat=res.get("q_lat"),
        e_skin=res.get("e_skin"),
        e_res=res.get("e_res"),
        e_rsw=res.get("e_rsw"),
        e_diff=res.get("e_diff"),
    )


# ========= SET 计算接口 =========

class SETInput(BaseModel):
    ta: float                           # 空气温度 [°C]
    tr: Optional[float] = None         # 平均辐射温度 [°C]，可选
    vel: float                         # 风速 [m/s]
    rh: float                          # 相对湿度 [%]
    met: float                         # 代谢率 [met]
    clo: float                         # 服装 [clo]
    wme: float = 0.0                   # 外功 [met]
    body_surface_area: float = 1.8     # 体表面积 [m²]
    pressure_pa: float = 101_325.0     # 大气压 [Pa]
    posture: Literal["sitting", "standing"] = "sitting"  # 姿势


class SETOutput(BaseModel):
    ok: bool
    set: float


@app.post("/api/set", response_model=SETOutput)
def api_set(payload: SETInput):
    """调用 SET.py 里的 set_tmp，计算 SET。"""

    # 如果没给 tr，就用 ta
    tr = payload.tr if payload.tr is not None else payload.ta

    # set_tmp 的参数名是 (tdb, tr, v, rh, met, clo, wme, body_surface_area, p_atm, position)
    value = set_tmp(
        payload.ta,                   # tdb
        tr,                           # tr
        payload.vel,                  # v
        payload.rh,                   # rh
        payload.met,                  # met
        payload.clo,                  # clo
        payload.wme,                  # wme
        payload.body_surface_area,    # body_surface_area
        payload.pressure_pa,          # p_atm
        payload.posture               # position: "sitting"/"standing"
    )

    value = float(value)  # numpy 标量统一转 float

    return SETOutput(ok=True, set=value)


# ========= 两节点 (Gagge) 模型接口 =========

class TwoNodeRow(BaseModel):
    tdb: float      # 干球温度（°C）
    tr: float       # 平均辐射温度（°C）
    v: float        # 风速（m/s）
    rh: float       # 相对湿度（%）
    met: float      # 代谢率（met）
    clo: float      # 服装（clo）
    wme: float      # 外功（met）
    weight: float   # 体重（kg）
    height: float   # 身高（m）

    # 以下为可选“高级调参”参数，不给则使用 two_node_excel.py 中的默认值
    temp_skin_neutral: Optional[float] = None
    temp_core_neutral: Optional[float] = None
    max_skin_blood_flow: Optional[float] = None
    max_sweating: Optional[float] = None
    c_sw: Optional[float] = None
    c_dil: Optional[float] = None
    c_str: Optional[float] = None
    skin_blood_flow_neutral: Optional[float] = None

    e_skin_coeff: Optional[float] = None
    e_comfort_coeff: Optional[float] = None
    w_max_coeff_clo0: Optional[float] = None
    w_max_exp_clo0: Optional[float] = None
    w_max_coeff_clo_pos: Optional[float] = None
    w_max_exp_clo_pos: Optional[float] = None
    i_cl_nude: Optional[float] = None
    i_cl_clothed: Optional[float] = None

    air_speed_min: Optional[float] = None
    h_c_s_min: Optional[float] = None
    length_time_simulation: Optional[int] = None


class TwoNodeBatchInput(BaseModel):
    rows: List[TwoNodeRow]


class TwoNodeBatchOutput(BaseModel):
    ok: bool
    results: List[Dict[str, RawValue]]


@app.post("/api/two_node", response_model=TwoNodeBatchOutput)
def api_two_node(payload: TwoNodeBatchInput):
    """批量调用 two_node_excel.calculate_comfort_parameters"""

    results: List[Dict[str, RawValue]] = []

    def _clean(val: RawValue) -> RawValue:
        """将 numpy / 非有限数字转换为 Python 标量或 None，避免 JSON 失败。"""
        if isinstance(val, np.generic):
            val = val.item()
        if isinstance(val, (float, int)):
            try:
                v = float(val)
            except Exception:
                return None
            return v if math.isfinite(v) else None
        return val

    for row in payload.rows:
        # exclude_none=True：只把用户真正填写的高级调参项传下去，其余使用默认
        row_dict = row.dict(exclude_none=True)
        try:
            series = pd.Series(row_dict)
            out = calculate_comfort_parameters(series)

            merged: Dict[str, RawValue] = {}
            merged.update(row_dict)   # 原始输入
            merged.update(out)        # 计算结果
            merged = {k: _clean(v) for k, v in merged.items()}
            results.append(merged)
        except Exception as e:
            # 出错时，保证结构一致，填“计算错误”
            error_result: Dict[str, RawValue] = {
                k: "计算错误"
                for k in [
                    "n_simulation", "Q_sens", "q_sensible", "c_res", "Q_lat",
                    "e_skin", "e_res", "e_rsw", "e_diff", "Q_skin", "Q_resp",
                    "e_max", "m_bl", "m_rsw", "w", "w_max",
                    "t_skin", "t_core", "_set", "et",
                    "t_sens", "disc", "pmv_gagge", "pmv_set", "ps",
                    "r_clo_s", "h_c_s",
                ]
            }
            error_result.update(row_dict)
            error_result["error"] = str(e)
            results.append(error_result)

    return TwoNodeBatchOutput(ok=True, results=results)


# ========= JOS-3 工况定义 =========

class EnvStep(BaseModel):
    """单个阶段 / 工况。可以是非均匀 + 非稳态环境。"""

    duration_minutes: float = Field(..., gt=0, description="本阶段持续时间 [min]")
    Ta: Optional[ScalarOrArray] = Field(
        None, description="空气温度 Ta [°C]，可为标量或 17 段列表"
    )
    Tr: Optional[ScalarOrArray] = Field(
        None, description="平均辐射温度 Tr [°C]，可为标量或 17 段列表"
    )
    To: Optional[ScalarOrArray] = Field(
        None,
        description="作用温度 To [°C]（仅在 Ta == Tr 时使用），可为标量或 17 段列表",
    )
    Va: Optional[ScalarOrArray] = Field(
        None, description="风速 Va [m/s]，可为标量或 17 段列表"
    )
    RH: Optional[ScalarOrArray] = Field(
        None, description="相对湿度 RH [%]，可为标量或 17 段列表"
    )
    Icl: Optional[ScalarOrArray] = Field(
        None, description="服装热阻 Icl [clo]，可为标量或 17 段列表"
    )

    met: Optional[float] = Field(
        None, description="本阶段代谢率 [met]，若 par 未给定，则用于计算 PAR"
    )
    par: Optional[float] = Field(
        None,
        description="本阶段物理活动系数 PAR [-]，若给定则优先于 met",
    )
    posture: Optional[Literal["sitting", "standing", "lying"]] = Field(
        None, description="本阶段姿态"
    )

    time_step: Optional[float] = Field(
        None, gt=0, description="本阶段时间步长 [s]，缺省则使用全局 time_step"
    )


class SimInput(BaseModel):
    # ========= 个体参数（全部透传给 JOS-3 构造器） =========
    height: float = Field(1.7, description="Body height [m]")
    weight: float = Field(70.0, description="Body weight [kg]")
    age: int = Field(30, description="Age [years]")
    sex: Literal["male", "female"] = Field("male", description="Sex")
    fat: float = Field(20.0, description="Body fat percentage [%]")
    ci: float = Field(2.6432, description="Cardiac index [L/min/m²]")

    bmr_equation: Literal["harris-benedict", "japanese"] = Field(
        "harris-benedict", description="Basal metabolic rate equation"
    )
    bsa_equation: Literal["dubois", "fujimoto", "kurazumi", "takahira"] = Field(
        "dubois", description="Body surface area equation"
    )

    # ex_output 原样透传给 JOS-3：None / 'all' / ['BFsk', ...]
    ex_output: Optional[Union[str, List[str]]] = Field(
        "all",
        description=(
            'Extra outputs of JOS-3. "all" 输出全部，'
            '"none"/None 只输出默认参数，'
            '或传入字符串列表，例如 ["BFsk", "BFcr", "Tar"]'
        ),
    )

    # ========= 活动 & 姿态 =========
    met: float = Field(1.0, description="Metabolic rate [met]")
    par: Optional[float] = Field(
        None, description="Physical activity ratio PAR [-]，若给定则优先于 met"
    )
    posture: Literal["sitting", "standing", "lying"] = Field(
        "sitting", description="Initial posture"
    )

    # ========= 全局 / 初始环境条件（作为所有工况的起点） =========
    air_temperature: ScalarOrArray = Field(
        25.0, description="Initial / uniform air temperature Ta [°C]"
    )
    mean_radiant_temperature: ScalarOrArray = Field(
        25.0, description="Initial / uniform mean radiant temperature Tr [°C]"
    )
    operative_temperature: Optional[ScalarOrArray] = Field(
        None,
        description="Operative temperature To [°C]（仅在 Ta==Tr 时使用），标量或 17 段列表",
    )
    air_speed: ScalarOrArray = Field(
        0.1, description="Initial / uniform air velocity Va [m/s]"
    )
    relative_humidity: ScalarOrArray = Field(
        50.0, description="Initial / uniform relative humidity RH [%]"
    )
    clo: ScalarOrArray = Field(
        0.5, description="Initial / uniform clothing insulation Icl [clo]"
    )

    # ========= 单阶段工况参数（scenario='uniform' 时使用） =========
    exposure_minutes: float = Field(30.0, description="Exposure time [min]")
    time_step: float = Field(60.0, description="Global simulation time step [s]")

    # ========= 工况模式 =========
    scenario: Literal["uniform", "jos3_example", "custom_steps"] = Field(
        "uniform", description="Scenario type"
    )
    steps: Optional[List[EnvStep]] = Field(
        None,
        description="自定义多阶段 / 非均匀环境（仅在 scenario='custom_steps' 时使用）",
    )


class SimOutput(BaseModel):
    # 兼容以前接口 + 暴露全部原始结果
    time_min: List[float]
    body_parts: List[str]  # 17 个身体分段名称（从列名 TskXXX 解析）
    TskMean: List[float]
    Tcb: List[float]
    Met: List[float]
    tsk_local: List[List[float]]  # [n_parts][n_time]

    # Fiala 风格热感觉 / 热舒适输出
    DTS: List[float]  # Dynamic Thermal Sensation
    PPD: List[float]  # Predicted Percentage of Dissatisfied

    # 完整输出：dict_results() 的每一列原样给出（不丢任何 JOS-3 功能）
    raw: Dict[str, List[RawValue]]


def _build_model(sim_input: SimInput):
    """根据 SimInput 构建并配置 JOS-3 模型实例。"""

    # ex_output 解析：'none'/'None' -> None
    ex_output = sim_input.ex_output
    if isinstance(ex_output, str) and ex_output.lower() == "none":
        ex_output_param = None
    else:
        ex_output_param = ex_output

    model = jos3.JOS3(
        height=sim_input.height,
        weight=sim_input.weight,
        fat=sim_input.fat,
        age=sim_input.age,
        sex=sim_input.sex,
        ci=sim_input.ci,
        bmr_equation=sim_input.bmr_equation,
        bsa_equation=sim_input.bsa_equation,
        ex_output=ex_output_param,
    )

    # 获取 BMR，用于 met -> PAR 的转换
    bmr = getattr(model, "BMR", None)
    if bmr is None:
        bmr = getattr(model, "bmr", None)

    def met_to_par(met_val: Optional[float]) -> float:
        if met_val is None:
            return 1.2  # fallback：安静坐姿
        if bmr is None or bmr == 0:
            return 1.2
        return (met_val * 58.2) / float(bmr)

    # 设置初始 PAR
    if sim_input.par is not None:
        model.PAR = sim_input.par
    else:
        model.PAR = met_to_par(sim_input.met)

    # 初始姿势
    model.posture = sim_input.posture

    # 通用环境设置函数（支持标量 / 17 段列表）
    def set_env(value: Optional[ScalarOrArray], attr: str):
        if value is None:
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            setattr(model, attr, np.array(value, dtype=float))
        else:
            setattr(model, attr, float(value))

    # 初始化环境（所有 scenario 共用的“初始状态”）
    set_env(sim_input.air_temperature, "Ta")
    set_env(sim_input.mean_radiant_temperature, "Tr")
    if sim_input.operative_temperature is not None:
        set_env(sim_input.operative_temperature, "To")
    set_env(sim_input.air_speed, "Va")
    set_env(sim_input.relative_humidity, "RH")
    set_env(sim_input.clo, "Icl")

    return model, met_to_par


# ========= Fiala 风格 DTS / PPD 计算（基于 JOS-3 输出） =========

def _compute_dts_ppd(
    time_min: List[float],
    TskMean: List[float],
    Tcb: List[float],
    tsk_neutral: float = 33.7,
    tcb_neutral: float = 36.8,
) -> Tuple[List[float], List[float]]:

    if not time_min or not TskMean or not Tcb:
        return [], []

    t = np.asarray(time_min, dtype=float) * 60.0  # [s]
    tsk = np.asarray(TskMean, dtype=float)
    tcb_arr = np.asarray(Tcb, dtype=float)

    n = len(tsk)
    n_min = min(len(t), len(tsk), len(tcb_arr))
    if n_min == 0:
        return [], []
    if n_min != n:
        t = t[:n_min]
        tsk = tsk[:n_min]
        tcb_arr = tcb_arr[:n_min]
        n = n_min

    # 时间步长 [s]
    if n < 2:
        dt_sec = np.array([60.0], dtype=float)
    else:
        dt_sec = np.diff(t)
        if np.any(dt_sec <= 0):
            positive = dt_sec[dt_sec > 0]
            fallback = float(np.median(positive)) if positive.size > 0 else 60.0
            dt_sec[dt_sec <= 0] = fallback

    # ---------- 静态部分：皮肤 + 核心温度偏差 ----------
    delta_tsk = tsk - float(tsk_neutral)
    delta_tcb = tcb_arr - float(tcb_neutral)

    k_skin = 0.35  # 皮肤权重
    k_core = 0.50  # 核心权重

    ts_static = k_skin * delta_tsk + k_core * delta_tcb

    # ---------- 动态部分：基于 dTsk/dt 的指数记忆 ----------
    dTdt = np.zeros_like(tsk)
    if n > 1:
        dTdt[1:] = np.diff(tsk) / dt_sec

    tau_pos = 300.0  # [s] 升温记忆时间常数 ~5 min
    tau_neg = 600.0  # [s] 降温记忆时间常数 ~10 min

    ts_dyn = np.zeros_like(tsk)
    pos_state = 0.0
    neg_state = 0.0

    for i in range(1, n):
        dt = float(dt_sec[i - 1])
        alpha_pos = float(np.exp(-dt / tau_pos))
        alpha_neg = float(np.exp(-dt / tau_neg))

        d = float(dTdt[i])
        if d >= 0.0:
            pos_state = pos_state * alpha_pos + d
        else:
            neg_state = neg_state * alpha_neg + d

        k_pos = 200.0
        k_neg = 200.0
        ts_dyn[i] = k_pos * pos_state + k_neg * neg_state

    DTS = ts_static + ts_dyn
    DTS = np.clip(DTS, -3.0, 3.0)

    # ---------- PPD：Fanger 型公式 ----------
    PPD = 100.0 - 95.0 * np.exp(-0.03353 * DTS**4 - 0.2179 * DTS**2)

    return DTS.astype(float).tolist(), PPD.astype(float).tolist()


# ========= 核心接口：调用 JOS-3（完整功能 + Fiala 输出） =========

@app.post("/api/simulate", response_model=SimOutput)
def simulate(sim_input: SimInput):
    try:
        # ---- 1. 构建模型 ----
        model, met_to_par = _build_model(sim_input)

        # ---- 2. 按 scenario 执行模拟 ----
        if sim_input.scenario == "uniform":
            loops = max(
                1,
                int(round(sim_input.exposure_minutes * 60.0 / sim_input.time_step)),
            )
            model.simulate(times=loops, dtime=sim_input.time_step)

        elif sim_input.scenario == "jos3_example":
            # 示例工况（保持不动）
            model.Ta = 28
            model.Tr = 30
            model.RH = 40
            model.Va = 0.2
            model.PAR = 1.2
            model.posture = "sitting"
            model.Icl = np.array(
                [
                    0.00, 0.00, 1.14, 0.84, 1.04,
                    0.84, 0.42, 0.00,
                    0.84, 0.42, 0.00,
                    0.58, 0.62, 0.82,
                    0.58, 0.62, 0.82,
                ]
            )
            model.simulate(times=30, dtime=60)

            model.To = 20
            model.Va = np.array(
                [
                    0.2, 0.4, 0.4, 0.1, 0.1,
                    0.4, 0.4, 0.4,
                    0.4, 0.4, 0.4,
                    0.1, 0.1, 0.1,
                    0.1, 0.1, 0.1,
                ]
            )
            model.simulate(times=60, dtime=60)

            model.Ta = 30
            model.Tr = 35
            model.simulate(times=30, dtime=60)

        elif sim_input.scenario == "custom_steps":
            if not sim_input.steps:
                raise HTTPException(
                    status_code=400,
                    detail="scenario='custom_steps' 时必须提供 steps 列表。",
                )

            def set_env(value: Optional[ScalarOrArray], attr: str):
                if value is None:
                    return
                if isinstance(value, (list, tuple, np.ndarray)):
                    setattr(model, attr, np.array(value, dtype=float))
                else:
                    setattr(model, attr, float(value))

            for step in sim_input.steps:
                # 活动强度 / 姿态
                if step.par is not None:
                    model.PAR = step.par
                elif step.met is not None:
                    model.PAR = met_to_par(step.met)

                if step.posture is not None:
                    model.posture = step.posture

                # 环境条件
                set_env(step.Ta, "Ta")
                set_env(step.Tr, "Tr")
                set_env(step.To, "To")
                set_env(step.Va, "Va")
                set_env(step.RH, "RH")
                set_env(step.Icl, "Icl")

                dt = step.time_step or sim_input.time_step
                loops = max(1, int(round(step.duration_minutes * 60.0 / float(dt))))
                model.simulate(times=loops, dtime=dt)

        else:
            raise HTTPException(status_code=400, detail=f"未知 scenario: {sim_input.scenario}")

        # ---- 3. 取结果 ----
        results = model.dict_results()
        df = pd.DataFrame(results)

        # 时间列
        if "ModTime" in df.columns:
            time_col = df["ModTime"]
        elif "Time" in df.columns:
            time_col = df["Time"]
        else:
            raise RuntimeError("JOS-3 结果中找不到时间列（ModTime / Time）")

        if np.issubdtype(time_col.dtype, np.timedelta64):
            time_min = (time_col.dt.total_seconds() / 60.0).astype(float).tolist()
        else:
            time_min = (pd.to_numeric(time_col, errors="coerce") / 60.0).astype(float).tolist()

        # 全身平均皮肤温度
        if "TskMean" not in df.columns:
            raise RuntimeError("JOS-3 结果中没有 TskMean 列")
        TskMean = pd.to_numeric(df["TskMean"], errors="coerce").astype(float).tolist()

        # 核心温度
        if "Tcb" in df.columns:
            Tcb = pd.to_numeric(df["Tcb"], errors="coerce").astype(float).tolist()
        else:
            last_val = float(TskMean[-1]) if TskMean else 37.0
            Tcb = [last_val] * len(time_min)

        # 代谢率
        if "Met" in df.columns:
            Met = pd.to_numeric(df["Met"], errors="coerce").astype(float).tolist()
        else:
            Met = [sim_input.met * 58.2] * len(time_min)

        # 计算 DTS / PPD
        DTS, PPD = _compute_dts_ppd(time_min, TskMean, Tcb)

        if len(DTS) == len(df):
            df["DTS"] = DTS
            df["PPD"] = PPD

        # 局部皮肤温度列
        tsk_local_cols = [c for c in df.columns if c.startswith("Tsk") and c != "TskMean"]
        body_parts = [c[3:] for c in tsk_local_cols]

        tsk_local: List[List[float]] = []
        for col in tsk_local_cols:
            series = pd.to_numeric(df[col], errors="coerce").astype(float).tolist()
            tsk_local.append(series)

        if not tsk_local:
            body_parts = ["WholeBody"]
            tsk_local = [TskMean]

        # raw 输出：用 None 代替 NaN / NaT / numpy 类型等
        raw_df = df.where(pd.notnull(df), None)

        raw: Dict[str, List[RawValue]] = {}
        for col in raw_df.columns:
            series = raw_df[col]
            values: List[RawValue] = []
            for v in series:
                if v is None:
                    values.append(None)
                    continue
                if isinstance(v, pd.Timedelta):
                    values.append(v.total_seconds())
                    continue
                if isinstance(v, np.timedelta64):
                    values.append(float(v / np.timedelta64(1, "s")))
                    continue
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, (float, int, str)):
                    values.append(v)
                else:
                    values.append(str(v))
            raw[col] = values

        return SimOutput(
            time_min=time_min,
            body_parts=body_parts,
            TskMean=TskMean,
            Tcb=Tcb,
            Met=Met,
            tsk_local=tsk_local,
            DTS=DTS,
            PPD=PPD,
            raw=raw,
        )

    except HTTPException:
        # FastAPI 会自动处理 HTTPException
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ========= 本地启动 =========

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
