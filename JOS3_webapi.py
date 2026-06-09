import pathlib
from typing import List, Literal, Optional, Union, Dict, Tuple
import traceback
import math
import sys
import mimetypes
import importlib.util

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
from met_prediction_function import (
    list_available_activities,
    load_model_bundle,
    predict_met,
)

PMV_C_PATH = BASE_DIR / "PMV-C.py"
_pmv_c_spec = importlib.util.spec_from_file_location("pmv_c", str(PMV_C_PATH))
pmv_c = None
if _pmv_c_spec and _pmv_c_spec.loader:
    pmv_c = importlib.util.module_from_spec(_pmv_c_spec)
    sys.modules[_pmv_c_spec.name] = pmv_c
    _pmv_c_spec.loader.exec_module(pmv_c)

PMV_E_PATH = BASE_DIR / "PMV-E.py"
_pmv_e_spec = importlib.util.spec_from_file_location("pmv_e", str(PMV_E_PATH))
pmv_e = None
if _pmv_e_spec and _pmv_e_spec.loader:
    pmv_e = importlib.util.module_from_spec(_pmv_e_spec)
    sys.modules[_pmv_e_spec.name] = pmv_e
    _pmv_e_spec.loader.exec_module(pmv_e)

MET_MODEL_PATH = BASE_DIR / "all_final_m5_gam_models.joblib"
_met_model_bundle = None


def _get_met_model_bundle():
    global _met_model_bundle
    if _met_model_bundle is None:
        _met_model_bundle = load_model_bundle(MET_MODEL_PATH)
    return _met_model_bundle
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

# Ensure SVGs are served with the correct MIME type (avoid forced downloads).
mimetypes.add_type("image/svg+xml", ".svg")

# 静态文件（前端资源）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/static/{svg_name}.svg")
def serve_svg(svg_name: str):
    path = STATIC_DIR / f"{svg_name}.svg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SVG not found")
    return FileResponse(
        str(path),
        media_type="image/svg+xml",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/")
def serve_index():
    """返回前端主页面 index.html"""
    return FileResponse(str(STATIC_DIR / "index.html"))


# ========= Debug 接口：确认 Render 上导入的是哪个 jos3 =========
@app.get("/api/debug/jos3")
def debug_jos3():
    return {
        "jos3_file": getattr(jos3, "__file__", None),
        "jos3_version": getattr(jos3, "__version__", None),
        "vendored_src_exists": bool(JOS3_LOCAL_SRC.exists()),
        "vendored_src_path": str(JOS3_LOCAL_SRC),
        "sys_path_head": sys.path[:5],
    }


# ========= 一些通用类型 =========

ScalarOrArray = Union[float, List[float]]
RawValue = Union[float, int, str, None]

# ========= PMV 计算接口 =========


class PMVInput(BaseModel):
    ta: float               # 空气温度
    tr: Optional[float] = None  # 平均辐射温度，可不填
    vel: float              # 风速
    rh: float               # 相对湿度
    met: Optional[float] = None  # 代谢率；PMV 模式可由 activity_type 自动预测
    clo: float              # 服装
    wme: float = 0.0        # 外部功（met），默认 0
    age: Optional[float] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    sex: Optional[Union[int, float, str]] = None
    activity_type: Optional[str] = None
    child_activity: Optional[
        Literal["seated_rest", "reading_writing", "standing_rest", "walk_3kmh"]
    ] = None
    elder_activity: Optional[str] = None
    eskin_mode: Optional[Literal["literature", "measured", "mass_g", "excel_model", "model", "fanger"]] = None
    variant: Literal["adult", "child", "elder"] = "adult"


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


class MetPredictInput(BaseModel):
    activity_type: str
    age: float
    sex: Union[int, float, str] = "male"
    height: float = Field(..., description="Body height [m]; values > 3 are treated as cm")
    weight: float = Field(..., description="Body weight [kg]")


class MetPredictOutput(BaseModel):
    ok: bool
    activity_type: str
    predicted_met: float
    predicted_W_m2: float
    age: float
    sex: int
    sex_label: str
    height_m: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: float
    ffm_alsallami: float
    bsa_m2_dubois: Optional[float] = None
    approx_PI95_lower_met: float
    approx_PI95_upper_met: float
    cv_rmse_used_for_PI: float


@app.get("/api/met_activities")
def api_met_activities():
    """返回代谢率预测模型支持的活动状态。"""
    bundle = _get_met_model_bundle()
    return {"ok": True, "activities": list_available_activities(bundle)}


@app.post("/api/met_predict", response_model=MetPredictOutput)
def api_met_predict(payload: MetPredictInput):
    """根据活动状态、年龄、性别、身高、体重预测代谢率。"""
    height = float(payload.height)
    if height > 3.0:
        height = height / 100.0
    if height <= 0 or payload.weight <= 0:
        raise HTTPException(status_code=400, detail="身高和体重必须为正数")
    try:
        result = predict_met(
            bundle=_get_met_model_bundle(),
            activity_type=payload.activity_type,
            age=payload.age,
            sex=payload.sex,
            height_m=height,
            weight_kg=payload.weight,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MetPredictOutput(ok=True, **result)


def _resolve_pmv_met(payload: PMVInput) -> float:
    if payload.activity_type:
        if payload.age is None or payload.height is None or payload.weight is None or payload.sex is None:
            raise HTTPException(
                status_code=400,
                detail="使用活动状态预测代谢率时必须提供 age、sex、height、weight",
            )
        result = api_met_predict(
            MetPredictInput(
                activity_type=payload.activity_type,
                age=payload.age,
                sex=payload.sex,
                height=payload.height,
                weight=payload.weight,
            )
        )
        return float(result.predicted_met)
    if payload.met is None:
        raise HTTPException(status_code=400, detail="必须提供 met 或 activity_type")
    return float(payload.met)


def _build_pmv_output(res: Dict[str, float]) -> PMVOutput:
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


@app.post("/api/pmv_adult", response_model=PMVOutput)
def api_pmv_adult(payload: PMVInput):
    """严格调用 PMV.py 进行成人 PMV 计算。"""
    tr = payload.tr if payload.tr is not None else payload.ta
    met = _resolve_pmv_met(payload)
    res = pmv_with_components(
        payload.ta,   # tdb
        tr,           # tr
        payload.vel,  # vr
        payload.rh,   # rh
        met,          # met
        payload.clo,  # clo
        payload.wme,  # wme
    )
    return _build_pmv_output(res)


@app.post("/api/pmv_child", response_model=PMVOutput)
def api_pmv_child(payload: PMVInput):
    """严格调用 PMV-C.py 进行儿童 PMV 计算。"""
    if pmv_c is None:
        raise HTTPException(status_code=500, detail="PMV-C 模块未加载，无法计算儿童模式")
    if payload.age is None:
        raise HTTPException(status_code=400, detail="儿童 PMV 需要提供年龄，且范围为 8-18 岁")
    try:
        age_value = float(payload.age)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="儿童年龄必须是有效数字，且范围为 8-18 岁")
    if not age_value.is_integer() or age_value < 8 or age_value > 18:
        raise HTTPException(status_code=400, detail="儿童年龄范围必须为 8-18 岁整数")
    if payload.height is None or payload.weight is None:
        raise HTTPException(status_code=400, detail="儿童 PMV 需要提供身高与体重")
    tr = payload.tr if payload.tr is not None else payload.ta
    met = _resolve_pmv_met(payload)
    res = pmv_c.pmv_with_components_auto(
        payload.ta,   # tdb
        tr,           # tr
        payload.vel,  # vr
        payload.rh,   # rh
        met,          # met
        payload.clo,  # clo
        payload.wme,  # wme
        age=payload.age,
        height=payload.height,
        weight=payload.weight,
        child_activity=None if payload.activity_type else payload.child_activity,
    )
    return _build_pmv_output(res)


def _build_pmv_elder_output(res: Dict[str, float]) -> PMVOutput:
    hl1 = float(res["hl1_e"])
    hl2 = float(res["hl2_e"])
    hl3 = float(res["hl3_e"])
    hl4 = float(res["hl4_e"])
    hl5 = float(res["hl5_e"])
    hl6 = float(res["hl6_e"])
    q_sensible = hl5 + hl6
    c_res = hl4
    e_skin = hl1 + hl2
    e_res = hl3
    return PMVOutput(
        ok=bool(res.get("ok", True)),
        pmv=res["pmv_e"],
        ppd=res["ppd_e_fanger"],
        hl1=hl1,
        hl2=hl2,
        hl3=hl3,
        hl4=hl4,
        hl5=hl5,
        hl6=hl6,
        mw=res.get("mw", res.get("M_Wm2")),
        q_total=hl1 + hl2 + hl3 + hl4 + hl5 + hl6,
        q_sens=q_sensible + c_res,
        q_sensible=q_sensible,
        c_res=c_res,
        q_lat=e_skin + e_res,
        e_skin=e_skin,
        e_res=e_res,
        e_rsw=hl2,
        e_diff=hl1,
    )


@app.post("/api/pmv_elder", response_model=PMVOutput)
def api_pmv_elder(payload: PMVInput):
    """严格调用 PMV-E.py 进行老年人 PMV 修正计算。"""
    if pmv_e is None:
        raise HTTPException(status_code=500, detail="PMV-E 模块未加载，无法计算老人模式")
    elder_age = 70.0 if payload.age is None else float(payload.age)
    if elder_age < 60.0:
        raise HTTPException(status_code=400, detail="PMV-E 老年人模型要求年龄 age >= 60")
    tr = payload.tr if payload.tr is not None else payload.ta
    met = _resolve_pmv_met(payload)
    res = pmv_e.pmv_e_with_components(
        tdb=payload.ta,
        tr=tr,
        vr=payload.vel,
        rh=payload.rh,
        met=met,
        clo=payload.clo,
        wme=payload.wme,
        activity=payload.elder_activity or "静坐",
        age=elder_age,
        height=payload.height if payload.height is not None else 1.65,
        weight=payload.weight if payload.weight is not None else 65.0,
        eskin_mode=payload.eskin_mode or "literature",
    )
    return _build_pmv_elder_output(res)


@app.post("/api/pmv", response_model=PMVOutput)
def api_pmv(payload: PMVInput):
    """兼容旧接口：根据 variant 路由到成人/儿童/老人计算。"""
    if payload.variant == "child":
        return api_pmv_child(payload)
    if payload.variant == "elder":
        return api_pmv_elder(payload)
    return api_pmv_adult(payload)


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
        row_dict = row.dict(exclude_none=True)
        try:
            series = pd.Series(row_dict)
            out = calculate_comfort_parameters(series)

            merged: Dict[str, RawValue] = {}
            merged.update(row_dict)
            merged.update(out)
            merged = {k: _clean(v) for k, v in merged.items()}
            results.append(merged)
        except Exception as e:
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
    Ta: Optional[ScalarOrArray] = Field(None, description="空气温度 Ta [°C]，可为标量或 17 段列表")
    Tr: Optional[ScalarOrArray] = Field(None, description="平均辐射温度 Tr [°C]，可为标量或 17 段列表")
    To: Optional[ScalarOrArray] = Field(
        None,
        description="作用温度 To [°C]（仅在 Ta == Tr 时使用），可为标量或 17 段列表",
    )
    Va: Optional[ScalarOrArray] = Field(None, description="风速 Va [m/s]，可为标量或 17 段列表")
    RH: Optional[ScalarOrArray] = Field(None, description="相对湿度 RH [%]，可为标量或 17 段列表")
    Icl: Optional[ScalarOrArray] = Field(None, description="服装热阻 Icl [clo]，可为标量或 17 段列表")

    met: Optional[float] = Field(None, description="本阶段代谢率 [met]，若 par 未给定，则用于计算 PAR")
    par: Optional[float] = Field(None, description="本阶段物理活动系数 PAR [-]，若给定则优先于 met")
    posture: Optional[Literal["sitting", "standing", "lying"]] = Field(None, description="本阶段姿态")

    time_step: Optional[float] = Field(None, gt=0, description="本阶段时间步长 [s]，缺省则使用全局 time_step")


class SimInput(BaseModel):
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

    ex_output: Optional[Union[str, List[str]]] = Field(
        "all",
        description=(
            'Extra outputs of JOS-3. "all" 输出全部，'
            '"none"/None 只输出默认参数，'
            '或传入字符串列表，例如 ["BFsk", "BFcr", "Tar"]'
        ),
    )

    met: float = Field(1.0, description="Metabolic rate [met]")
    par: Optional[float] = Field(None, description="Physical activity ratio PAR [-]，若给定则优先于 met")
    posture: Literal["sitting", "standing", "lying"] = Field("sitting", description="Initial posture")

    air_temperature: ScalarOrArray = Field(25.0, description="Initial / uniform air temperature Ta [°C]")
    mean_radiant_temperature: ScalarOrArray = Field(25.0, description="Initial / uniform mean radiant temperature Tr [°C]")
    operative_temperature: Optional[ScalarOrArray] = Field(
        None,
        description="Operative temperature To [°C]（仅在 Ta==Tr 时使用），标量或 17 段列表",
    )
    air_speed: ScalarOrArray = Field(0.1, description="Initial / uniform air velocity Va [m/s]")
    relative_humidity: ScalarOrArray = Field(50.0, description="Initial / uniform relative humidity RH [%]")
    clo: ScalarOrArray = Field(0.5, description="Initial / uniform clothing insulation Icl [clo]")

    exposure_minutes: float = Field(30.0, description="Exposure time [min]")
    time_step: float = Field(60.0, description="Global simulation time step [s]")

    scenario: Literal["uniform", "jos3_example", "jos3_example_1", "custom_steps"] = Field(
        "uniform", description="Scenario type"
    )
    steps: Optional[List[EnvStep]] = Field(
        None,
        description="自定义多阶段 / 非均匀环境（仅在 scenario='custom_steps' 时使用）",
    )


class SimOutput(BaseModel):
    time_min: List[float]
    body_parts: List[str]
    TskMean: List[float]
    Tcb: List[float]
    Met: List[float]
    tsk_local: List[List[float]]

    DTS: List[float]
    PPD: List[float]

    raw: Dict[str, List[RawValue]]


def _build_model(sim_input: SimInput):
    """根据 SimInput 构建并配置 JOS-3 模型实例。"""

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

    bmr = getattr(model, "BMR", None)
    if bmr is None:
        bmr = getattr(model, "bmr", None)

    def met_to_par(met_val: Optional[float]) -> float:
        if met_val is None:
            return 1.2
        if bmr is None or bmr == 0:
            return 1.2
        return (met_val * 58.2) / float(bmr)

    if sim_input.par is not None:
        model.PAR = sim_input.par
    else:
        model.PAR = met_to_par(sim_input.met)

    model.posture = sim_input.posture

    def set_env(value: Optional[ScalarOrArray], attr: str):
        if value is None:
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            setattr(model, attr, np.array(value, dtype=float))
        else:
            setattr(model, attr, float(value))

    set_env(sim_input.air_temperature, "Ta")
    set_env(sim_input.mean_radiant_temperature, "Tr")
    if sim_input.operative_temperature is not None:
        set_env(sim_input.operative_temperature, "To")
    set_env(sim_input.air_speed, "Va")
    set_env(sim_input.relative_humidity, "RH")
    set_env(sim_input.clo, "Icl")

    return model, met_to_par


def _compute_dts_ppd(
    time_min: List[float],
    TskMean: List[float],
    Tcb: List[float],
    tsk_neutral: float = 33.7,
    tcb_neutral: float = 36.8,
) -> Tuple[List[float], List[float]]:

    if not time_min or not TskMean or not Tcb:
        return [], []

    t = np.asarray(time_min, dtype=float) * 60.0
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

    if n < 2:
        dt_sec = np.array([60.0], dtype=float)
    else:
        dt_sec = np.diff(t)
        if np.any(dt_sec <= 0):
            positive = dt_sec[dt_sec > 0]
            fallback = float(np.median(positive)) if positive.size > 0 else 60.0
            dt_sec[dt_sec <= 0] = fallback

    delta_tsk = tsk - float(tsk_neutral)
    delta_tcb = tcb_arr - float(tcb_neutral)

    k_skin = 0.35
    k_core = 0.50

    ts_static = k_skin * delta_tsk + k_core * delta_tcb

    dTdt = np.zeros_like(tsk)
    if n > 1:
        dTdt[1:] = np.diff(tsk) / dt_sec

    tau_pos = 300.0
    tau_neg = 600.0

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

    PPD = 100.0 - 95.0 * np.exp(-0.03353 * DTS**4 - 0.2179 * DTS**2)

    return DTS.astype(float).tolist(), PPD.astype(float).tolist()


@app.post("/api/simulate", response_model=SimOutput)
def simulate(sim_input: SimInput):
    try:
        model, met_to_par = _build_model(sim_input)

        if sim_input.scenario == "uniform":
            loops = max(1, int(round(sim_input.exposure_minutes * 60.0 / sim_input.time_step)))
            model.simulate(times=loops, dtime=sim_input.time_step)

        elif sim_input.scenario == "jos3_example":
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

        elif sim_input.scenario == "jos3_example_1":
            # Official example.py: first 60 minutes at To = 28°C, then lower To to 20°C
            model.Ta = 28
            model.Tr = 28
            model.To = 28
            model.RH = 40
            model.Va = 0.2
            model.PAR = 1.2
            model.posture = "sitting"
            model.simulate(times=60, dtime=60)

            # Only change operative temperature to 20°C while keeping other conditions
            model.To = 20
            model.simulate(times=60, dtime=60)

        elif sim_input.scenario == "custom_steps":
            if not sim_input.steps:
                raise HTTPException(status_code=400, detail="scenario='custom_steps' 时必须提供 steps 列表。")

            def set_env(value: Optional[ScalarOrArray], attr: str):
                if value is None:
                    return
                if isinstance(value, (list, tuple, np.ndarray)):
                    setattr(model, attr, np.array(value, dtype=float))
                else:
                    setattr(model, attr, float(value))

            for step in sim_input.steps:
                if step.par is not None:
                    model.PAR = step.par
                elif step.met is not None:
                    model.PAR = met_to_par(step.met)

                if step.posture is not None:
                    model.posture = step.posture

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

        results = model.dict_results()
        df = pd.DataFrame(results)

        # ✅ 正确位置：打印列名到 Render Logs（或本地控制台）
        print("JOS3 columns:", list(df.columns))

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

        if "TskMean" not in df.columns:
            raise RuntimeError("JOS-3 结果中没有 TskMean 列")
        TskMean = pd.to_numeric(df["TskMean"], errors="coerce").astype(float).tolist()

        if "Tcb" in df.columns:
            Tcb = pd.to_numeric(df["Tcb"], errors="coerce").astype(float).tolist()
        else:
            last_val = float(TskMean[-1]) if TskMean else 37.0
            Tcb = [last_val] * len(time_min)

        if "Met" in df.columns:
            Met = pd.to_numeric(df["Met"], errors="coerce").astype(float).tolist()
        else:
            Met = [sim_input.met * 58.2] * len(time_min)

        DTS, PPD = _compute_dts_ppd(time_min, TskMean, Tcb)

        if len(DTS) == len(df):
            df["DTS"] = DTS
            df["PPD"] = PPD

        tsk_local_cols = [c for c in df.columns if c.startswith("Tsk") and c != "TskMean"]
        body_parts = [c[3:] for c in tsk_local_cols]

        tsk_local: List[List[float]] = []
        for col in tsk_local_cols:
            series = pd.to_numeric(df[col], errors="coerce").astype(float).tolist()
            tsk_local.append(series)

        if not tsk_local:
            body_parts = ["WholeBody"]
            tsk_local = [TskMean]

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
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ========= 本地启动 =========

if __name__ == "__main__":
    import uvicorn
    # Use a browser-friendly address; 0.0.0.0 is only for binding.
    uvicorn.run(app, host="127.0.0.1", port=8000)
