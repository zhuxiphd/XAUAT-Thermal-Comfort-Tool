# -*- coding: utf-8 -*-
#下面这一大段是Python 的文档字符串（docstring），是正式的代码组成部分
"""
Prediction utility for the final activity-specific M5-GAM metabolic rate models.

Final model:
    met ~ s(Age) + Sex + s(BMI) + s(FFM-AlSallami)

Default sex coding:
    0 = male
    1 = female

Example:
    from met_prediction_function import load_model_bundle, predict_met

    bundle = load_model_bundle("./met_step04_final_m5_gam_outputs/trained_m5_gam_models/all_final_m5_gam_models.joblib")
    result = predict_met(
        bundle=bundle,
        activity_type="Reclining",
        age=10,
        sex="male",
        height_cm=136,
        weight_kg=38,
    )
    print(result)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MET_TO_W_M2 = 58.2
M5_FEATURES = ["Age", "Sex", "BMI", "FFM-AlSallami"]


def parse_sex(sex: int | float | str) -> int:
    if isinstance(sex, str):
        s = sex.strip().lower()
        if s in ["male", "m", "0", "男", "男性"]:
            return 0
        if s in ["female", "f", "1", "女", "女性"]:
            return 1
        raise ValueError("sex must be 0/1, male/female, 男/女")
    sex_num = int(sex)
    if sex_num not in [0, 1]:
        raise ValueError("sex must be coded as 0 for male or 1 for female")
    return sex_num

#BMI的计算
def calculate_bmi(height_m: float, weight_kg: float) -> float:
    return float(weight_kg / (height_m ** 2))

#BSA的计算
def calculate_bsa_dubois(height_m: float, weight_kg: float) -> float:
    """Du Bois body surface area equation. Height is in meters."""
    height_cm = height_m * 100.0
    return float(0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425))

#FFM 计算（Al-Sallami 公式）
def calculate_ffm_alsallami(age: float, sex: int | float | str, height_m: float, weight_kg: float) -> float:
    """
    Calculate FFM using the Al-Sallami age-adjusted Janmahasatian approach.

    Sex coding:
        0 or 'male'   -> male
        1 or 'female' -> female
    """
    sex_num = parse_sex(sex)
    bmi = calculate_bmi(height_m, weight_kg)
    if sex_num == 0:
        adult_ffm = 9270.0 * weight_kg / (6680.0 + 216.0 * bmi)
        factor = 0.88 + (1.0 - 0.88) / (1.0 + (age / 13.4) ** (-12.7))
    else:
        adult_ffm = 9270.0 * weight_kg / (8780.0 + 244.0 * bmi)
        factor = 1.11 + (1.0 - 1.11) / (1.0 + (age / 7.1) ** (-1.1))
    return float(factor * adult_ffm)

#模型加载
def load_model_bundle(model_bundle_path: str | Path | None = None) -> Dict[str, Any]:
    """Load the saved model bundle.

    If model_bundle_path is not provided, the function first looks for the
    model bundle in a trained_m5_gam_models folder next to this script.
    This makes the utility portable when it is kept inside the Step 04 output
    folder.  自动查找模型文件，支持默认路径搜索
    """
    if model_bundle_path is None:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "trained_m5_gam_models" / "all_final_m5_gam_models.joblib",
            script_dir / "met_step04_final_m5_gam_outputs" / "trained_m5_gam_models" / "all_final_m5_gam_models.joblib",
            Path.cwd() / "met_step04_final_m5_gam_outputs" / "trained_m5_gam_models" / "all_final_m5_gam_models.joblib",
        ]
        for c in candidates:
            if c.exists():
                return joblib.load(c)
        raise FileNotFoundError("Model bundle not found. Please pass model_bundle_path explicitly.")
    model_bundle_path = Path(model_bundle_path)
    if not model_bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_bundle_path}")
    return joblib.load(model_bundle_path)


def list_available_activities(bundle: Dict[str, Any]) -> list[str]:
    return sorted(bundle["models"].keys())

#核心预测函数
def predict_met(
    bundle: Dict[str, Any],
    activity_type: str,
    age: float,
    sex: int | float | str,
    height_m: float | None = None,
    height_cm: float | None = None,
    weight_kg: float | None = None,
    bmi: float | None = None,
    ffm_alsallami: float | None = None,
) -> Dict[str, Any]:
    """
    Predict metabolic rate for a selected activity.

    You may provide either:两种输入方式：
      A) bmi and ffm_alsallami directly, or  A) 直接提供 bmi 和 ffm_alsallami
      B) height + weight, from which BMI and FFM-AlSallami will be calculated. B) 提供 height + weight，自动计算

    Returns predicted met and W/m². The approximate 95% prediction interval uses
    activity-specific grouped-CV RMSE and is not a formal confidence interval.
    """
    #验证活动类型
    if activity_type not in bundle["models"]:
        available = ", ".join(list_available_activities(bundle))
        raise ValueError(f"Unknown activity_type: {activity_type}. Available activities: {available}")
    #解析性别
    sex_num = parse_sex(sex)
    if height_m is None and height_cm is not None:
        height_m = float(height_cm) / 100.0
    #计算或验证 BMI
    if bmi is None:
        if height_m is None or weight_kg is None:
            raise ValueError("Either bmi must be provided, or height_m/height_cm and weight_kg must be provided.")
        bmi = calculate_bmi(float(height_m), float(weight_kg))
    #计算或验证 FFM
    if ffm_alsallami is None:
        if height_m is None or weight_kg is None:
            raise ValueError("Either ffm_alsallami must be provided, or height_m/height_cm and weight_kg must be provided.")
        ffm_alsallami = calculate_ffm_alsallami(float(age), sex_num, float(height_m), float(weight_kg))
    #构建特征向量（必须按 M5_FEATURES 顺序）
    x = pd.DataFrame([
        {
            "Age": float(age),
            "Sex": sex_num,
            "BMI": float(bmi),
            "FFM-AlSallami": float(ffm_alsallami),
        }
    ])[M5_FEATURES]
    #预测
    model_entry = bundle["models"][activity_type]
    model = model_entry["model"]
    metadata = model_entry["metadata"]
    pred_met = float(model.predict(x)[0])
    cv_rmse = float(metadata.get("cv_rmse", 0.0))
    #构建结果字典
    result = {
        "activity_type": activity_type,
        "predicted_met": pred_met,
        "predicted_W_m2": pred_met * MET_TO_W_M2,
        "age": float(age),
        "sex": sex_num,
        "sex_label": "Male" if sex_num == 0 else "Female",
        "bmi": float(bmi),
        "ffm_alsallami": float(ffm_alsallami),
        "approx_PI95_lower_met": pred_met - 1.96 * cv_rmse,
        "approx_PI95_upper_met": pred_met + 1.96 * cv_rmse,
        "cv_rmse_used_for_PI": cv_rmse,
        "note": "Approximate prediction interval uses activity-specific grouped-CV RMSE; it is not a formal confidence interval.",
    }
    if height_m is not None and weight_kg is not None:
        result["height_m"] = float(height_m)
        result["weight_kg"] = float(weight_kg)
        result["bsa_m2_dubois"] = calculate_bsa_dubois(float(height_m), float(weight_kg))
    return result


if __name__ == "__main__":
    bundle = load_model_bundle()
    print("Available activities:")
    for act in list_available_activities(bundle):
        print(" -", act)
    print("\nExample prediction:")
    print(predict_met(bundle, activity_type="Reclining", age=10, sex="male", height_cm=136, weight_kg=38))
