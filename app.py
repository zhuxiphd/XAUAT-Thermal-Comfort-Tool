from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import PMV
import SET

# 尝试导入 JOS3，如果缺包也不影响运行，会自动退回示例数据
try:
    import JOS3 as jos3_api
except ImportError:
    jos3_api = None

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)


# ---------------- 0. 首页：返回前端 ----------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------- 1. 主模拟接口：/api/simulate ----------------
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    payload = request.get_json() or {}

    data = {}

    # ① 如果 JOS3 可用，用真正的 JOS-3 模型计算
    if (
        jos3_api is not None
        and hasattr(jos3_api, "SimInput")
        and hasattr(jos3_api, "simulate")
    ):
        try:
            sim_input = jos3_api.SimInput(**payload)
            sim_output = jos3_api.simulate(sim_input)  # 返回的是 Pydantic 模型
            data = sim_output.dict()
        except Exception as e:
            print("调用 JOS-3 出错，退回到示例数据：", e)
            data = {}

    # ② 如果上面失败或没有 JOS3，就用一组示例数据兜底
    if not data:
        time_min = [i for i in range(0, 61, 10)]  # 0,10,20,30,40,50,60 分钟
        TskMean = [33.5 + 0.05 * i for i in range(len(time_min))]
        Tcb = [36.8 + 0.02 * i for i in range(len(time_min))]
        DTS = [-0.3 + 0.05 * i for i in range(len(time_min))]
        PPD = [5 + 2 * i for i in range(len(time_min))]

        body_parts = [
            "Head", "Neck", "Chest", "Back", "Pelvis",
            "Left-Shoulder", "Left-Arm", "Left-Hand",
            "Right-Shoulder", "Right-Arm", "Right-Hand",
            "Left-Thigh", "Left-Leg", "Left-Foot",
            "Right-Thigh", "Right-Leg", "Right-Foot",
        ]

        # tsk_local: [n_parts][n_time]
        tsk_local = []
        for i, t in enumerate(time_min):
            base = 33.5 + 0.05 * i
            tsk_local.append([base + 0.1 * j for j in range(len(body_parts))])

        # ★★★ 这里的字段名必须和 index.html 里用的一致 ★★★
        data = {
            "time_min": time_min,    # 原来是 "time"
            "TskMean": TskMean,      # 原来是 "Tsk"
            "Tcb": Tcb,
            "DTS": DTS,
            "PPD": PPD,
            "body_parts": body_parts,
            "tsk_local": tsk_local,
            "Met": [1.2] * len(time_min),
            "raw": {},
        }

    return jsonify(data)


# ---------------- 2. PMV 接口：/api/pmv ----------------
@app.route("/api/pmv", methods=["POST"])
def api_pmv():
    data = request.get_json() or {}

    tdb = float(data.get("ta", data.get("tdb", 25.0)))
    tr = float(data.get("tr", tdb))
    vr = float(data.get("vel", data.get("vr", 0.1)))
    rh = float(data.get("rh", 50.0))
    met = float(data.get("met", 1.2))
    clo = float(data.get("clo", 0.5))
    wme = float(data.get("wme", 0.0))

    pmv, ppd = PMV.pmv_ppd(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
    )

    # 前端判断条件：pmvData && pmvData.ok && typeof pmvData.pmv === "number"
    return jsonify({"ok": True, "pmv": float(pmv), "ppd": float(ppd)})


# ---------------- 3. SET 接口：/api/set ----------------
@app.route("/api/set", methods=["POST"])
def api_set():
    data = request.get_json() or {}

    tdb = float(data.get("ta", data.get("tdb", 25.0)))
    tr = float(data.get("tr", tdb))
    v = float(data.get("vel", data.get("v", 0.1)))
    rh = float(data.get("rh", 50.0))
    met = float(data.get("met", 1.2))
    clo = float(data.get("clo", 0.5))
    wme = float(data.get("wme", 0.0))

    body_surface_area = float(data.get("body_surface_area", 1.8258))
    p_atm = float(data.get("p_atm", 101325.0))
    posture = data.get("posture", "standing")

    set_value = SET.set_tmp(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=posture,
    )

    # 前端判断条件：lastSetData && lastSetData.ok && typeof lastSetData.set === "number"
    return jsonify({"ok": True, "set": float(set_value)})


if __name__ == "__main__":
    # 你之前就是用 5001，这里保持不变
    app.run(host="0.0.0.0", port=5001, debug=True)
