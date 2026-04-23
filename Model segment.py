import json
from pathlib import Path
import numpy as np
import trimesh

ABC_GLTF = Path("/Users/zhuxi/Downloads/博1任务/PMV:SET:JOS3模型UI开发/jos3_webapp2/static/models/humanbody.gltf")
STL_PATH = Path("static/models/1.stl")
OUT_PATH = Path("static/models/vertex_segment_map.json")

SEGMENT_NAMES = [
    "Head","Neck","Chest","Back","Pelvis",
    "Left-Shoulder","Left-Arm","Left-Hand",
    "Right-Shoulder","Right-Arm","Right-Hand",
    "Left-Thigh","Left-Leg","Left-Foot",
    "Right-Thigh","Right-Leg","Right-Foot"
]
# 如果 glTF 枚举名不同，可在此表里自定义（例如 UpperArm = Left-Shoulder + Left-Arm）

gltf = trimesh.load(ABC_GLTF, force='scene')

# 逐个子网格收集包围盒（glTF 中命名为 GLTF_n）
segment_boxes = []
for part_idx, name in enumerate(SEGMENT_NAMES, start=1):
    geom_key = f"GLTF_{part_idx}"
    geom = gltf.geometry.get(geom_key)
    if geom is None:
        raise SystemExit(f"找不到第 {part_idx} 个子网格（{name}）：尝试的键 {geom_key}")
    bbox_min = geom.bounds[0]
    bbox_max = geom.bounds[1]
    center = (bbox_min + bbox_max) / 2.0
    segment_boxes.append({
        "name": name,
        "index": part_idx - 1,
        "min": bbox_min.tolist(),
        "max": bbox_max.tolist(),
        "center": center.tolist()
    })

stl = trimesh.load(STL_PATH, process=False)
vertices = stl.vertices

def point_to_box_distance(p, box):
    min_ = np.array(box["min"])
    max_ = np.array(box["max"])
    inside = np.all((p >= min_) & (p <= max_))
    if inside:
        return 0.0
    # 计算到包围盒的欧氏距离（点到 AABB）
    delta = np.maximum(np.maximum(min_ - p, 0.0), p - max_)
    return np.linalg.norm(delta)

vertex_to_segment = []
for i, v in enumerate(vertices):
    best = min(segment_boxes, key=lambda box: point_to_box_distance(v, box))
    vertex_to_segment.append(best["index"])

OUT_PATH.write_text(json.dumps({
    "segments": segment_boxes,
    "vertex_map": vertex_to_segment
}, indent=2))
print(f"生成 {OUT_PATH}，共 {len(vertices)} 个顶点。")
