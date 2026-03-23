# AniPose
# 9.1.Link-to-Anime Mixamo 数据集文件结构

## 目录结构

数据集以"模型"为根目录（如 `68 model/`），每个模型下包含多个视角子目录。

```
<model_name>/
├── 01 visual_view/
│   ├── 01_1 Rendering/              # 渲染图（72帧，PNG/JPG）
│   ├── 01_4 Optical_flow_exr/       # 光流数据
│   ├── 01_5 line/                   # 线稿
│   ├── 01_5 Occ_mask_bac/           # 遮挡 mask（背景）
│   ├── 01_5 Occ_mask_for/           # 遮挡 mask（前景）
│   ├── 01_7 bone_coordinates_cam.json
│   ├── 01_8 bone_coordinates_image.json
│   ├── 01_9 bone_coordinates_pixels.json
│   └── 01_10 Bounding_Box.json
├── 02 visual_view/
│   └── ...（结构同上，前缀为 02）
├── 03 visual_view/
├── 04 visual_view/
└── 05 visual_view/
```

目录命名规则：`{视角编号(两位)} visual_view`，文件命名规则：`{视角编号}_{类型编号} {描述}`。

---

## 关键文件说明

### 骨骼坐标 JSON（三种坐标系）

每个 JSON 的 key 为帧编号字符串（`"1"`、`"2"` … `"72"`），value 为骨骼名称到坐标的映射。

| 文件                               | 坐标系           | 用途                                 |
| ---------------------------------- | ---------------- | ------------------------------------ |
| `*_7 bone_coordinates_cam.json`    | 相机坐标系（3D） | 相机空间三维位置                     |
| `*_8 bone_coordinates_image.json`  | 归一化图像坐标系 | 投影后的归一化坐标                   |
| `*_9 bone_coordinates_pixels.json` | 像素坐标系       | **主要使用**，直接对应渲染图像素位置 |

骨骼命名遵循 Mixamo 标准，前缀为 `mixamorig:`，例如：

```
mixamorig:Hips, mixamorig:Spine, mixamorig:Neck, mixamorig:Head,
mixamorig:LeftArm, mixamorig:LeftForeArm, mixamorig:LeftHand,
mixamorig:LeftUpLeg, mixamorig:LeftLeg, mixamorig:LeftFoot ...
```

### Bounding Box JSON（`*_10 Bounding_Box.json`）

Key 格式为 `"{帧号四位}_{相机编号}"`，value 为 COCO 格式 `[x, y, w, h]`。

```json
{
  "0001_1": [810, 389, 707, 1051],
  "0002_1": [815, 392, 701, 1048],
  ...
}
```

**注意**：相机编号与视角编号一致（01视角→`_1`，02视角→`_2`，以此类推），读取时需按视角动态匹配后缀，不能硬编码 `_1`。

---

## 标注转换说明

### Mixamo → COCO-17 骨骼映射

| COCO-17 关节       | Mixamo 骨骼                         |
| ------------------ | ----------------------------------- |
| left_shoulder      | mixamorig:LeftArm                   |
| right_shoulder     | mixamorig:RightArm                  |
| left_elbow         | mixamorig:LeftForeArm               |
| right_elbow        | mixamorig:RightForeArm              |
| left_wrist         | mixamorig:LeftHand                  |
| right_wrist        | mixamorig:RightHand                 |
| left_hip           | mixamorig:LeftUpLeg                 |
| right_hip          | mixamorig:RightUpLeg                |
| left_knee          | mixamorig:LeftLeg                   |
| right_knee         | mixamorig:RightLeg                  |
| left_ankle         | mixamorig:LeftFoot                  |
| right_ankle        | mixamorig:RightFoot                 |
| neck               | mixamorig:Neck                      |
| nose               | 由 RTMLib 补全，或取 Head+Neck 中点 |
| left/right eye/ear | 由 RTMLib wholebody 补全            |

### Visibility 规则

- `2`：点坐标落在图像像素范围内（可见）
- `0`：点坐标超出图像边界（画面外，不参与训练 loss）

面部5点（nose/eye/ear）仅在 `mixamorig:Head` 与 `mixamorig:HeadTop_End` **同时**位于画面内时，才调用 RTMLib 进行补全推理；否则跳过，对应点设为 `visibility=0`。