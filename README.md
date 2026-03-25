# AnimePose2D

AnimePose2D是一个用于自动化标注2D动画（主要是日本动画）中人物角色的各种形态学特征的工具。本仓库试图将2D动画中的人物这一运动主体进行特别标注，这将引入 pose、skeleton、角色掩码这些角色中心模态。

在自动中割等研究中，由于日本动画中的人物镜头通常涉及角色层与背景层的分离（一个典型证据便是律表中A~F层的设置）、“有限动画”思想下对关键动作帧与时序的重视，以及的 layout 中对角色行为的显式约束，人物镜头在一定程度上适合被视为一个结构化角色运动问题。

本仓库包含：

- TODO：一个在rtmpose基础上针对2D日本动画角色进行微调的姿态估计模型，以及一个对应的演示示例代码。（所用数据集基于多部日本动画和针对本任务增强后的link-to-anime数据集）
- 一个训练脚本，用于微调rtmlib模型并展示基本微调效果。基于MMPose，需要在本仓库根目录cloneMMPose。位于`ft.ipynb`
- 一个用于扩张mixmo特征点到COCO-17标准（去除多余特征点，通过自动化标注补充头部五点：双眼、双耳、鼻子）并展示效果的脚本，依赖rtmlib。本仓库目录结构暂时依托link-to-anime的实现。位于`mixamo_pose_preview&convert.ipynb`。
- TODO：一个基于sam3的动画人物掩码提取工具，支持视频和图片。
- 一个数据集中姿态骨骼可视化工具。位于`mixamo_pose_preview&convert.ipynb`；`ft.ipynb`中的相应单元格也可以让你预览转换为可训练格式的姿态特征点。



# 1.目录结构

## 1.1.link-to-anime数据集中一个人物模型对应的文件结构

该数据集由多个人物模型构成，每个人物模型都包含多张不同机位下连续运动的渲染图和对应的mixamo姿态特征点（包括bbox）、光流、线稿构成。

数据集由多个人物模型构成，每个人物模型都包含多张不同机位下连续运动的渲染图和对应的mixamo姿态特征点（包括bbox）、光流、线稿构成。

```
 tree --filelimit 20
.
├── 68 model
│   ├── 01 visual_view
│   │   ├── 01_10 Bounding_Box.json  # bbox
│   │   ├── 01_1 Rendering  [72 entries exceeds filelimit, not opening dir]  # 渲染图
│   │   ├── 01_4 Optical_flow_exr  [72 entries exceeds filelimit, not opening dir]  # 光流
│   │   ├── 01_5 line  [72 entries exceeds filelimit, not opening dir]  # 线稿
│   │   ├── 01_5 Occ_mask_bac  [71 entries exceeds filelimit, not opening dir]
│   │   ├── 01_5 Occ_mask_for  [71 entries exceeds filelimit, not opening dir]
│   │   ├── 01_7 bone_coordinates_cam.json  
│   │   ├── 01_8 bone_coordinates_image.json
│   │   └── 01_9 bone_coordinates_pixels.json # 789都是不同策略下得到的特征点，默认采用9
│   ├── 02 visual_view
|		... 

```

进行对特征点的扩张和两次转换以后，会变成：

```
 tree --filelimit 20
.
├── 68 model
│   ├── 01_0001_coco17_strict1.json # 测试样例，可删除
│   ├── 01 visual_view
│   │   ├── 01_10 Bounding_Box.json  # bbox
│   │   ├── 01_1 Rendering  [72 entries exceeds filelimit, not opening dir]  # 渲染图
│   │   ├── 01_4 Optical_flow_exr  [72 entries exceeds filelimit, not opening dir]  # 光流
│   │   ├── 01_5 line  [72 entries exceeds filelimit, not opening dir]  # 线稿
│   │   ├── 01_5 Occ_mask_bac  [71 entries exceeds filelimit, not opening dir]
│   │   ├── 01_5 Occ_mask_for  [71 entries exceeds filelimit, not opening dir]
│   │   ├── 01_7 bone_coordinates_cam.json  
│   │   ├── 01_8 bone_coordinates_image.json
│   │   └── 01_9 bone_coordinates_pixels.json # 789都是不同策略下得到的特征点，默认采用9
│   ├── 02 visual_view
|		... 
│   └── exported_coco17_strict1  # 运行mixamo_pose_preview&convert.ipynb进行扩张后生成的新的姿态json，但每个view下每张图都生成独立的文件。
│       ├── 01 visual_view  [72 entries exceeds filelimit, not opening dir]  
│       ├── 02 visual_view  [72 entries exceeds filelimit, not opening dir]
│       ├── 03 visual_view  [72 entries exceeds filelimit, not opening dir]
│       ├── 04 visual_view  [72 entries exceeds filelimit, not opening dir]
│       └── 05 visual_view  [72 entries exceeds filelimit, not opening dir]
├── anime_coco_train.json   # 运行ft.ipynb的转换为训练数据单元格后生成训练集
├── anime_coco_val.json # 运行ft.ipynb的转换为训练数据单元格后生成验证集

```

`/exported_coco17_strict1`是根据`/68 model`中各个view下的`bone_coordinates_pixels.json`（每个该文件都对应该view下共72帧的mixamo规范特征点）。

mixamo规范缺少部分特征点。使用rtmlib补全面部五个特征点后，根据一些数据集的特定标准（这里默认使用coco17）导出的新的角色姿态特征点，放入`/exported_coco17_strict1`目录。其每个`/* visual_view`内容编辑自`/68 model`中的同名子目录。

> 不过和`bone_coordinates_pixels.json`不同的是，这里为了方便微调等工作取用，分成了72个子文件，而不是置于同一个json中。



## 1.2.AnimePose2D生成数据集文件结构

### 1.2.1.将mixamo扩展为coco-17

对mixamo的转换、点扩充与可视化notebook：`mixamo_pose_preview&convert.ipynb`

在转换为可训练数据之前，首先会把原本的mixamo数据扩张为coco-17，这一中间结果也通过json保存。在对mixamo的转换、点扩充与可视化notebook中，通过函数`def batch_export_all_views_frames():`：

- 接受一个关键参数output

  dir -从`output_dir`扫描所有视角目录，并把每个视角目录中的姿态特征点输出在`output_dir`下，默认名称为`'exported

  {standard}_strict{int(strict)}'`的子目录下。

  - 对于子目录的默认名称，默认状态下{standard}=coco17； strict=True

- 工作目录`ROOT`默认为当前人物模型目录，如`68 model` *TODO：把ROOT解耦成输入参数以实现批处理*

输出的新目录结构(位于每一个人物模型目录下)如下：

```
exported_coco17_strict1/
  ├── 01 visual_view/
  │   ├── frame_0001_coco17_strict1.json
  │   └── ...
  ├── 02 visual_view/
  └── ...
```

单个JSON文件的数据结构示例如下

```json
// 68 mdoel/01 visual_view/frame_0001_coco17_strict1.json
{
  "view": "01 visual_view",
  "frame_id": 1,
  "standard": "coco17",
  "strict": true,
  "bbox": [
    810,
    389,
    707,
    1051
  ],
  "face5_info": "inferred",
  "num_keypoints": 13,
  "keypoints": {
    "nose": {
      "x": 1174.6085067325168,
      "y": 662.7132822672527,
      "z": 0.0,
      "visibility": 2
    },
	 //...
    "right_ankle": {
      "x": 962.0318434289209,
      "y": 2055.0497568344476,
      "z": 0.0,
      "visibility": 0
    }
  },
  "keypoints_flat": [
    1174.6085067325168,
    662.7132822672527,
    2,
	//...
    962.0318434289209,
    2055.0497568344476,
    0
  ]
}
```

### 1.2.2.coco格式可训练JSON

通过调用coco_converter.py对扩张、转换为coco17标准的特征点json进一步转换为可训练格式。详细讲：

- 输入：`68 model/exported_coco17_strict1/`下各个视角目录中的json文件

- 输出一个训练集和一个验证集，位于项目根目录（和`68 model/`同级）

  - `anime_coco_train.json`
  - `anime_coco_val.json`

  

> TODO: 适配link-to-anime数据集的完整目录结构，在现在的基础上还要多两层。



输出json分为四部分，每部分都包含*每一条*数据对应字段下的信息。

1. info
2. images-该条数据的id、文件目录、宽高
3. annotations-该条数据的详细信息：主要是bbox和特征点坐标
4. categories

#### annotations单条数据示例

COCO-17pts标准：

```json
"annotations": [
    {
      "id": 172,
      "image_id": 172,
      "category_id": 1,
      "bbox": [
        1003.4527335399849,
        420.5579350111529,
        517.0430776418385,
        2044.375385504148
      ],
      "area": 1057030.1411762848,
      "keypoints": [
        1280.5279757181802,
        579.7132288614907,
        2,
        ...
        1438.085060226139,
        1621.0762272778913,
        2
      ],
      "num_keypoints": 17,
      "iscrowd": 0,
      "segmentation": []
    },
    {
      "id": 178,
      ...
```

#### images单条数据示例

```json
    {
      "id": 232,
      "file_name": "04 visual_view/04_1 Rendering/0016_4.png",
      "height": 1440,
      "width": 2560
    },
    ...
    
```



# TODO

- requirements

  
