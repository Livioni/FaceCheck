# FaceCheck

FaceCheck 是一个基于 Vision Transformer (ViT) 的面部瘫痪二分类系统（affected / unaffected）。模型输入为多模态数据：
- RGB 人脸图像（BGR/RGB 均可，推理时会统一预处理到 224×224）
- 深度图（`.npy`，与图像同分辨率或可被 resize）
- 关键点向量（`landmark_vec`，默认 27 维；训练/推理可选择由 DynaFace 自动生成）

仓库同时内置两个上游子模块（用于端到端推理脚本）：
- `dynaface/`：DynaFace（MTCNN + SPIGA）用于推理 97 点 landmarks
- `smirk/`：SMIRK（3D 重建）用于从单张图片估计深度（需要额外依赖 + 预训练权重）

## 项目结构

```
FaceCheck/
├── facecheck/                 # 训练 / 推理 / API
│   ├── training/              # train.py / evaluate.py
│   ├── inference/             # predictor.py / preprocess.py
│   └── api/                   # FastAPI 服务
├── infer_pipeline.py          # 端到端 demo：输入单张图 -> crop -> SMIRK depth -> DynaFace landmarks -> FaceCheck 分类
├── dynaface/                  # 子模块（可独立使用）
├── smirk/                     # 子模块（可独立使用）
└── tests/                     # 单元测试（模型/预处理）
```

## 获取代码（含子模块）

如果你是首次拉取，推荐使用：

```bash
git clone --recurse-submodules <repo_url>
```

如果已经 clone 但未拉子模块：

```bash
git submodule update --init --recursive
```

## 安装

基础环境（用于 FaceCheck 训练/评估/推理/API）：

```bash
pip install -r requirements.txt
```

说明：
- `requirements.txt` 覆盖 FaceCheck 主流程（PyTorch / timm / OpenCV / scikit-learn / FastAPI 等）
- 若在推理阶段不提供 `landmark_vec`，`FaceCheckPredictor` 会尝试调用 DynaFace 自动生成关键点（需要子模块 `dynaface/` 可用）

### 端到端推理脚本的额外依赖（infer_pipeline.py）

`infer_pipeline.py` 还会用到 SMIRK + mediapipe + scikit-image + PyTorch3D 等依赖。建议在同一环境里额外安装：

```bash
pip install -r smirk/requirements.txt
```

并按 SMIRK 的说明安装 PyTorch3D（不同 CUDA / Python 版本需要对应 wheel；参考 [smirk/readme.md](file:///home/shengyu/Documents/PENG/github/FaceCheck/smirk/readme.md)）。

另外需要下载 SMIRK 的预训练权重（推荐在 `smirk/` 下执行）：

```bash
cd smirk
bash quick_install.sh
```

默认会得到类似：`smirk/pretrained_models/SMIRK_em1.pt`

## 数据集结构（训练/评估）

FaceCheck 默认支持 palsynet 风格目录结构：

```
datasets/palsynet-5230/
├── affected/
│   ├── 1/
│   │   ├── cropped_img/        # 面部视频帧（png/jpg）
│   │   ├── depth/              # 对应深度文件（与图片同名 .npy）
│   │   └── landmarks.json      # 可选：关键点/特征（若不存在可用 DynaFace 自动生成）
│   └── ...
└── unaffected/
    ├── 1/
    └── ...
```

划分方式：按 subject（人的目录名）划分，默认每个类别 20% 的不同 subject 进入测试集（其余再切 val）。

## 训练

训练输出目录默认 `outputs/facecheck/`，会生成 `best.pt`（包含模型权重 + 训练配置 + 预处理统计量：depth min/max 与 landmark mean/std）。

```bash
python -m facecheck.training.train \
  --dataset_root /abs/path/to/datasets/palsynet-5230 \
  --output_dir outputs/facecheck \
  --epochs 100 \
  --batch_size 32
```

当数据集中没有 `landmarks.json`，你可以启用 DynaFace 自动生成关键点（注意会强制将 `num_workers` 置 0，以避免多进程初始化模型带来的问题）：

```bash
python -m facecheck.training.train \
  --dataset_root /abs/path/to/datasets/palsynet-5230 \
  --use_dynaface_landmarks
```

## 评估

```bash
python -m facecheck.training.evaluate \
  --ckpt outputs/facecheck/best.pt \
  --dataset_root /abs/path/to/datasets/palsynet-5230
```

同样支持在缺少 `landmarks.json` 时启用 DynaFace：

```bash
python -m facecheck.training.evaluate \
  --ckpt outputs/facecheck/best.pt \
  --dataset_root /abs/path/to/datasets/palsynet-5230 \
  --use_dynaface_landmarks
```

## 推理（Python）

`FaceCheckPredictor` 会优先使用你显式传入的 `landmark_vec`；如果不提供，会从输入图像调用 DynaFace 自动推理 landmark 并映射到模型所需维度。

```python
import numpy as np
import cv2
from facecheck.inference.predictor import FaceCheckPredictor

pred = FaceCheckPredictor.load("outputs/facecheck/best.pt", device="cuda")
img = cv2.imread("/path/to/cropped_img/0001.jpg")
depth = np.load("/path/to/depth/0001.npy")
out = pred.predict_bgr_depth(img, depth)
print(out.prob_affected, out.label)
```

也可以直接用路径：

```python
out = pred.predict_paths("/path/to/0001.jpg", "/path/to/0001.npy")
```

## 端到端推理（infer_pipeline.py）

该脚本将单张输入图片串起来跑完整流程：
1) mediapipe 检测关键点并 crop 到 224×224（对齐 SMIRK demo 的 crop 策略）  
2) SMIRK 推理深度与 mask，并导出可视化  
3) DynaFace 推理 97 点 landmarks（并导出 overlay）  
4) FaceCheck 推理分类结果  
5) 导出最终对比图（原图 / crop / depth_vis / overlay+label）

示例：

```bash
python infer_pipeline.py \
  --input /abs/path/to/image.png \
  --out_dir output_infer \
  --smirk_ckpt smirk/pretrained_models/SMIRK_em1.pt \
  --facecheck_ckpt outputs/facecheck/best.pt \
  --device cuda
```

输出文件（在 `--out_dir` 下）：
- `00_input.png`：原图备份
- `01_cropped_224.png`：crop 后 224×224
- `01_crop.json`：crop 的几何变换参数与 mediapipe 关键点
- `02_depth.npy`：SMIRK 深度（float32）
- `02_depth_mask.npy`：SMIRK mask（uint8）
- `02_depth_vis.png`：深度可视化（colormap）
- `02_smirk_vertices.npy`：FLAME vertices（调试/可视化用）
- `02_smirk_raw.json`：SMIRK 输出的 key 列表与 vertices shape（轻量摘要）
- `03_dynaface_landmarks.json`：DynaFace landmarks（xy）
- `03_dynaface_overlay.png`：关键点 overlay（若检测到人脸）
- `04_landmark_vec.npy`：提供给 FaceCheck 的 landmark_vec（float32）
- `04_facecheck_result.json`：分类结果（prob_affected / label）
- `final.png`：汇总图（带 label 文本）

## 推理服务（FastAPI）

通过环境变量指定 checkpoint 路径后启动服务：

```bash
export FACECHECK_CKPT=/abs/path/to/outputs/facecheck/best.pt
uvicorn facecheck.api.app:app --host 0.0.0.0 --port 8000
```

端点：
- `GET /health`：健康检查
- `POST /predict`：表单上传 `image`（jpg/png）与 `depth`（.npy）；可选 `landmark`（JSON 数组）

## 测试

仓库包含基础单元测试（模型前向与预处理 shape）。在安装 `pytest` 后运行：

```bash
pytest -q
```

## 致谢

- DynaFace：landmark 推理（MTCNN + SPIGA）
- SMIRK：单图 3D 重建/深度估计
- timm：ViT 主干实现
- PyTorch：训练与推理框架
