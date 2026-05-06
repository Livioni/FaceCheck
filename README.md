# FaceCheck

FaceCheck 是一个基于 **DINOv3 ViT** 主干的面部瘫痪二分类系统(`affected` / `unaffected`)。模型为多模态融合架构,同时吃 **RGB + 单通道深度 + 面部测量特征向量**:

- RGB 人脸图像(BGR/RGB 都行,推理统一 resize 到 224×224)
- 深度图(`.npy`,float32,会被 resize 到 224×224 + min-max 归一化)
- landmark/measure 特征向量(默认 30 维,可由 DynaFace 自动生成,也可显式传入)

仓库内置两个上游子模块以支持端到端推理:
- `dynaface/`:DynaFace(MTCNN + SPIGA),输出 97 点 landmarks 与一组面部测量值
- `smirk/`:SMIRK(单图 3D 重建),用于在没有真实深度的图片上估计深度

## 项目结构

```
FaceCheck/
├── facecheck/                 # 训练 / 推理 / API
│   ├── data/                  # dataset.py / transforms.py / utils.py
│   ├── models/                # vit_facecheck.py / layers.py
│   ├── training/              # train.py / evaluate.py / config.py
│   ├── inference/             # predictor.py / preprocess.py
│   └── api/                   # FastAPI 服务(/health, /predict, /quicktest, /heic, /pipeline)
├── infer_pipeline.py          # 端到端 demo:单张图 → crop → 深度 → landmarks → 分类
├── dynaface/                  # 子模块:facial landmark 检测
├── smirk/                     # 子模块:单图深度估计
└── tests/                     # 单元测试(模型/预处理)
```

## 模型架构

`facecheck/models/vit_facecheck.py` 中的 `ViTFaceCheck` 是一个改造后的 timm ViT,以 DINOv3 base/large 等带 RoPE + register token 的 backbone 为主:

```
            ┌─────────────────────────────┐
RGB 224×224 │                             │
            │   patch_embed (4-ch)        │   ← in_chans=4: 把单通道 depth 接到 RGB 后面
Depth 224×224│   16×16 patches → 196 tokens│
            └─────────────┬───────────────┘
                          │
landmark_vec (30) ──► LandmarkMLP (Linear→GELU→Linear) ──► 1 token
                          │
        prefix tokens [cls] + [reg×4] + [landmark]   ← register tokens 来自 DINOv3
                          │
                  ┌───────▼───────┐
                  │  ViT blocks   │  ← RoPE 只作用于 patch tokens,prefix(含 landmark)不旋转
                  │  ×L (12/24/…) │
                  └───────┬───────┘
                          │
                       norm + cls token  ──► BinaryHead(Linear) ──► [logit_unaff, logit_aff]
```

关键设计点:
- **patch_embed 的 in_chans = 4**:在 timm 创建模型时把 RGB 的 3 通道 conv 扩展成 4 通道。加载预训练 RGB 权重时,`_adapt_patch_embed_weight` 会把第 4 个通道初始化为前 3 通道的均值,从而尽量保留 ImageNet/DINOv3 预训练分布。
- **DINOv3 检测**:构造模型时检查 `vit.rope` 是否存在。若是 RoPE backbone:
  - 把每个 EvaBlock 的 `attn.num_prefix_tokens` 加 1(为 landmark token 留位),保证 RoPE **只作用于 patch tokens**,不会污染 prefix。
  - 每层显式传 `blk(x, rope=rot_pos_embed)`。普通 ViT 走另一条 forward 路径,landmark token 作为序列尾的额外 token,pos_embed 末尾补一个零。
- **landmark token**:30 维测量向量(brow.d / fai / oce.l/r / pd / yaw 等 DynaFace 测量值)经 2 层 MLP 升到 embed_dim,作为额外的 prefix token 与 cls/reg 一起拼到序列前。它给图像 token 提供面部几何先验,但不参与 RoPE。
- **head**:`BinaryHead = Linear(embed_dim → 2)`,直接对 cls token 输出做交叉熵分类。

预训练权重加载顺序(`facecheck/training/train.py`):

1. 用户给 `--pretrained_ckpt path` → 通过 `model.load_pretrained()` 加载并自动适配 patch_embed 通道扩展。
2. 否则尝试 `model.load_timm_pretrained()` → 直接用 timm 拉对应 backbone 的官方权重(如 `vit_base_patch16_dinov3.lvd1689m`)。
3. 失败则保持随机初始化,日志中写 `pretrained_load_skipped`。

`landmark` MLP 与 `head` 永远是从零训的(预训练权重不会包含这两部分)。

### Checkpoint 内容

`best.pt` 是单个 dict,可直接 `torch.load`:

| key             | 说明                                                |
| --------------- | --------------------------------------------------- |
| `model`         | 整个 `ViTFaceCheck` 的 `state_dict`                 |
| `config`        | 训练时的 `TrainConfig.to_dict()`(含 backbone / landmark_dim 等) |
| `depth_minmax`  | `{"vmin", "vmax"}`,推理时用同一对值做 min-max 归一化 |
| `landmark_mean` | landmark 向量的训练集均值(list[float])            |
| `landmark_std`  | landmark 向量的训练集标准差(list[float])          |

`FaceCheckPredictor.load()` 根据 `config` 复原 backbone 与 landmark_dim,根据 `depth_minmax`/`landmark_*` 复原预处理统计量,所以**部署只需要这一个文件**,不需要单独发布配置。

## 获取代码(含子模块)

首次拉取:

```bash
git clone --recurse-submodules <repo_url>
```

已经 clone 但没拉子模块:

```bash
git submodule update --init --recursive
```

## 安装

```bash
pip install -r requirements.txt
```

涵盖:PyTorch / timm(≥1.0.20,自带 DINOv3) / OpenCV / scikit-learn / FastAPI / facenet_pytorch / pillow_heif 等。

不显式传 `landmark_vec` 时,Predictor 会调用 DynaFace 自动生成关键点 → 需要 `dynaface/` 子模块以及它依赖的 facenet_pytorch / SPIGA 权重(首次调用时由 `dynaface.models.download_models()` 自动下载,可用 `DYNAFACE_MODEL_DIR` 指定缓存目录)。

### `infer_pipeline.py` / `/pipeline` 的额外依赖

需要 SMIRK + mediapipe + scikit-image + PyTorch3D:

```bash
pip install -r smirk/requirements.txt
```

PyTorch3D 需要按 SMIRK 的指南装对应 CUDA / Python 版本的 wheel,参考 [smirk/readme.md](smirk/readme.md)。

下载 SMIRK 权重:

```bash
cd smirk
bash quick_install.sh
```

得到 `smirk/pretrained_models/SMIRK_em1.pt`。`/heic` 端点(iPhone HEIC 内嵌深度)不依赖 SMIRK。

## 数据集结构(训练/评估)

palsynet 风格:

```
datasets/palsynet-5230/
├── affected/
│   ├── 1/
│   │   ├── cropped_img/        # 224×224 面部帧(png/jpg)
│   │   ├── depth/              # 同名 .npy 深度
│   │   ├── landmarks.json      # 可选:97 点 landmarks(没有就用 measures.csv 或 DynaFace 兜底)
│   │   └── 1_measures.csv      # 优先来源:30 维测量值,列名见 brow.d/fai/oce.l 等
│   └── ...
└── unaffected/
    └── ...
```

划分按 **subject 级**,默认 20% subjects → test,10% → val,其余训练。这样保证验证/测试集的人在训练里完全没出现过。

可选辅助数据集 `Faces_Dataset/`(只用作 label=0 的负样本扩充,默认不启用):

```
Faces_Dataset/
├── cropped_images/             # 单张 jpg/png
├── depth_results/              # 同名 .npy
└── csv/human_faces_measures.csv  # 列名同 palsynet 的 measures.csv
```

## 训练

**默认配置已经按 DINOv3 fine-tune 调好了**(详见 `facecheck/training/config.py`):

- backbone: `vit_base_patch16_dinov3`(timm 自动加载预训练)
- 学习率:head/landmark MLP = 1e-4,backbone = 1e-5(`backbone_lr_mult=0.1`)
- AdamW + weight_decay 0.1(bias / norm / cls_token / reg_token / pos_embed / gamma 不施加)
- 3 epoch 线性 warmup → cosine 衰减到 `min_lr_ratio=0.01`
- `CrossEntropyLoss(label_smoothing=0.1)` + 自动按 inverse-frequency 计算的类权重
- gradient clip max_norm=1.0
- dropout=0.1
- 数据增强:hflip / 旋转 / color jitter / depth jitter,seed 完全随机(每 epoch 重新采样)

最简单的用法:

```bash
python -m facecheck.training.train --output_dir outputs/facecheck_version2
```

常用 override:

```bash
python -m facecheck.training.train \
  --output_dir outputs/facecheck_version2 \
  --backbone vit_base_patch16_dinov3 \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --backbone_lr_mult 0.1 \
  --warmup_epochs 3 \
  --label_smoothing 0.1
```

掺入 Faces_Dataset 作为额外 label=0 样本(默认关闭,因为会引入类不平衡):

```bash
python -m facecheck.training.train \
  --human_faces_root datasets/Faces_Dataset \
  --human_faces_max_train 1000
```

数据集里没有 `landmarks.json` 也没有 `*_measures.csv` 时,可让 DynaFace 兜底(会强制 `num_workers=0`):

```bash
python -m facecheck.training.train --use_dynaface_landmarks
```

输出目录会得到:`best.pt` / `config.json` / `train_log.jsonl` / `metrics.csv` / `curves.png`。`train_log.jsonl` 每个 epoch 有 `train_loss / val_loss / val_{acc,precision,recall,f1,auc} / lr_max / lr_min`,方便排查训练健康度。

## 评估

```bash
python -m facecheck.training.evaluate \
  --ckpt outputs/facecheck_version2/best.pt \
  --dataset_root datasets/palsynet-5230
```

加 `--use_dynaface_landmarks` 在没有 measures.csv / landmarks.json 时启用 DynaFace。

## 推理 Pipeline 详解

### Stage 0:输入

支持的输入路径:
- 已经 crop 好的 224×224 面部 + 对应深度 → 直接走 `FaceCheckPredictor`(最快、最准)
- 任意 RGB 单图 → 需要先走 SMIRK 估计深度,适合普通 photo
- iPhone HEIC(内嵌 depth + portrait matte)→ 直接读取嵌入深度,不依赖 SMIRK

### Stage 1:Crop & Align(`infer_pipeline.crop_like_smirk_demo`)

- 用 mediapipe 检测人脸 landmarks,以 face bbox 中心 + scale=1.4 扩展
- warp 到 224×224 正方形,记录正逆变换矩阵
- 输出 `cropped_bgr_no_resize`(原始分辨率裁切框)与 `224×224` resize 版本

### Stage 2:Depth

- **`/pipeline` 与 CLI**:SMIRK 单图重建 → 渲染深度图 + face mask,导出 `02_depth.npy`、`02_depth_mask.npy`、`02_depth_vis.png`
- **`/heic`**:从 HEIC 文件读 `depth` aux image(以及 `portraiteffectsmatte` 作为面部 mask),不需要 SMIRK 权重
- 拿不到深度时,模型会无法走 4 通道融合 → 这两个端点会 422 报错

### Stage 3:Landmarks(`facecheck/inference/predictor.py::_dynaface_landmark_vec`)

DynaFace 的 `AnalyzeFace` + `AnalyzeLandmarks()` 输出 97 点 (x, y),按图像宽高归一化到 [0, 1] 后 flatten,并截取/补 0 到 `landmark_dim=30`。也可以由调用方直接传 30 维向量(例如 palsynet 的 `*_measures.csv` 测量值,精度更高,因为模型就是按 measures 训练的)。

### Stage 4:Preprocess(`facecheck/inference/preprocess.py::FaceCheckPreprocessor`)

- BGR → RGB,resize 224×224,/255 → float32 [0, 1]
- depth resize 224×224(NEAREST),用 `depth_minmax` 做 min-max 归一化
- RGB + depth concat 成 4 通道 tensor
- landmark 向量做 (x − mean) / std(mean、std 来自训练集,存在 ckpt 里)

### Stage 5:推理

`ViTFaceCheck` forward → softmax → `prob_affected ∈ [0, 1]`,阈值 0.5 决定 label。

```python
import numpy as np, cv2
from facecheck.inference.predictor import FaceCheckPredictor

pred = FaceCheckPredictor.load("outputs/facecheck_version2/best.pt", device="cuda")
img = cv2.imread("/path/to/cropped_img/0001.jpg")
depth = np.load("/path/to/depth/0001.npy")
out = pred.predict_bgr_depth(img, depth)            # 不传 landmark,自动跑 DynaFace
print(out.prob_affected, out.label)

# 或显式传 30 维 measure 向量(顺序需与训练 csv 列序一致)
out = pred.predict_bgr_depth(img, depth, landmark_vec=measure_vec_30)

# 也支持路径
out = pred.predict_paths("/path/to/0001.jpg", "/path/to/0001.npy")
```

### CLI:端到端单图推理

```bash
python infer_pipeline.py \
  --input /abs/path/to/image.png \
  --out_dir output_infer \
  --smirk_ckpt smirk/pretrained_models/SMIRK_em1.pt \
  --facecheck_ckpt outputs/facecheck_version2/best.pt \
  --device cuda
```

`--out_dir` 下产物:

| 文件                          | 内容                                                 |
| ----------------------------- | ---------------------------------------------------- |
| `00_input.png`                | 原图备份                                             |
| `01_cropped_224.png`          | crop 后 224×224                                      |
| `01_crop.json`                | crop 几何变换 + mediapipe 关键点                     |
| `01_dynaface_quicktest.jpg`   | 在原始分辨率上用全部 measures 跑一次的 overlay     |
| `02_depth.npy`                | SMIRK 深度(float32)                                |
| `02_depth_mask.npy`           | SMIRK face mask(uint8)                             |
| `02_depth_vis.png`            | 深度 colormap 可视化                                 |
| `02_smirk_vertices.npy`       | FLAME vertices                                       |
| `02_smirk_raw.json`           | SMIRK 输出 key 列表 + vertices shape                 |
| `03_dynaface_landmarks.json`  | 97 点 (x, y)                                         |
| `03_dynaface_overlay.png`     | landmark overlay                                     |
| `04_landmark_vec.npy`         | 真正喂给 FaceCheck 的 landmark 向量(已截到 30 维) |
| `04_facecheck_result.json`    | `{prob_affected, label}`                             |
| `final.png`                   | 汇总图(原图 / crop / depth / overlay+label)       |

## 推理服务(FastAPI)

最新模型默认存放在 `outputs/facecheck_version2/best.pt`,API 启动时按以下优先级查 ckpt:

1. `FACECHECK_CKPT` 环境变量(支持 `~` / 相对路径)
2. `outputs/facecheck_version2/best.pt`
3. `outputs/facecheck_version1/best.pt`
4. `outputs/facecheck/best.pt`

### 启动

```bash
conda activate fcheck
# 可选:显式覆盖 ckpt
# export FACECHECK_CKPT="$(pwd)/outputs/facecheck_version2/best.pt"
# /pipeline 需要 SMIRK 权重
export SMIRK_CKPT="$(pwd)/smirk/pretrained_models/SMIRK_em1.pt"
# 输出 zip 的临时目录,默认 /tmp/facecheck_outputs
export OUTPUT_ROOT="/tmp/facecheck_outputs"
# 推理设备
export DEVICE="cuda"

python -m uvicorn facecheck.api.app:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000/docs` 查看 OpenAPI 文档。

### 端点

| 方法    | 路径         | 说明                                                                                                  |
| ------- | ------------ | ----------------------------------------------------------------------------------------------------- |
| `GET`   | `/health`    | 健康检查,返回 `{status: "ok"}`                                                                        |
| `POST`  | `/predict`   | 直接打分。`image`(jpg/png) + `depth`(.npy),可选 `landmark`(JSON 30 维数组)。返回 `{prob_affected, label}` |
| `POST`  | `/quicktest` | 仅跑 DynaFace 全套测量,导出 overlay 图(zip),不需要深度,不调用分类器                              |
| `POST`  | `/heic`      | iPhone HEIC 整张图,自动读嵌入深度 + matte → 跑完整 pipeline,返回 zip                                  |
| `POST`  | `/pipeline`  | 任意 RGB 图 → SMIRK 估计深度 → 完整 pipeline,返回 zip(需要 `SMIRK_CKPT`)                            |

`/heic` 与 `/pipeline` 返回的是 `infer_pipeline.run_pipeline` 的全部产物打包成的 zip(对应"端到端 CLI"那张表),`/predict` 只返回 JSON。

### 调用示例

```bash
# 直接打分:cropped 224×224 + .npy 深度
curl -X POST http://localhost:8000/predict \
  -F "image=@cropped.jpg" \
  -F "depth=@depth.npy"

# iPhone HEIC(内嵌深度)
curl -X POST http://localhost:8000/heic \
  -F "image=@IMG_0267.HEIC" \
  -o result.zip

# 普通 RGB 图(走 SMIRK 估计深度)
curl -X POST http://localhost:8000/pipeline \
  -F "image=@photo.png" \
  -o result.zip
```

### Docker

仓库根目录有 `Dockerfile`(基于 python:3.10-slim,装 `requirements.txt` + `smirk/requirements.txt`)。PyTorch3D wheel 因 CUDA/Python 版本而异,通过 `--build-arg` 注入:

```bash
docker build -t facecheck \
  --build-arg PYTORCH3D_WHL=https://example.com/pytorch3d-XXX.whl .

docker run --gpus all -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/smirk/pretrained_models:/app/smirk/pretrained_models \
  -e DEVICE=cuda \
  facecheck
```

## 测试

```bash
pytest -q
```

包含:模型 forward shape、预处理 shape 校验。

## 致谢

- [DINOv3](https://github.com/facebookresearch/dinov3):自监督 ViT 预训练
- [DynaFace](https://github.com/jeffheaton/dynaface):面部 landmarks 与测量
- [SMIRK](https://github.com/georgeretsi/smirk):单图 3D 重建/深度估计
- [timm](https://github.com/huggingface/pytorch-image-models):ViT 主干实现
- PyTorch / FastAPI
