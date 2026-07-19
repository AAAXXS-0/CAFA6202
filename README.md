# AFAC 2026 文档解析工作流

本仓库按赛题已经分好的数据目录，分别处理长图文档和图表图片；不做自动路由。两个分支共享精确去重、图像读取、缓存、识别后端和提交文件生成，但检测、切块与后处理逻辑彼此独立。

## 项目结构

```text
AFAC2026_challger/
├── afac_pipeline/
│   ├── common/                 # SHA-256、图像后端、缓存、识别客户端、CSV
│   ├── long/                   # 长图代码、配置示例和分支 README
│   └── table/                  # 图表步骤001～011、配置、工具、归档和分支 README
├── experiments/legacy_long/   # 早期随机切图和模型试验脚本，仅供追溯
├── tests/                     # 自动化测试
├── requirements-firered.txt   # FireRed-OCR-2B 独立环境依赖
├── requirements-local-vl.txt  # PaddleOCR-VL 独立环境依赖
├── requirements-local-ocr.txt # 可选 RapidOCR 离线识别依赖
├── 一键生成本地模型CSV.py     # 推荐：4060 本地模型完整流程
├── 一键生成FireRed模型CSV.py  # FireRed 单实例完整流程
├── 测试FireRed单模型.py        # FireRed 单实例顺序测试入口
├── 一键生成最终CSV.py         # 备用：赛事官方 API 完整流程
├── main.py                    # 统一命令行入口
├── requirements.txt
└── 赛题.txt
```

分支文档：

- [长图分支](afac_pipeline/long/README.md)：严格 Title 0.60、独立全宽墨迹扫描、排版样式聚类、H2/H3 语义切块和祖先标题上下文；历史标题算法已归档，旧安全切割保留为 legacy。
- [图表分支](afac_pipeline/table/README.md)：5%密度分表、20%白带、50%黑线、竖线98%免灰度对比、黑线孤立压字伪线清理、按分表区域自适应清理细窄列白带、顶部候选标题和固定物理网格拼接。行白带仍保留1像素检测能力。

## 普通预处理环境

建议使用 Python 3.10～3.12：

```bash
python -m pip install -r requirements.txt
```

超大图片建议安装系统 `libvips`，否则会自动使用 Pillow，功能不受影响但峰值内存更高。

## 4060 本地 FireRed-OCR-2B

FireRed 使用完全独立的 Torch 环境，不与 PaddleOCR-VL 同时实例化：

```text
/home/zero/miniconda3/envs/AFAC_FIRERED
```

生成最终 100 行 CSV：

```bash
/usr/bin/python3 一键生成FireRed模型CSV.py
```

只检查 GPU、权重和预处理清单，不加载模型：

```bash
AFAC_FIRERED_DRY_RUN=1 /usr/bin/python3 一键生成FireRed模型CSV.py
```

测试一张或多张已有切块时，模型只加载一次，图片严格顺序执行：

```bash
/usr/bin/python3 测试FireRed单模型.py 图片1.png 图片2.png
```

输出保存在 `work/FireRed单模型测试/`，每张图片旁边还会生成包含耗时、峰值显存和模型签名的 `firered_raw/*.json`。测试入口固定使用 FireRed 官方 Markdown 转换提示词，不加载 Paddle、Nemotron 或第二份 FireRed。

正式流程固定 `max_workers=1`，长图约 80 万像素、输出上限 4096 token；图表约 160 万像素、输出上限 8192 token。实测正文块峰值显存 4.15 GiB；2199×707 宽表峰值 4.48 GiB、94.6 秒，完整输出 14×16 的 224 个单元格。

需要重建环境时：

```bash
/home/zero/miniconda3/bin/conda create -n AFAC_FIRERED python=3.11 pip -y

/home/zero/miniconda3/envs/AFAC_FIRERED/bin/python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

/home/zero/miniconda3/envs/AFAC_FIRERED/bin/python -m pip install \
  -r requirements-firered.txt
```

运行位置：

```text
SQLite 缓存：work/FireRed正式运行/
阶段 CSV：outputs/FireRed最终提交/长图结果.csv、图表结果.csv
最终 CSV：outputs/FireRed最终提交/finix_ab_A_submit.csv
```

## 推荐：4060 本地 PaddleOCR-VL

当前机器已经配置好独立环境：

```text
/home/zero/miniconda3/envs/AFAC_LOCAL_VL
```

不需要手动 `conda activate`，也不要用这个环境运行 YOLO。直接在项目根目录执行：

```bash
/usr/bin/python3 一键生成本地模型CSV.py
```

脚本会自动完成以下工作：

1. 用 `/usr/bin/python3` 复用或生成长图、图表预处理清单；
2. 自动切换到 `AFAC_LOCAL_VL`；
3. 用 GPU 0 单并行运行 `PaddleOCR-VL-1.6`；
4. 每个成功切块立即写入独立 SQLite 缓存；
5. 分别聚合长图和图表结果，按模板生成严格 100 行的最终 CSV。

只检查 GPU、环境、模型缓存和预处理清单，不开始识别：

```bash
AFAC_LOCAL_VL_DRY_RUN=1 /usr/bin/python3 一键生成本地模型CSV.py
```

### 长图本地模型策略

RTX 4060 Laptop 8GB 的实测甜点位为 30 万像素、1024 token。50 万像素以上在高长比切块上会进入非线性慢区间，100 万像素可能数分钟没有结果。

长图根据请求块高度自适应：

- 高度不超过 2048px：保留 `PP-DocLayoutV3`，适合标题、目录和短正文，帮助模型及时结束；
- 普通高块：关闭二次版面检测，使用整块 `ocr`。否则一个 H2 请求块可能再次拆成十几个内部 VLM 调用；
- 极端长宽比块：若超过 2048px 且按 30 万像素缩放后的预计宽度不足 512px，只在本地请求层沿横向空白带临时拆成约 1500px 的子块；
- 双栏目录先按中央空白槽分列，再按“左列自上而下、右列自上而下”切块；每块复制同一目录头，聚合只保留第一份；
- 每个临时子块立即写入独立缓存，重启后不重复识别已经成功的子块；
- 内部 VLM worker 队列固定关闭，避免 WSL 下进程挂起；
- 单块超过 30 秒时，每 30 秒打印一次累计耗时心跳；超过 120 秒主动失败，避免 CPU 单核假忙而 GPU 长期为 0%。

普通请求块实测耗时约 2.3～14.8 秒。极端长图临时切割属于本地模型容量适配，不改变官方 API 流程和原图语义坐标。本地 Markdown 后处理会结合原图 H1/H2/H3/H4 检测和编号深度恢复 `#`～`#####`；目录项保持普通目录行，不擅自升级成正文标题。

### 图表本地模型策略

图表文字更密，暂时保留 100 万像素、4096 token 和版面检测。特大逻辑 tile 会生成带样式 HTML，单块仍可能很慢；后续应按更少逻辑行再次细分，而不是降低整个表格的识别清晰度。

如需实验，可分别覆盖两类参数：

```bash
PADDLEOCR_MAX_PIXELS=300000 \
PADDLEOCR_MAX_NEW_TOKENS=1024 \
PADDLEOCR_TABLE_MAX_PIXELS=1000000 \
PADDLEOCR_TABLE_MAX_NEW_TOKENS=4096 \
/usr/bin/python3 一键生成本地模型CSV.py
```

运行位置：

```text
预处理：work/正式运行/
本地缓存：work/本地模型正式运行/
模型原始 JSON：对应 prepared 图片目录/local_vl_raw/
本地极端长图子块：对应 prepared 图片目录/local_vl_parts/
阶段 CSV：outputs/本地模型最终提交/长图结果.csv
阶段 CSV：outputs/本地模型最终提交/图表结果.csv
最终 CSV：outputs/本地模型最终提交/finix_ab_A_submit.csv
```

手动中断后重新运行同一命令即可续跑。程序会跳过相同模型签名下已经成功的切块。

如以后需要重建独立环境：

```bash
/home/zero/miniconda3/bin/conda create -n AFAC_LOCAL_VL python=3.10 -y

/home/zero/miniconda3/envs/AFAC_LOCAL_VL/bin/python -m pip install \
  paddlepaddle-gpu==3.2.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

/home/zero/miniconda3/envs/AFAC_LOCAL_VL/bin/python -m pip install \
  -r requirements-local-vl.txt
```

不要在该环境安装 CUDA 12.8/13.0 的 Torch，它会替换 Paddle 3.2.1 使用的 CUDA 12.6 运行库。长图 YOLO 仍由普通预处理环境负责。

## 备用：官方 API 一键流程

官方接口可用时执行：

```bash
/usr/bin/python3 一键生成最终CSV.py
```

也可以执行：

```bash
./一键生成最终CSV.sh
```

脚本会自动准备两个分支、断点续跑官方 API、按模板合并 100 行结果，并输出到
`outputs/最终提交/finix_ab_A_submit.csv`。官方协议严格只上传
`userId/apiKey/fileName/file`，不发送自定义提示词。默认同时识别 6 张唯一图片，
可用 `FINIXDOC_WORKERS=1～32` 调整；单张图片内部仍按原顺序聚合。每次请求默认超时
600 秒，可用 `FINIXDOC_TIMEOUT` 覆盖；`FINIXDOC_MAX_RETRIES` 控制最多 15 次平方退避重试。
配置变化时会自动迁移图片字节完全相同的旧切片缓存，并强制校验两个 50 行分支和
最终 100 行 CSV。

每次重试日志都会同时打印原图名和切块名。单块退让耗尽时只隔离对应原图：
已成功切块继续保存在 SQLite，该原图不写整图成功缓存，其余原图和另一分支继续
运行。整批结束后失败详情写入各工作目录的 `recognition_failures.json`，成功部分
写入 `partial_results.csv`；重新运行时只补失败原图及其尚未成功的切块。存在失败
时不会生成新的最终 100 行提交文件。

## 统一命令行

```bash
python main.py --help
```

当前命令包括：

- `hash-report`：统计字节完全相同的图片；
- `prepare-tables` / `run-tables`：准备、识别图表目录；
- `prepare-long` / `run-long`：准备、识别长图目录；
- `combine-submissions`：按官方模板顺序合并长图和图表 CSV。

`run-long` 和 `run-tables` 可传 `--workers 6` 并行识别唯一图片；本地 8GB GPU 客户端固定使用 1，避免多个任务同时挤爆显存。

## 轻量本地 OCR 备用线

RapidOCR 是早期离线保底方案，质量低于 PaddleOCR-VL，但代码继续保留。安装到独立目录，不污染项目环境：

```bash
python3 -m pip install --target /tmp/afac_rapidocr -r requirements-local-ocr.txt
```

使用同一个 Python 一键识别两个分支，并按官方模板生成最终 CSV：

```bash
PYTHONPATH=/tmp/afac_rapidocr python3 main.py run-local-all \
  --long-manifest work/long/dataset_manifest.json \
  --table-manifest work/tables/dataset_manifest.json \
  --template finix_ab_A_submit_mock.csv \
  --work-dir work/local_ocr \
  --output-csv outputs/local_ocr_submission.csv
```

本地 OCR 默认把请求图继续切成最长边约 2000px、带 160px 重叠的小块。长图使用小模型 Title/Text 坐标恢复标题和正文；图表把文字框投回 v6 已检测的逻辑单元格，确定性生成 HTML。

配置示例分别位于：

- `afac_pipeline/table/config.example.json`
- `afac_pipeline/long/config.example.json`

## SHA-256 去重说明

SHA-256 根据文件的全部字节生成摘要。摘要相同可以视为文件字节完全相同，因此能够安全复用解析结果；重新压缩、修改元数据或改变一个像素都会产生不同摘要。它不判断“视觉相似”，不会把内容接近但文字不同的表格误合并。

当前 A 榜数据实测：图表 50 张中 49 张唯一图；长图 50 张中 33 张唯一图，后者可直接复用 17 张的完整结果。

## 官方 API 细节与最终提交

官方 `FinixDoc_VL调用.txt` 使用 multipart 表单上传。客户端会解析外层业务 JSON、内层模型 JSON 和 Markdown 代码围栏。官方网关临时繁忙时会轮换说明文件中的白名单账号，并按 `64、81、100……484` 秒退让，最多重试 15 次。官方请求严格按说明文件只上传图片，不附加自定义提示词。

### 失败处理和断点续跑

请求、切块、原图和整批是四个不同层次，不能混为一个“失败”状态：

1. **请求级临时故障**：连接超时、HTTP 429/5xx、官方返回“识别失败/请求过载/服务器繁忙”等，按账号轮换和平方退让重试，最多 15 次。
2. **图表空响应**：接口正常返回但正文为空时，单独退让 3 次（共最多 4 次请求）。仍为空不算接口崩溃，而是按预处理得到的物理行列生成全空矩阵，并照常缓存、参与合并。
3. **预处理空表**：墨迹矩阵确认整块没有文字时，不调用模型，直接按同一物理行列生成全空矩阵；这不是错误。
4. **切块级最终失败**：某块所有退让都用尽后，不写成功缓存，记录原图名、区域和切块名，并继续处理同一张图的后续块。
5. **原图级不完整**：一张图的所有块都尝试完仍有缺块，才标记该原图不完整，写入 `recognition_failures.json`，进入下一张原图；已经成功的块缓存保留，重跑只补缺失块。
6. **整批级结果**：只要仍有原图不完整，就输出 `partial_results.csv` 和失败明细，不生成正式的 100 行提交 CSV，避免把缺图结果误提交。所有原图完整后才合并最终 CSV。

图表物理行列始终以预处理结果为准，模型只能填内容，不能扩大矩阵。读取旧
切片缓存时也会重新检查HTML闭合、重复table、围栏循环和结构膨胀；坏缓存
自动失效。模型正常返回全空、但墨迹bool认为有字时，会复核3次；连续全空
便按预处理行列生成完整空矩阵，这是针对墨迹bool约0.1%误判的明确特例，
不属于普通损坏。区域最终合并失败时禁止再把原始坏HTML直接拼入提交结果。

没有可靠物理行列的像素兜底块，以及没有物理网格的长图块，无法安全凭空补空矩阵；它们会作为切块失败记录，但不会阻塞同图后续块或其他原图。

长图和图表各自生成 50 行 CSV 后，按 100 行官方模板严格合并：

```bash
python main.py combine-submissions \
  --template finix_ab_A_submit_mock.csv \
  --input-csv outputs/long_submission.csv \
  --input-csv outputs/table_submission.csv \
  --output-csv outputs/finix_ab_A_submit.csv
```

合并时会拒绝重复、缺失、多余文件名和错误表头。

## 测试

```bash
python -m compileall -q afac_pipeline main.py tests
python -m unittest discover -s tests -v
```

测试覆盖精确哈希、图表检测和聚合、长图检测滑窗、H2/H3 语义切块、祖先标题去重、自适应安全切割、缓存、本地模型客户端和 CSV 输出。
