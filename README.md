# AFAC 2026 文档解析工作流

本仓库按赛题已经分好的数据目录，分别处理长图文档和图表图片；不做自动路由。两个分支共享精确去重、图像读取、缓存、本地 OCR/FinixDoc-VL 识别后端和提交文件生成，但检测、切块与后处理逻辑彼此独立。

## 项目结构

```text
AFAC2026_challger/
├── afac_pipeline/
│   ├── common/                 # SHA-256、图像后端、缓存、VLM 客户端、CSV
│   ├── long/                   # 长图代码、配置示例和分支 README
│   └── table/                  # 图表代码、配置、文档和分支工具
├── experiments/legacy_long/   # 早期随机切图和模型试验脚本，仅供追溯
├── tests/                     # 自动化测试
├── requirements-local-ocr.txt # 可选 RapidOCR 离线识别依赖
├── main.py                    # 统一命令行入口
├── requirements.txt
└── 赛题.txt
```

分支文档：

- [长图分支](afac_pipeline/long/README.md)：固定检测滑窗、general6 保护框、墨水投影、自适应安全切块和 Markdown 聚合。
- [图表分支](afac_pipeline/table/README.md)：表格检测、缩放/二维切片、Markdown 表格拼接和失败处理。

## 安装

建议使用 Python 3.10～3.12：

```bash
python -m pip install -r requirements.txt
```

超大图片建议安装系统 `libvips`，否则会自动使用 Pillow，功能不受影响但峰值内存更高。

## 最简单的完整运行方式

无需填写任何路径或 API 参数，直接运行：

```bash
/usr/bin/python3 一键生成最终CSV.py
```

也可以执行：

```bash
./一键生成最终CSV.sh
```

脚本会自动准备两个分支、断点续跑官方 API、按模板合并 100 行结果，并输出到 `outputs/最终提交/finix_ab_A_submit.csv`。API 繁忙时稍后再次运行同一个文件即可。

## 统一命令行

```bash
python main.py --help
```

当前命令包括：

- `hash-report`：统计字节完全相同的图片；
- `prepare-tables` / `run-tables`：准备、识别图表目录；
- `prepare-long` / `run-long`：准备、识别长图目录；
- `combine-submissions`：按官方模板顺序合并长图和图表 CSV。

## 完全本地 OCR（不调用官方 API）

第一版本地后端使用轻量中文 RapidOCR。安装到独立目录，不污染项目环境：

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

也可以只跑一个分支：

```bash
PYTHONPATH=/tmp/afac_rapidocr python3 main.py run-local-long \
  --manifest work/long/dataset_manifest.json \
  --work-dir work/local_ocr \
  --output-csv outputs/long_local_ocr.csv

PYTHONPATH=/tmp/afac_rapidocr python3 main.py run-local-tables \
  --manifest work/tables/dataset_manifest.json \
  --work-dir work/local_ocr \
  --output-csv outputs/table_local_ocr.csv
```

本地 OCR 默认把请求图继续切成最长边约 2000px、带 160px 重叠的小块，
所以不会为了识别一张 3900px 表格而把所有小字强行缩小。每个小块立即写入
`work/local_ocr/cache`；中断后执行同一命令会从缓存继续。

长图使用小模型 Title/Text 坐标恢复 Markdown 标题和正文段落。图表不会把
OCR 视觉行直接猜成表格，而是把每个文字框投回 v6 已检测的逻辑单元格，
删除整行/整列完全无字的冗余白带后确定性生成 HTML。公共层只约定文字框
接口，以后可换 PaddleOCR GPU 或其他本地模型，不需要重写两个分支后处理。

配置示例分别位于：

- `afac_pipeline/table/config.example.json`
- `afac_pipeline/long/config.example.json`

## SHA-256 去重说明

SHA-256 根据文件的全部字节生成摘要。摘要相同可以视为文件字节完全相同，因此能够安全复用解析结果；重新压缩、修改元数据或改变一个像素都会产生不同摘要。它不判断“视觉相似”，不会把内容接近但文字不同的表格误合并。

当前 A 榜数据实测：图表 50 张中 49 张唯一图；长图 50 张中 33 张唯一图，后者可直接复用 17 张的完整结果。

## 官方 API 与最终提交

官方 `FinixDoc_VL调用.txt` 使用 multipart 表单上传，客户端会解析外层业务 JSON、内层模型 JSON 和 Markdown 代码围栏：

```bash
python main.py run-long \
  --manifest work/long/dataset_manifest.json \
  --work-dir work/long \
  --credentials-file FinixDoc_VL调用.txt \
  --user-id finixB2002 \
  --output-csv outputs/long_submission.csv
```

如果官方网关以 HTTP 200 返回“服务器繁忙”HTML，程序会识别为临时错误并按配置重试，而不会把 HTML 当作 Markdown。官方说明中的 5 个白名单 userId 会在重试时循环切换，第 n 次重试前等待 `n × log₂(n)` 秒；默认最多重试 50 次。一键脚本可用环境变量 `FINIXDOC_MAX_RETRIES` 修改上限。每个成功切片即时进入 SQLite 缓存。

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

测试覆盖精确哈希、图表检测和聚合、长图检测滑窗、自适应安全切割、标题编号校正、缓存和 CSV 输出。
