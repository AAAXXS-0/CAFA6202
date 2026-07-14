# 长图分支说明

长图分支严格使用赛题已经分好的长图目录，不做自动路由。整体流程如下：

```text
2048 高滑窗、1792 步长
    → general6-8n 基础版面检测
    → 全局坐标映射与 256 重叠区去重
    → 多行逻辑 Title 合并
    → 居中标题推断目录标题和正文 H1
    → 连续 Title 组生成 H2/H3
    → 按标题树二次语义切块
    → 相邻小块打包为不超过 3900px 的 VLM 请求图
    → FinixDoc-VL
    → 相邻请求结果接缝去重与 Markdown 聚合
```

## 代码阅读顺序

```text
步骤001_数据定义.py             # 滑窗、检测框、标题和语义块的数据结构
步骤002_图片读写与裁切.py       # 从超长原图保存滑窗和最终裁片
步骤003_滑窗与YOLO检测.py       # 2048/1792 滑窗、general6、责任区与 NMS
步骤004_标题层级与二次分块.py   # Title 合并、H1/H2/H3 和语义分段
步骤005_大模型请求打包.py       # 把细粒度语义块合并成 VLM 请求图
步骤006_全流程调度.py           # 准备目录、API 缓存、Markdown 聚合与 CSV
```

`工具/` 中的文件同样按使用顺序编号：先检查准备结果，再绘制检测图，最后估算 API 请求量。

## 1. 配置

配置示例为 `afac_pipeline/long/config.example.json`，关键参数为：

- 检测模型：`360LayoutAnalysis/general6-8n.pt`；
- 窗口高度：2048；
- 窗口步长：1792；
- 检测窗口重叠：256；
- YOLO 输入尺寸：640（与 legacy 调用默认值一致）；
- Title 最低置信度：0.60（规整标题采用更严格阈值）；
- Text 最低置信度：0.50；
- FinixDoc-VL 请求图最大高度：3900；
- 超长语义段物理重叠：200。

模型标签只使用 `Text、Title、Figure、Table、Equation、Caption`，标题推断不依赖不稳定的 Toc 标签。

YOLO 参数以 legacy 实测有效调用为基线：`imgsz=640、conf=0.5`。曾使用的 `imgsz=1280、conf=0.15` 会产生大量低置信度重叠 Text/Title 框，因此不再作为默认配置。

## 2. 准备长图

```bash
python main.py prepare-long \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images" \
  --work-dir work/long \
  --config afac_pipeline/long/config.example.json
```

输出结构：

```text
work/long/
├── cache.sqlite3
├── dataset_manifest.json
└── prepared/<文件名_哈希>/
    ├── manifest.json
    ├── detection_windows/       # 送入 YOLO 的原始滑窗
    ├── yolo_raw/               # Ultralytics Results.save 原始标框与 JSON
    ├── semantic_crops/
    │   ├── _document/front_matter/
    │   ├── _document/toc/
    │   └── h2_0000/h3_0000/
    └── vlm_requests/
```

标题文字在小模型阶段未知，因此磁盘目录使用稳定 ID；FinixDoc-VL 返回标题文字后写入 Markdown，不用未经 OCR 清洗的标题文字重命名目录。

当配置 `save_yolo_debug=true` 时，`yolo_raw/` 中的图片是 Ultralytics 原生 `model.predict(save=True)` 自动输出；为避免批次间的 `image0.jpg` 重名，图片按 `批次000/`、`批次001/` 分目录保存。`predictions.json` 记录每个原始框是否通过分类阈值和滑窗责任区；可用于区分模型误检与代码后处理问题。

`semantic_crops/` 保留按 H2/H3 组织的细粒度审计切块。`vlm_requests/` 将坐标连续的小块重新打包，以控制 API 调用量，同时保证每个请求图不超过 3900px 高。

## 3. 标题规则

1. 极近、尺寸和中心位置相似的 Title 先合并为多行逻辑标题。
2. 正文前只有一个可信居中标题时，该标题为正文 H1。
3. 正文前有两个或更多可信居中标题时，最后一个为正文 H1，前一个为目录标题，更早内容归入前置信息。
4. 正文中连续且中间没有内容的 Title 组成标题组。
5. 至少两个 Title 的标题组中，第一个为 H2，其余为 H3。
6. 当前 H2 后的单个标题为 H3；如果全文没有可用 H2，第一个单标题降级为 H2。
7. H2 后紧跟 H3 时，第一个 H3 裁片从 H2 顶部开始，保证父标题不会丢失。

## 4. 当前 A 榜切块实测

50 张输入中有 33 张 SHA-256 唯一图，17 张完全重复图直接复用结果。33 张唯一图共生成 1261 个检测窗口和 5193 个逻辑图片块：

- 逻辑块高度：中位数 308px，P95 为 1102px，最大 3900px；
- 3822 个逻辑块低于 512px，因此不适合逐块请求大模型；
- 打包后为 644 个 VLM 请求图，平均每张唯一图 19.52 个；
- 请求图高度中位数 3670px，P95 为 3878px，最大 3900px；
- 0 个请求图超过 4096px，所有原图纵向覆盖缺口为 0。

因此，5193 是可追踪标题层级的逻辑切块数，不是 API 调用数；实际预计调用约 644 次。

## 5. 调用 FinixDoc-VL

```bash
python main.py run-long \
  --manifest work/long/dataset_manifest.json \
  --work-dir work/long \
  --credentials-file FinixDoc_VL调用.txt \
  --user-id finixB2002 \
  --request-timeout 240 \
  --max-retries 50 \
  --output-csv outputs/long_submission.csv
```

客户端按官方 multipart 协议上传图片，并解析双层 JSON 中的 Markdown。切片响应和完整图片结果均写入 SQLite 缓存，中断后重跑不会重复请求成功切片，完全重复图片也直接复用完整 Markdown。官方网关偶尔会以 HTTP 200 返回“服务器繁忙”HTML；程序会将其识别为临时错误，在 5 个官方白名单账号间轮换，并按第 n 次重试等待 `n × log₂(n)` 秒，默认最多重试 50 次，不会用错误 HTML 污染识别结果。

## 6. 校验与调试

```bash
python afac_pipeline/long/工具/工具001_检查准备结果.py --manifest work/long/dataset_manifest.json
python afac_pipeline/long/工具/工具003_估算请求数量.py --manifest work/long/dataset_manifest.json
python -m unittest discover -s tests -v
```

校验脚本检查原图纵向覆盖、切片尺寸、标签分布、标题角色以及每张图的窗口/切片数量。
