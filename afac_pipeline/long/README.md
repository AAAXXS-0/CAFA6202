# 长图分支说明

长图分支只处理赛题已经分好的长图目录，不做自动路由。新流程把“小模型检测窗口”和“最终 VLM 图片块”彻底分开：

```text
固定 2048/1792 检测滑窗
    → general6-8n 低阈值版面保护框
    → 检测框映射回原图并全局去重

检测窗口责任区逐行墨水投影
    + 全局版面保护区间
    → 在 3200px 附近寻找安全空白带
    → 生成最高 3900px 的连续原图块
    → 只有找不到安全空白时才使用 200px 重叠兜底
    → FinixDoc-VL
    → 仅对真实重叠块进行接缝去重
    → 根据 VLM 已识别的标题编号校正 Markdown 层级
```

小模型只负责提供“哪里可能有内容”的保守版面地图，不再依据连续 Title 推断 H1/H2/H3，也不再按照推测出的标题树切割原图。

## 代码阅读顺序

```text
步骤001_数据定义.py             # 检测窗口、版面框和最终安全块数据结构
步骤002_图片读写与裁切.py       # 从超长原图保存检测窗口和 VLM 请求图
步骤003_滑窗与YOLO检测.py       # 固定检测滑窗、低阈值保护框、责任区和 NMS
步骤004_自适应安全切块.py       # 墨水投影、保护区间、空白带与重叠兜底
步骤005_大模型请求打包.py       # 请求提示词和 Markdown 标题编号校正
步骤006_全流程调度.py           # 准备目录、API 缓存、聚合与 CSV
```

旧版“连续 Title 推断标题层级”的实现已移动到：

```text
工具/工具004_旧标题层级分析.py
```

它只用于历史对照和旧测试，不参与新版正式切块。

## 1. 关键配置

配置示例为 `afac_pipeline/long/config.example.json`。

检测阶段：

- 检测窗口高度：2048；
- 窗口步长：1792；
- 检测重叠：256；
- YOLO 输入尺寸：640；
- YOLO 基础置信度：0.25；
- 切割保护置信度：0.25；
- 保护框上下扩展：16px；
- Title 语义参考阈值：0.60；
- Text 语义参考阈值：0.50。

基础置信度降低到 0.25 只用于安全切割保护。低置信度框不会被用于强制判断标题层级；误保护最多让切口稍微移动，漏保护则可能切断正文，因此这里采用偏保守策略。

自适应切割阶段：

- 墨水投影采样宽度：256；
- 白色像素阈值：245；
- 空白行最大墨水比例：0.01；
- 最小连续空白带：8px；
- 目标块高度：3200；
- 常规最小块高度：2200；
- 最大块高度：3900；
- 安全切线搜索半径：600；
- 无安全空白时兜底重叠：200。

## 2. 两套切割的职责

### 检测窗口

检测窗口固定为 2048/1792，不直接发送给 FinixDoc-VL。固定尺寸能保证 general6 的文字缩放比例稳定，256px 重叠使落在边缘的标题通常至少在一个窗口中完整出现。

每个窗口只有自己的 ownership 责任区可以保留检测框，随后所有框映射回原图坐标并做全局去重。

### 最终 VLM 安全块

程序利用检测窗口的 ownership 区域计算原图逐行墨水比例，因此不需要再次把十万像素高的原图整体解码到额外内存。

对于每个预计切口：

1. 从当前块起点向下约 3200px；
2. 在前后 600px 搜索连续空白带；
3. 排除穿过 Text、Title、Table、Figure 等保护框的位置；
4. 从得分最高的空白带切开，前后块不重叠；
5. 如果合法高度范围内完全没有安全空白，选择墨水最少的位置；
6. 兜底切口前后各保留约 100px，总重叠约 200px。

任何请求块都不得超过 3900px，并且所有请求块的并集必须完整覆盖原图。

## 3. 准备长图

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
    ├── detection_windows/       # 只送入小模型的固定滑窗
    ├── yolo_raw/                # Ultralytics 原生标框和审计 JSON
    └── vlm_requests/            # 从原图裁出的自适应安全块
```

单图 `manifest.json` 的关键字段：

- `windows`：固定检测滑窗和 ownership 范围；
- `layout_blocks`：达到保护阈值的全局版面框；
- `adaptive_cutting.protection_intervals`：禁止切割的纵向区间；
- `adaptive_cutting.blank_bands`：原图连续空白带；
- `safe_chunks`：最终安全块及其重叠信息；
- `request_packs`：实际 FinixDoc-VL 请求。

当 `save_yolo_debug=true` 时，`yolo_raw/` 继续保留 Ultralytics 的 `model.predict(save=True)` 原始输出。

## 4. Markdown 标题和拼接

程序不再用小模型的 Title 分布提前决定标题层级。FinixDoc-VL 先根据可见编号、字号和排版输出 Markdown，随后只对“已经是标题”的行校正井号：

```text
1 总则          → #
1.1 投保条件    → ##
1.1.1 责任范围  → ###
一、总则        → #
（一）合同构成  → ##
```

普通正文即使包含 `2.1` 等编号，也不会被程序提升为标题。

从安全空白带切开的相邻请求直接拼接；只有 `overlap_top > 0` 的兜底请求才执行接缝文本去重，从而避免相邻章节恰好出现相同句子时被误删。

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

成功切块和完整图片结果都进入 SQLite。服务器繁忙或超时时，客户端会在 5 个白名单 userId 间轮换，并按第 n 次重试等待 `n × log₂(n)` 秒。

## 6. 校验与调试

```bash
python afac_pipeline/long/工具/工具001_检查准备结果.py \
  --manifest work/long/dataset_manifest.json

python afac_pipeline/long/工具/工具003_估算请求数量.py \
  --manifest work/long/dataset_manifest.json

python -m unittest discover -s tests -v
```

检查工具会验证：

- 原图纵向覆盖是否存在缺口；
- 请求图是否超过 4096px；
- 安全空白切口和重叠兜底各有多少；
- 每张唯一图片的窗口数、保护框数和实际请求数。
