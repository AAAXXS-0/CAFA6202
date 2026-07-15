# 长图分支说明

长图默认使用 `semantic` 语义章节策略，旧版自适应安全切块保留为 `legacy`。当前正式语义流程是 v2：严格小模型标题候选与全宽墨迹扫描彼此独立，只在排版样式聚类阶段汇合。

## 正式流程

```mermaid
flowchart TD
    A["赛题长图"] --> B["固定窗口<br/>2048 高 / 1792 步长"]
    B --> C["general6-8n 检测"]
    C --> D["只保留置信度不低于 0.60 的 Title 候选"]
    B --> E["独立扫描窗口完整责任区墨迹"]
    E --> F["视觉文字行、实际行高、缩进和上下留白"]
    D --> G["标题候选与墨迹行按原图坐标匹配"]
    F --> G
    F --> H["没有模型框但显著大字号的独立墨迹候选"]
    G --> I["按相对字号与缩进聚类排版样式"]
    H --> I
    I --> J{"存在可靠非居中大标题样式？"}
    J -- "否" --> K["明确回退 legacy<br/>不强制制造 H2"]
    J -- "是" --> L["最高可信样式作为 H2<br/>后续样式作为 H3/H4 候选"]
    L --> M{"完整 H2 是否不超过 3900px？"}
    M -- "是" --> N["H2 及全部子标题一次请求"]
    M -- "否" --> O["按 H3 分组"]
    O --> P{"单个 H3 是否仍过长？"}
    P -- "否" --> Q["H2 标题条 + 若干完整 H3"]
    P -- "是" --> R["H2 + H3 标题条<br/>安全空白切正文"]
    N --> S["FinixDoc-VL 输出 Markdown"]
    Q --> S
    R --> S
    S --> T["按稳定标题 ID 和原图顺序聚合"]
```

小模型始终只看固定窗口，不会接收整个超长 H2。H2 很长只影响最终大模型请求规划。

## 历史实现归档

已经退出正式流程的算法位于：

```text
归档/
├── 连续标题层级_v0.py
├── 语义标题分析_v1.py
└── README.md
```

- v0：连续 Title 组中第一个直接视为 H2，后续视为 H3。
- v1：模型、墨迹与旧规则加权，但“连续组开头”和“旧规则 H2”相关性过高，被重复计权。

正式 `semantic` 不再导入这些文件。本地 OCR 备用线为了兼容旧输出，可以显式调用 v0，但不影响 FinixDoc-VL 主流程。

## 两条独立证据链

### 小模型链

检测窗口参数：

```json
{
  "window_height": 2048,
  "window_step": 1792,
  "yolo_imgsz": 640,
  "yolo_base_confidence": 0.25,
  "title_confidence": 0.60
}
```

`yolo_base_confidence=0.25` 只用于保留版面框和安全切割保护。正式标题候选必须再次满足：

```text
Title confidence ≥ 0.60
```

低于 0.60 的 Title 不参与样式聚类，也不能借助其他分数重新变成模型标题。

### 独立墨迹链

墨迹分析读取窗口完整图片，不裁 YOLO Title/Text 框。

```json
{
  "semantic_ink_threshold": 225,
  "semantic_full_width_active_ratio": 0.002,
  "semantic_line_merge_gap": 2,
  "semantic_min_ink_line_height": 6,
  "semantic_min_ink_width_ratio": 0.02
}
```

- 灰度低于 225 视为墨迹。
- 一横行至少有 0.2% 墨迹才是活跃文字行。
- 相隔不超过 2px 的活跃行合并。
- 高度低于 6px 或宽度低于图片 2% 的小噪声删除。
- 每个窗口只保留中心位于自身 ownership 范围的文字行，避免重叠窗口重复。
- 全文视觉行高中位数作为正文字号基准。

因此，即使 YOLO 漏掉一个特别大的 H2，墨迹链仍能独立看到它。

## 候选准入

### 模型候选

模型 Title 必须与独立墨迹行匹配，并满足：

```json
"semantic_model_title_min_ratio": 1.05
```

即实际墨迹行高至少是本文中位字号的 1.05 倍。高置信度但与正文同字号的误框不能进入标题样式。

### 纯墨迹候选

没有模型 Title 支持时，必须同时满足：

```json
{
  "semantic_ink_only_title_ratio": 1.35,
  "semantic_ink_only_min_whitespace_ratio": 0.60
}
```

即字号至少为正文 1.35 倍，并有足够上下留白。纯墨迹候选采用更严格条件，避免普通粗体正文大量冒充标题。

## 排版样式聚类

候选按照两个维度聚类：

- 实际墨迹行高 ÷ 正文字号；
- 左边距 ÷ 图片宽度。

参数：

```json
{
  "semantic_style_height_tolerance": 0.10,
  "semantic_style_indent_tolerance": 0.06,
  "semantic_min_heading_ratio": 1.08,
  "semantic_h2_cluster_height_tolerance": 0.08
}
```

- 字号差不超过 10%、缩进差不超过图片宽度 6% 的候选视为同一排版样式。
- H2 样式至少比正文高 8%。
- 最高的可靠非居中样式作为 H2。
- 与最高 H2 样式字号差不超过 8%、且没有更深缩进的相邻样式也可并入 H2。
- 其余可信样式按字号和缩进顺序成为 H3/H4 候选。
- 找不到可靠 H2 时直接回退 legacy，不选“最高分标题”凑数。

这里不再使用旧连续 Title 规则，也没有 H2 加权分数。

## H2/H3 请求规划

```json
{
  "max_vlm_height": 3900,
  "semantic_title_padding": 12,
  "semantic_context_gap": 10
}
```

- 完整 H2 不超过 3900px：整章一次请求。
- 超长 H2：相邻完整 H3 尽量组合到同一请求。
- 超长 H3：正文从 H3 标题下沿开始，顶部复制 H2+H3 原图标题条。
- 标题框会扩展到匹配墨迹的真实上下边缘，避免从模型框边缘切掉字。
- H2 前方完全空白时不生成单独的空请求。

超长 H3 的最终兜底仍使用：

```json
{
  "adaptive_target_height": 3200,
  "adaptive_min_height": 2200,
  "safe_cut_search": 600,
  "projection_blank_ratio": 0.01,
  "minimum_blank_band": 8,
  "vlm_overlap": 200
}
```

## 中间产物

```text
prepared/<文件名_哈希>/
├── manifest.json
├── detection_windows/
├── yolo_raw/
├── semantic_audit/
│   ├── 004_独立墨迹行窗口图/
│   ├── 005_严格模型标题窗口图/
│   ├── 006_标题样式聚类.json
│   ├── 007_最终标题层级窗口图/
│   └── 008_请求切块清单.json
├── vlm_request_parts/
└── vlm_requests/
```

`006_标题样式聚类.json` 包含：

- 每条独立墨迹行；
- 正文字号中位数；
- 每个严格模型候选及匹配墨迹；
- 纯墨迹候选；
- 排版样式簇；
- 最终 H1/H2/H3/H4；
- 被拒绝候选及原因；
- 是否需要回退 legacy。

## 代码阅读顺序

```text
步骤001_数据定义.py
步骤002_图片读写与裁切.py
步骤003_滑窗与YOLO检测.py
步骤004_语义标题分析.py
步骤004_自适应安全切块.py
步骤005_大模型请求打包.py
步骤006_全流程调度.py
步骤007_本地OCR识别.py
归档/
```

## 运行

```bash
/usr/bin/python3 main.py prepare-long \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images" \
  --work-dir work/long \
  --config afac_pipeline/long/config.example.json

/usr/bin/python3 main.py run-long \
  --manifest work/long/dataset_manifest.json \
  --work-dir work/long \
  --credentials-file FinixDoc_VL调用.txt \
  --user-id finixB2002 \
  --output-csv outputs/long_submission.csv
```

每个 API 原始回答保存在 `responses/request_*.md`，去掉重复上下文标题后真正用于聚合的版本保存在 `responses/request_*_聚合输入.md`。

## 当前真实验证

验证目录：

```text
work/验证/长图语义_v2/输出_修正/
```

样例 `3ed0a71a...` 高 30690px：

```text
旧 v1：7 个 H2、47 个 H3 候选、13 个请求
新 v2：1 个 H2、36 个 H3 候选、11 个请求
```

新版本识别出的唯一 H2 原图范围为 `y=102～182`，实际墨迹行高 80px；正文中位行高 49px。其余主要标题样式约 57px，被归入 H3，和该样例“释义章节 + 多个释义条目”的结构一致。

## 测试

```bash
/usr/bin/python3 -m unittest discover -s tests -v
```

测试明确覆盖 Title 0.60 门槛、模型与墨迹独立性、纯墨迹 H2、无可靠 H2 回退、样式层级、超长 H3 上下文以及重复标题去除。
