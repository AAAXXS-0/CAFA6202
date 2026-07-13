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
    → 超过 3900 的语义段继续物理切块
    → FinixDoc-VL
    → 相邻物理块接缝去重与 Markdown 聚合
```

## 1. 配置

默认配置在 `long_config.example.json`，关键参数为：

- 检测模型：`360LayoutAnalysis/general6-8n.pt`；
- 窗口高度：2048；
- 窗口步长：1792；
- 检测窗口重叠：256；
- YOLO 输入尺寸：1280；
- Title 最低置信度：0.20；
- Text 最低置信度：0.25；
- FinixDoc-VL 切片最大高度：3900；
- 超长语义段物理重叠：200。

模型只使用 `Text、Title、Figure、Table、Equation、Caption`，不使用 Toc 标签。

## 2. 准备长图

```bash
python main.py prepare-long \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images" \
  --work-dir work/long \
  --config long_config.example.json
```

输出结构：

```text
work/long/
├── cache.sqlite3
├── dataset_manifest.json
└── prepared/<文件名_哈希>/
    ├── manifest.json
    ├── detection_windows/
    └── semantic_crops/
        ├── _document/front_matter/
        ├── _document/toc/
        └── h2_0000/h3_0000/
```

标题文字在小模型阶段未知，所以目录使用稳定 ID。FinixDoc-VL 返回标题文字后保存在 Markdown 和响应文件中，不直接用未清洗的标题文字重命名文件夹。

为控制 3 小时内的 API 调用量，`semantic_crops/` 继续逐 H2/H3 保存审计切块，程序另外把坐标连续的小语义段打包到 `vlm_requests/`，每个请求仍不超过 3900 像素高。当前 A 榜 33 张唯一长图共有 5193 个逻辑语义块，打包后预计只需 644 次 FinixDoc-VL 请求。

`manifest.json` 保存窗口责任区、全部全局检测框、逻辑标题、H1/H2/H3、语义段和最终裁片坐标。

## 3. 标题规则

1. 极近、尺寸和中心位置相似的 Title 先合并为多行逻辑标题。
2. 正文前只有一个可信居中标题时，该标题为正文 H1。
3. 正文前有两个或更多可信居中标题时，最后一个为正文 H1，前一个为目录标题，更早内容归入前置信息。
4. 正文中连续且中间无内容的 Title 组成标题组。
5. 长度大于等于 2 的标题组：首个为 H2，其余为 H3。
6. 当前 H2 后的单个标题为 H3；如果全文没有可用 H2，第一个单标题降级为 H2。
7. H2 后紧跟 H3 时，第一个 H3 图片从 H2 顶部开始，保证 H2 不会丢失。

## 4. 调用 FinixDoc-VL

```bash
export FINIXDOC_API_KEY="你的密钥"

python main.py run-long \
  --manifest work/long/dataset_manifest.json \
  --work-dir work/long \
  --api-url "主办方完整接口地址" \
  --model "FinixDoc-VL" \
  --output-csv outputs/long_submission.csv
```

切片响应和完整图片结果均写入 SQLite 缓存。中断后再次运行不会重复请求已经成功的切片，完全重复图片也直接复用完整 Markdown。

## 5. 校验

```bash
python scripts/validate_long_prepared.py \
  --manifest work/long/dataset_manifest.json

python -m unittest discover -s tests -v
```

校验脚本会检查原图纵向覆盖、切片尺寸、标签分布、标题角色和每张图片的窗口/切片数量。
