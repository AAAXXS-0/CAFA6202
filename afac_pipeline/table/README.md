# 图表分支说明

图表分支只接收赛题已经分好的图表目录，不做自动路由。流程为：原图元数据读取与 SHA-256 精确去重 → 预览图表格检测 → 检测框映射回原图 → 整体缩放或二维切片 → FinixDoc-VL → Markdown 表格拼接。

## 1. 配置与检测器

配置示例为 `afac_pipeline/table/config.example.json`。检测器有三种模式：

- `auto`：存在 Ultralytics 和权重时使用 YOLO，否则退回横线投影检测；
- `yolo`：强制使用配置的 YOLO 权重，依赖或权重缺失时直接报错；
- `projection`：使用无参数横向网格线投影，适合规整有框表格。

YOLO 只在缩小后的预览图上定位，检测框会映射回原图，最终切片始终从原图裁取，避免小字因预览缩放而丢失。

## 2. 准备图表

```bash
python main.py prepare-tables \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images" \
  --work-dir work/tables \
  --config afac_pipeline/table/config.example.json
```

该阶段不访问网络，输出：

```text
work/tables/
├── cache.sqlite3
├── dataset_manifest.json
└── prepared/<文件名_哈希>/
    ├── manifest.json
    ├── preview.png
    ├── preview_detected.png
    └── tiles/
```

`preview_detected.png` 用于快速检查漏表和错框；`manifest.json` 保存所有表格框、切片与原图坐标。

## 3. 切片原则

表格等比缩到 `max_vlm_side` 后如果仍能保留配置要求的分辨率，就整体送入模型；否则从原图裁成带重叠的二维网格。默认所有输出切片的最长边不超过 3900px。

## 4. 调用 FinixDoc-VL

```bash
export FINIXDOC_API_KEY="你的密钥"

python main.py run-tables \
  --manifest work/tables/dataset_manifest.json \
  --work-dir work/tables \
  --api-url "主办方提供的完整接口地址" \
  --model "FinixDoc-VL" \
  --output-csv outputs/table_submission.csv
```

公共客户端位于 `afac_pipeline/common/vlm_client.py`。每个切片响应写入 `responses/` 并进入 SQLite 缓存，中断后重跑不会重复请求成功切片。

## 5. Markdown 聚合与失败策略

二维切片先横向合并列，再纵向合并行；重叠区完全一致的行或列会去重。如果相邻切片的行列结构无法对应，程序会停止该图聚合，保存 `merge_error.json` 和原始响应并明确报错，避免静默输出结构错误的表格。

当前 A 榜图表目录 50 张图片中有 49 个 SHA-256 唯一文件，可少解析 1 张完全重复图片。

## 6. 校验

```bash
python afac_pipeline/table/tools/validate_prepared.py --manifest work/tables/dataset_manifest.json
python -m unittest discover -s tests -v
```
