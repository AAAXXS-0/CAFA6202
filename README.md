# AFAC 2026 文档解析工作流

当前仓库实现：

- 公共图片发现、元数据读取与 SHA-256 精确去重；
- 已由赛题分好的“图表目录”处理，不做自动路由；
- YOLO 版面检测与无参数横线投影检测；
- 检测框从缩略图映射回原图，再从原图裁切；
- 超大表格二维切片、切片级 API 缓存、Markdown 表格合并；
- FinixDoc-VL API 隔离适配器与赛事 CSV 输出；
- 长图 2048/1792 滑窗、general6 版面检测、标题层级分析与二次切块；
- 长图和图表由各自命令显式接收赛题目录，不做自动路由。

## 1. SHA-256 为什么能用于这里

SHA-256 会把文件的全部字节计算成一个 256 位摘要。两份文件只要有一个字节不同，摘要通常就会完全不同；不同文件意外得到同一摘要的概率可以忽略。因此：

- 摘要相同：可以认为图片文件字节完全一致，安全复用 Markdown；
- 图片重新压缩、修改元数据或改变一个像素：摘要会改变，不会复用；
- 它不是“看起来相似”的判断，不会误把两个相似费率表合并；
- 本项目不使用感知哈希，因为本赛题要求文字和表格精确一致，视觉相似不足以安全复用答案。

在当前 A 榜图表目录中，50 张图片有 49 个唯一文件，可少调用一次完整解析。

## 2. 安装

建议使用 Python 3.10～3.12。

```bash
python -m pip install -r requirements.txt
```

超大 PNG 强烈建议安装系统 libvips。Ubuntu 示例：

```bash
sudo apt-get install libvips42
```

如果封闭环境无法安装 libvips，可在配置中设为 `"backend": "pillow"`。Pillow 后端可以运行，但处理 2～3 亿像素图片时峰值内存会明显更高。

## 3. 第一步：检查精确重复图片

```bash
python main.py hash-report \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images"
```

这里只枚举真实图片扩展名，并排除 `:Zone.Identifier` 文件。

## 4. 第二步：检测并切分图表

先复制并修改配置：

```bash
cp config.example.json config.json
```

然后执行：

```bash
python main.py prepare-tables \
  --input-dir "raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images" \
  --work-dir work/tables \
  --config config.json
```

该命令不访问网络，只生成：

```text
work/tables/
├── cache.sqlite3
├── dataset_manifest.json
└── prepared/
    └── <文件名_哈希>/
        ├── manifest.json
        ├── preview.png
        ├── preview_detected.png
        └── tiles/
```

`preview_detected.png` 用于快速检查检测区域；`manifest.json` 保存每个表格框和切片的原图坐标，可以追查任何漏表、错切问题。

### 检测器选择

- `"detector": "auto"`：有 Ultralytics 和权重时用 YOLO，否则退回投影检测；
- `"detector": "yolo"`：强制使用 `report-8n.pt`，缺少依赖时直接报错；
- `"detector": "projection"`：只使用横向网格线投影，适合规整有框表格。

YOLO 只在最长边约 1800 的预览图上定位。最终切片始终从原图裁取，不会拿低分辨率预览图识别小字。

### 切片原则

中等表格若等比缩到 `max_vlm_side` 后仍保留至少 65% 分辨率，就整体送入模型；否则按原图切为带重叠的二维网格。所有输出切片最长边不超过配置值，默认 3900。

## 5. 第三步：调用 FinixDoc-VL

API Key 默认从环境变量读取：

```bash
export FINIXDOC_API_KEY="你的密钥"
```

执行：

```bash
python main.py run-tables \
  --manifest work/tables/dataset_manifest.json \
  --work-dir work/tables \
  --api-url "主办方提供的完整接口地址" \
  --model "FinixDoc-VL" \
  --output-csv outputs/table_submission.csv
```

当前 [afac_pipeline/vlm_client.py](afac_pipeline/vlm_client.py) 按常见 Chat Completions 图片消息格式实现。拿到主办方正式接口文档后，如果请求或响应字段不同，只修改这个适配器即可。

每个切片响应都会写入 `responses/` 并进入 SQLite 缓存。程序中断后重新运行不会重复请求已经成功的切片。

## 6. 表格聚合与失败策略

二维切片先横向合并列，再纵向合并行。重叠区出现完全一致的行或列时会去重。如果相邻切片输出的行数或列数无法对应，程序会：

1. 停止该图片的聚合；
2. 写出 `merge_error.json` 和全部原始响应；
3. 明确报错，不静默生成结构错误的表格。

这是有意的保守策略：错误拼接会同时损害文本编辑距离和 TEDS，显式失败更便于调整切片或 Prompt 后重跑。

## 7. 长图分支

长图实现位于 afac_pipeline/long_pipeline.py，使用 general6-8n.pt，不依赖 Toc 标签。完整标题规则、目录结构、运行命令和校验方式见 README_LONG.md。图表与长图分支保持隔离，不会互相套用切块逻辑。

## 8. 测试

测试只依赖标准库、Pillow 和 NumPy：

```bash
python -m unittest discover -s tests -v
```

覆盖内容包括精确哈希分组、图表聚合、长图滑窗、标题层级、语义覆盖、缓存和 CSV 输出。
