# AFAC 2026 文档解析工作流

本仓库按赛题已经分好的数据目录，分别处理长图文档和图表图片；不做自动路由。两个分支共享精确去重、图像读取、缓存、FinixDoc-VL 客户端和提交文件生成，但检测、切块与 Markdown 聚合逻辑彼此独立。

## 项目结构

```text
AFAC2026_challger/
├── afac_pipeline/
│   ├── common/                 # SHA-256、图像后端、缓存、VLM 客户端、CSV
│   ├── long/                   # 长图代码、配置示例和分支 README
│   └── table/                  # 图表代码、配置、文档和分支工具
├── experiments/legacy_long/   # 早期随机切图和模型试验脚本，仅供追溯
├── tests/                     # 自动化测试
├── main.py                    # 统一命令行入口
├── requirements.txt
└── 赛题.txt
```

分支文档：

- [长图分支](afac_pipeline/long/README.md)：2048/1792 滑窗、general6 检测、标题层级、二次切块和请求打包。
- [图表分支](afac_pipeline/table/README.md)：表格检测、缩放/二维切片、Markdown 表格拼接和失败处理。

## 安装

建议使用 Python 3.10～3.12：

```bash
python -m pip install -r requirements.txt
```

超大图片建议安装系统 `libvips`，否则会自动使用 Pillow，功能不受影响但峰值内存更高。

## 统一命令行

```bash
python main.py --help
```

当前命令包括：

- `hash-report`：统计字节完全相同的图片；
- `prepare-tables` / `run-tables`：准备、识别图表目录；
- `prepare-long` / `run-long`：准备、识别长图目录。

配置示例分别位于：

- `afac_pipeline/table/config.example.json`
- `afac_pipeline/long/config.example.json`

## SHA-256 去重说明

SHA-256 根据文件的全部字节生成摘要。摘要相同可以视为文件字节完全相同，因此能够安全复用解析结果；重新压缩、修改元数据或改变一个像素都会产生不同摘要。它不判断“视觉相似”，不会把内容接近但文字不同的表格误合并。

当前 A 榜数据实测：图表 50 张中 49 张唯一图；长图 50 张中 33 张唯一图，后者可直接复用 17 张的完整结果。

## 测试

```bash
python -m compileall -q afac_pipeline main.py tests
python -m unittest discover -s tests -v
```

测试覆盖精确哈希、图表检测和聚合、长图滑窗、标题层级、语义覆盖、请求打包、缓存和 CSV 输出。
