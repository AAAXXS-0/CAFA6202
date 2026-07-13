import re
from pathlib import Path

def read_and_sort(path="./sliced_chunks") ->list:
    def extract_number(filepath: Path) -> int:
        """
        从文件名中提取下划线后的数字，例如 'chunk_0020.png' -> 20
        如果没有匹配到，返回一个极大值，排到最后
        """
        # 匹配 下划线 + 数字 + 点（扩展名之前）
        match = re.search(r'_(\d+)\.', filepath.name)
        if match:
            return int(match.group(1))
        return 999999  # 没有数字的放到末尾

    # 你的图片文件夹路径（例如 'sliced_chunks'）
    folder = Path(path)  # 请修改为实际路径

    # 支持的图片格式（按需增减）
    extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    # 1. 获取文件夹内所有图片（仅当前目录，不递归子文件夹）
    image_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    # 2. 按提取的数字升序排序（0000 → 0001 → 0002 ...）
    sorted_files = sorted(image_files, key=extract_number)

    # print(sorted_files)

    # # 3. 有序遍历处理
    # for img_path in sorted_files:
    #     print(img_path)  # 查看顺序
    return sorted_files
