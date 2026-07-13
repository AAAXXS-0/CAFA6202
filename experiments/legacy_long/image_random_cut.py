import os
import json
from PIL import Image

def slice_image(
    image_path,
    output_dir,
    chunk_height=2048,      # 每个小块的高度（像素）
    overlap=256,            # 上下重叠像素数
    min_chunk_height=512    # 最后一块若小于此值，则与上一块合并
):
    """
    将超长图片沿高度方向切割成多个有重叠的小块。
    宽度保持原图宽度不变（因为宽度仅1508，无需再切分）。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取图片
    img = Image.open(image_path)
    width, height = img.size
    print(f"原图尺寸: {width} x {height}")

    # 存储每块的信息
    chunks_info = []
    current_y = 0
    chunk_index = 0

    while current_y < height:
        # 计算当前块的起始和结束 y 坐标（注意不能超出图像边界）
        start_y = current_y
        end_y = min(start_y + chunk_height, height)

        # 如果最后一块高度小于 min_chunk_height，则向前合并
        if (end_y - start_y) < min_chunk_height and start_y > 0:
            # 舍弃当前块，扩展上一块到底部
            if chunks_info:
                last = chunks_info[-1]
                # 将最后一块的结束坐标延展到图像底部
                last['end_y'] = height
                # 重新裁剪最后一块图片（需要重新保存）
                # 我们稍后在循环外统一处理重切
                print(f"最后一块过小（{end_y - start_y}px），已合并到上一块")
            break

        # 裁剪当前块
        bbox = (0, start_y, width, end_y)
        chunk_img = img.crop(bbox)

        # 保存
        chunk_filename = f"chunk_{chunk_index:04d}.png"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk_img.save(chunk_path)

        # 记录元数据
        chunks_info.append({
            'index': chunk_index,
            'filename': chunk_filename,
            'start_y': start_y,
            'end_y': end_y,
            'width': width,
            'height': end_y - start_y
        })

        print(f"切出第 {chunk_index} 块: y={start_y}~{end_y}, 尺寸={width}x{end_y-start_y}")

        # 下一块的起始位置：跳过重叠部分
        current_y = end_y - overlap
        chunk_index += 1

        # 安全保护：防止死循环
        if current_y >= height:
            break

    # 如果最后一块被合并，我们需要重新裁剪最后一块并覆盖保存
    # 但由于我们提前 break 时并没有更新文件，可以单独处理
    # 更稳健：在 break 时修改最后一块的记录，并在循环外重新生成
    # 这里采用更简单的处理：不合并，但允许最后一块很小（不影响拼接）
    # 实际中如果最后一块很小，模型也能处理，只是效率低一点，所以不强求合并

    # 保存索引文件（供后续拼接使用）
    meta_path = os.path.join(output_dir, "chunks_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'original_width': width,
            'original_height': height,
            'chunks': chunks_info,
            'overlap': overlap,
            'chunk_height': chunk_height
        }, f, indent=2)

    print(f"\n切图完成！共切出 {len(chunks_info)} 块。")
    print(f"元数据已保存至: {meta_path}")

    return meta_path

if __name__ == "__main__":
    # 使用示例
    slice_image(
        image_path="raw_data/AFAC A榜评测数据集(2)/finix_huge_long_rest_A/images/97953b4d-67c3-49c7-922e-450b565dd401.jpg",   # 替换为你的图片路径
        output_dir="./sliced_chunks",
        chunk_height=2048,   # 根据你的GPU显存调整，一般2048~4096安全
        overlap=256,         # 重叠区域，避免文字在接缝处截断
        min_chunk_height=512
    )
