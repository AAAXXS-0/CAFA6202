from ultralytics import YOLO

# image_path = 'raw_data/AFAC A榜评测数据集(2)/finix_huge_table_rest_A/images/0cd74f08-df57-421d-923e-3fa3f1d017c1.jpg'  # 待预测图片路径
# model_path = '360LayoutAnalysis/general6-8n.pt'  # 权重路径
# model = YOLO(model_path)

# result = model(image_path, save=True, conf=0.5, save_crop=False, line_width=2)
# print(result)

# print(result[0].names)         # 输出id2label map
# print(result[0].boxes)         # 输出所有的检测到的bounding box
# print(result[0].boxes.xyxy)    # 输出所有的检测到的bounding box的左上和右下坐标
# print(result[0].boxes.cls)     # 输出所有的检测到的bounding box类别对应的id
# print(result[0].boxes.conf)    # 输出所有的检测到的bounding box的置信度

from read_all_random_pieces import read_and_sort

sort_paths=read_and_sort()
# print(sort_paths)

for img_path in sort_paths:
    image_path = img_path  # 待预测图片路径
    model_path = '360LayoutAnalysis/general6-8n.pt'  # 权重路径
    model = YOLO(model_path)

    result = model(image_path, save=True, conf=0.5, save_crop=False, line_width=2)
    # print(result[0].names)         # 输出id2label map
    # print(result[0].boxes)         # 输出所有的检测到的bounding box
    # print(result[0].boxes.xyxy)    # 输出所有的检测到的bounding box的左上和右下坐标
    # print(result[0].boxes.cls)     # 输出所有的检测到的bounding box类别对应的id
    # print(result[0].boxes.conf)    # 输出所有的检测到的bounding box的置信度
    print(result)
