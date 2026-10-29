import json
import cv2
import os
import numpy as np
from Controller import _resolvePath

# 创建输出目录
os.makedirs('./marked_imgs_for_test', exist_ok=True)

# 读取标注文件
filePath = _resolvePath('dataset/test/annotations.jsonl')

with open(filePath, 'r') as f:
    for line in f:
        data = json.loads(line.strip())

        # 读取原图
        img = cv2.imdecode(np.fromfile(_resolvePath(f"dataset/{data['image']}"), dtype=np.uint8), 
                           cv2.IMREAD_COLOR)

        # 为unknown和已知身份设置不同颜色
        for face in data['faces']:
            x, y, w, h = face['bbox']
            identity_id = face['identity_id']

            # 已知绿色，未知红色
            color = (0, 255, 0) if identity_id != "unknown" else (0, 0, 255)

            # 恢复右下角绝对坐标 (x + w, y + h)，确保 OpenCV 能精准画框
            cv2.rectangle(img,
                          (x, y),
                          (x + w, y + h),
                          color, 2)

            # 添加身份标签
            label = f"{identity_id}"
            cv2.putText(img, label,
                        (x, y - 10 if y - 10 > 10 else y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # 保存结果
        output_name = f"./marked_imgs_for_test/annotated_{os.path.basename(data['image'])}"
        cv2.imwrite(output_name, img)
        print(f"已保存: {output_name}")
