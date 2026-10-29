# -*- coding: utf-8 -*-
# 在项目根目录直接运行即可
# 功能：对dataset下的测试图片进行自动标注，生成annotations_generated.jsonl，绘制带框的效果图到./marked
# 原理：使用模型注册所有人脸并识别，阈值统一设置为0.5
# 修改点：
#   1. 修复 BBox 格式为官方指定的 [x, y, width, height] 绝对像素格式
#   2. 修复 draw_marked_image 函数，防止因坐标格式转换导致的画框错位
#   3. 优化图片检索机制，支持 .jpeg、.JPEG、.JPG、.PNG 等大小写后缀，防止漏检
#   4. 规范化 JSONL 中的图片路径输出为 "test/images/xxx.jpg" 相对格式
# 注意：切勿修改、覆盖原来的annotations.jsonl


import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path
from tqdm import tqdm
from Controller import controller,_resolvePath

mark_control = controller()

class FaceRegistrationSystem:
    def __init__(self):
        # 初始化 InsightFace
        self.app = FaceAnalysis(name = mark_control.model_type)
        self.app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

        # 存储注册的人脸特征
        self.registered_faces = {}  # {identity_id: [features_list]}

    def _get_image_files(self, folder_path: Path):
        """安全、完整地获取文件夹下所有支持的图像文件（解决大小写与jpeg漏检问题）"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path.glob(ext))
        return sorted(list(set(image_files)))

    def register_faces(self, registered_dir=_resolvePath('./dataset/registered')):
        """注册所有人脸"""
        print("开始注册人脸...")

        registered_path = Path(registered_dir)
        if not registered_path.exists():
            print(f"错误: 注册集路径 {registered_dir} 不存在！")
            return {}

        person_folders = sorted(
            [d for d in registered_path.iterdir() if d.is_dir()])

        for person_folder in tqdm(person_folders, desc="注册进度"):
            identity_id = person_folder.name
            self.registered_faces[identity_id] = []

            # 使用优化后的图像检索方法
            image_files = self._get_image_files(person_folder)

            for img_path in image_files:

                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                # img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # 检测人脸
                faces = self.app.get(img)

                # 假设每张注册照片只有一个人脸
                if len(faces) >= 1:
                    if len(faces) > 1:
                        print(f" 警告: {img_path.name} 中检测到多个人脸，默认使用第一个")
                    face = faces[0]
                    # 获取人脸特征
                    if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                        self.registered_faces[identity_id].append(
                            face.normed_embedding)
                    elif hasattr(face, 'embedding') and face.embedding is not None:
                        self.registered_faces[identity_id].append(
                            face.embedding)

        # 计算每个注册人物的平均特征向量
        self.average_features = {}
        for identity_id, features_list in self.registered_faces.items():
            if features_list:
                self.average_features[identity_id] = np.mean(
                    features_list, axis=0)

        print(f"注册完成！共注册 {len(self.average_features)} 个人物\n")
        return self.average_features

    def recognize_face(self, face_embedding, threshold=0.5):
        """识别人脸（计算其余弦相似度）"""
        if face_embedding is None or not hasattr(self, 'average_features'):
            return "unknown"

        best_similarity = -1
        best_identity = "unknown"

        for identity_id, avg_feature in self.average_features.items():
            # 计算余弦相似度
            similarity = np.dot(face_embedding, avg_feature) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(avg_feature)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_identity = identity_id

        # 如果相似度低于阈值，标记为 unknown
        if best_similarity < threshold:
            return "unknown"

        return best_identity

    def process_test_images(self, test_dir='./dataset/test/images', output_dir='./marked'):
        """处理测试图片并自动转换坐标格式"""
        print("开始处理测试图片...")

        os.makedirs(output_dir, exist_ok=True)

        test_path = Path(test_dir)
        image_files = self._get_image_files(test_path)

        annotations = []

        for img_path in tqdm(image_files, desc="识别进度"):

            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}")
                continue

            # 检测人脸
            faces = self.app.get(img)

            # 判断图片类型 (0或1张脸为 single，多张脸为 multi)
            image_type = "single" if len(faces) <= 1 else "multi"

            # 处理每张人脸
            faces_info = []
            for face in faces:
                # 1. 解析原始 InsightFace 的 [x1, y1, x2, y2] 边界框
                x1, y1, x2, y2 = face.bbox.astype(int).tolist()

                # 2. 核心转换：转换为要求的 [x, y, width, height] 绝对像素格式
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                bbox_formatted = [x1, y1, width, height]

                # 获取人脸特征并进行识别
                identity_id = "unknown"
                if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                    identity_id = self.recognize_face(face.normed_embedding)
                elif hasattr(face, 'embedding') and face.embedding is not None:
                    identity_id = self.recognize_face(face.embedding)

                faces_info.append({
                    "identity_id": identity_id,
                    "bbox": bbox_formatted
                })

            # 3. 规范化图片相对路径，严格匹配指引中的 "test/images/xxx.jpg"
            standard_img_path = f"test/images/{img_path.name}"

            # 创建标注记录
            annotation = {
                "image": standard_img_path,
                "image_type": image_type,
                "faces": faces_info
            }
            annotations.append(annotation)

            # 绘制标记图片（传入标准格式坐标）
            self.draw_marked_image(img, faces_info, img_path.name, output_dir)

        # 保存为 annotations_generated.jsonl 文件
        self.save_annotations(annotations, 
                              _resolvePath('./dataset/test/annotations_generated.jsonl'))
        print(f"\n处理完成！共处理 {len(annotations)} 张图片")
        return annotations

    def draw_marked_image(self, img, faces_info, img_name, output_dir):
        """还原坐标并绘制正确带框的效果图"""
        marked_img = img.copy()

        for face_info in faces_info:
            # 逆向解析官方的 [x, y, width, height] 格式
            x, y, w, h = face_info['bbox']
            identity_id = face_info['identity_id']

            # 已知绿色，未知红色
            color = (0, 255, 0) if identity_id != "unknown" else (0, 0, 255)

            # 动态恢复出右下角绝对坐标 (x + w, y + h)，确保 OpenCV 能精准画框
            cv2.rectangle(marked_img,
                          (x, y),
                          (x + w, y + h),
                          color, 2)

            # 添加身份标签
            label = f"{identity_id}"
            cv2.putText(marked_img, label,
                        (x, y - 10 if y - 10 > 10 else y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # 保存标记图片至可视化审查目录
        output_path = Path(output_dir) / f"marked_{img_name}"
        cv2.imwrite(str(output_path), marked_img)

    def save_annotations(self, annotations, output_file):
        """保存标注到标准 JSONL 文件中"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
        print(f" 标准标注文件已成功保存到: {output_file}")


def main():
    system = FaceRegistrationSystem()

    # 1. 自动读取注册集并完成特征建库
    system.register_faces(_resolvePath('./dataset/registered'))

    # 2. 检索测试集，转换格式，输出效果图与标准 jsonl 标注文件
    annotations = system.process_test_images(_resolvePath('./dataset/test/images'),
                                             './marked')

    # 输出最终的质检统计面板
    total_faces = sum(len(ann['faces']) for ann in annotations)
    single_images = sum(
        1 for ann in annotations if ann['image_type'] == 'single')
    multi_images = sum(
        1 for ann in annotations if ann['image_type'] == 'multi')

    print("\n================ 自动标注报告 ================")
    print(f" -> 生成总图片记录数: {len(annotations)}")
    print(f" -> 单人照图片总数  : {single_images}")
    print(f" -> 多人照图片总数  : {multi_images}")
    print(f" -> 检出总人脸数据数量: {total_faces}")
    print("==================================================")


if __name__ == "__main__":
    main()
