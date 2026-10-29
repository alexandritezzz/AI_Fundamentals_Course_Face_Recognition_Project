# -*- coding: utf-8 -*-
# 功能：自动化评估脚本。支持一键切换测试 100人CelebA数据集 或 自建20人数据集。
# 6.11：
#   1. 修复元组解包错误：循环解析 Face_Recognize 返回的 List 结果，兼容单人与多人识别
#   2. 自动兼容 CelebA 的子文件夹结构与自建集的目录结构
#   3. 布尔值 Is_Customized_Dataset 现在能联动控制特征字典与测试路径

import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
import time
from Controller import controller
from Recognize import Load, Face_Recognize


test_control = controller()


def Evaluation(Test: str, Data: Dict[str, np.ndarray], Threshold: float = 0.2, Bad_Cases_Check: bool = False, \
               Is_Customized_Dataset: bool = test_control.Is_Customized_Dataset) -> Tuple[float, List[int]]:
    
    Direction = Path(Test)

    Total = 0
    Correct = 0
    Unknown = 0
    Wrong = 0

    image_tasks = []

    # 1. 智能目录解析层
    if Is_Customized_Dataset:
        images_dir = Direction / \
            "images" if (Direction / "images").exists() else Direction

        subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
        if subdirs:
            for folder in subdirs:
                true_id = folder.name
                for img_path in folder.glob("*.jpg"):
                    image_tasks.append((img_path, true_id))
        else:
            for img_path in images_dir.glob("*.jpg"):
                true_id = img_path.name.split('_')[0]
                image_tasks.append((img_path, true_id))
    else:
        if not Direction.exists():
            print(f"错误: 教师数据集路径 {Test} 不存在！")
            return 0.0, [0, 0, 0, 0]
        for folder in Direction.iterdir():
            if not folder.is_dir():
                continue
            true_id = folder.name[9:]  # 裁剪掉 "identity_" 前缀，提取纯编号
            for img_path in folder.glob("*.jpg"):
                image_tasks.append((img_path, true_id))

    # 2. 核心评估循环层
    for Image_Path, True_Identity in image_tasks:
        Total = Total + 1

        # 调用核心检索函数，返回的是一个列表：List[Tuple[Identity, Score, BBox]]
        Tuples = Face_Recognize(str(Image_Path), Data, Threshold)

        # 情况 A：整张图连一个人脸都没检测出来，直接记入 Unknown
        if not Tuples:
            Unknown = Unknown + 1
            if Bad_Cases_Check:
                Name = f"True_{True_Identity}_Score_0.00_{Image_Path.name}"
                shutil.copy(str(Image_Path), Path(f"{Bad_Cases_Path}/Unknown") / Name)
            continue

        # 提取图片中所有人脸被系统赋予的预测身份
        Predicted_Identities = [t[0] for t in Tuples]
        # 提取其中最高匹配得分，用于 Bad Cases 命名时的参考
        Max_Score = max([float(t[1]) for t in Tuples])

        # 多人照识别兼容判定业务逻辑：
        # 1. 只要图片中成功认出了目标名人（无论它在主位还是背景），即算整张图 Correct
        if True_Identity in Predicted_Identities:
            Correct = Correct + 1

        # 2. 如果检测到了脸，但所有人脸都被判定为 "Unknown."，计入 Unknown
        elif all(pid == "Unknown." for pid in Predicted_Identities):
            Unknown = Unknown + 1
            if Bad_Cases_Check:
                Name = f"True_{True_Identity}_Score_{Max_Score:.2f}_{Image_Path.name}"
                shutil.copy(str(Image_Path), Path(f"{Bad_Cases_Path}/Unknown") / Name)

        # 3. 系统没认出目标名人，且把其中某张脸错认成了数据库里的其他名人，计入 Wrong
        else:
            Wrong = Wrong + 1
            if Bad_Cases_Check:
                # 第一个错误名字
                wrong_ids = [pid for pid in Predicted_Identities if pid !=
                             "Unknown." and pid != True_Identity]
                wrong_id = wrong_ids[0] if wrong_ids else "Unknown"
                Name = f"True_{True_Identity}_Predict_{wrong_id}_Score_{Max_Score:.2f}_{Image_Path.name}"
                shutil.copy(str(Image_Path), Path(f"{Bad_Cases_Path}/Wrong") / Name)

    Accuracy = Correct / Total if Total > 0 else 0
    return Accuracy, [Total, Correct, Unknown, Wrong]


if __name__ == "__main__":
    
    # 让生成的Bad_Cases带有时间后缀
    now = time.localtime(time.time())
    Bad_Cases_Path = f'Bad_Cases_{now[1]:02}{now[2]:02}{now[3]:02}{now[4]:02}'
    Path(f"{Bad_Cases_Path}/Unknown").mkdir(parents=True, exist_ok=True)
    Path(f"{Bad_Cases_Path}/Wrong").mkdir(parents=True, exist_ok=True)

    # True : 测试自建的 20人 数据集
    # False: 测试教师提供的 100人 CelebA 数据集
    # 参见 Controller.py
    Is_Customized_Dataset = test_control.Is_Customized_Dataset
    Test_Dir = test_control.getTestPath()
    Data_Path = test_control.getDataPath()
    Reg_Path = test_control.getRegPath()

    try:
        Data = Load(Data_Path)
        if Data is None:
            raise BaseException(f'Data {Data_Path} Not Found')
    except BaseException:
        print(f"\n错误：未能加载特质库 {Data_Path}，自动尝试运行 Build\n")
        
        import Build
        Build.Build(Reg_Path, Is_Customized_Dataset)
        Data = Load(Data_Path)
        if Data is None:
            print(f"错误：未能加载到特质库 {Data_Path}，且自动尝试失败，请检查问题")
            exit()

    print(f"\n----- Testing at Dataset: {'Custom 20' if Is_Customized_Dataset else 'Provided 100'} -----\n")

    # 阈值：剔除置信度低于设定值的结果，阈值越低越容易输出结果，但是太低（0.1以下）会导致错判
    # 相当于让模型对输出根据自信成度进行二分类，低于阈值一律打成 Unknown
    # 自定义20人数据集相对来说要简单一些，模型给出的置信度较高
    Thresholds = [0.3, 0.5, 0.6] if Is_Customized_Dataset else [0.1, 0.2, 0.3, 0.5]
    Bad_Cases_Threshold = max(Thresholds)

    for Threshold in Thresholds:

        Accuracy, Score = Evaluation(
            Test_Dir, Data, Threshold, 
            Bad_Cases_Check = True if Threshold == Bad_Cases_Threshold else False, 
            Is_Customized_Dataset = Is_Customized_Dataset)

        print(f"----- Threshold: {Threshold} -----")
        print(f"Accuracy : {Accuracy * 100:.2f}%")
        print(f"Total    : {Score[0]}")
        print(f"Correct  : {Score[1]}")
        print(f"Unknown  : {Score[2]}")
        print(f"Wrong    : {Score[3]}\n")

    print(f"----- Bad Cases Folder with Threshold = {Bad_Cases_Threshold} Generated. -----\n")
    print("----- Test Completed Successfully. -----\n")
    input('按任意按键退出...\n')
