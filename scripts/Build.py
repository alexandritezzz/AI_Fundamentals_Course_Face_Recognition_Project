# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List

from Extract import Get_Embedding
from Controller import controller

# 输入注册集的路径和是否使用自建的20人注册集（默认为False，否则使用教师提供的100人注册集），
# 以此分别生成"Face_Data_Provided.pkl"和"Face_Data_Customized.pkl"，返回None。
# 需要注意的是，不管是哪种情况，都需要自行提供注册集路径。
# 最后的布尔值参数只负责调整提取编号的逻辑与生成数据集的名称。

build_control = controller()

def Build(Register: str, Is_Customized_Dataset: bool = build_control.Is_Customized_Dataset) -> None:
    Direction = Path(Register)

    if not Direction.exists():
        return

    Data: Dict[str, np.ndarray] = {}

    for Folder in Direction.iterdir():
        if not Folder.is_dir():
            continue

        if Is_Customized_Dataset:
            Identity = Folder.name
            print(f"Processing Customized Folder: identity_{Identity}.")

        else:
            Identity = Folder.name[9:]
            print(f"Processing Provided Folder: {Identity}.")

        Vectors: List[np.ndarray] = []
        Weights: List[float] = []

        for Image_Path in Folder.glob("*.jpg"):
            Vector, Weight = Get_Embedding(str(Image_Path))

            if Vector is not None:
                Vectors.append(Vector)
                Weights.append(Weight)

        if len(Vectors) > 0:
            Vectors_Matrix = np.array(Vectors)
            Weights_Array = np.array(Weights).reshape(-1, 1)

            Fusion = np.sum(Vectors_Matrix * Weights_Array,
                            axis=0) / np.sum(Weights_Array)

            Norm = np.linalg.norm(Fusion)

            Identity_Vector = Fusion / Norm

            Data[Identity] = Identity_Vector

    Data_Path = build_control.getDataPath()
    with open(Data_Path, "wb") as File:
        pickle.dump(Data, File)


if __name__ == "__main__":
    
    Reg_Path = build_control.getRegPath()
    Build(Reg_Path, build_control.Is_Customized_Dataset)
