import cv2
from insightface.app import FaceAnalysis
import numpy as np
from typing import Any, List, Optional, Tuple
from Controller import controller

extract_control = controller()

Application = FaceAnalysis(name = extract_control.model_type)
Application.prepare(ctx_id = 0, det_size = (640, 640), det_thresh = 0.5)

Weak_Application = FaceAnalysis(name = extract_control.model_type)
Weak_Application.prepare(ctx_id = 0, det_size = (640, 640), det_thresh = 0.2)


def Illumination_Control(Image: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    图像控制：平均亮度在
    [0,80):     2 (使用CLAHE增强亮度)
    [80,210]:   1
    (210,inf):  0
    """
    Gray_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    Mean_Brightness = np.mean(Gray_Image)
    if Mean_Brightness > 210:
        return Image, 0
    elif Mean_Brightness >= 80:
        return Image, 1
    else:
        LAB_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(LAB_Image)
        Clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
        CL = Clahe.apply(L)
        Enhanced_LAB_Image = cv2.merge((CL, A, B))
        Enhanced_Image = cv2.cvtColor(Enhanced_LAB_Image, cv2.COLOR_LAB2BGR)
        return Enhanced_Image, 2


# 返回找到的脸（Insightface对象，Any）和（可能）经Illumination_Control处理后的图像（np.ndarray）。
def Get_Primitive_Face(Image: Optional[np.ndarray], Illumination_Check: bool = False) -> Tuple[List[Any], Optional[np.ndarray]]:
    """
    可选是否进行光照检查
    先用高阈值检测，失败则用低阈值检测
    返回检测到的人脸列表和处理后的图像
    """
    if Image is None: return [], None

    if Illumination_Check:
        Image, Illumination_Judge = Illumination_Control(Image)
        #print(Illumination_Judge)
        if Illumination_Judge == 0: return [], None

    Faces = Application.get(Image)

    if len(Faces) == 0:
        #print("Face Undetected at Det_Thresh = 0.5.")
        Faces = Weak_Application.get(Image)
        if len(Faces) == 0:
            #print("Face Undetected at Det_Thresh = 0.2.")
            return [], None

    return Faces, Image


def Validation(Face: Any, Image: np.ndarray) -> int:
    """
    有效性：
    1: 检测分数 < 0.6
    2: 人脸框宽或高 < 60 像素（太小）
    3: 裁剪后人脸区域为空
    4: Laplacian 方差 < 100（图像模糊）
    """
    if Face.det_score < 0.6:
        return 1

    BBox = Face.bbox

    Width = int(BBox[2] - BBox[0])
    Height = int(BBox[3] - BBox[1])

    if Width < 60 or Height < 60:
        return 2

    Y1, Y2 = max(0, int(BBox[1])), min(Image.shape[0], int(BBox[3]))
    X1, X2 = max(0, int(BBox[0])), min(Image.shape[1], int(BBox[2]))
    Face_Area = Image[Y1: Y2, X1: X2]

    if Face_Area.size == 0:
        return 3

    if cv2.Laplacian(cv2.cvtColor(Face_Area, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 100:
        return 4

    return 0


# 建库时使用的Embedding提取，输入文件路径，返回（特征向量，置信度）元组。
# 提取失败时返回（None，-1.0）。
def Get_Embedding(Image_Path: str) -> Tuple[Optional[np.ndarray], float]:

    Image = cv2.imdecode(np.fromfile(Image_Path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Image = cv2.imread(Image_Path)

    if Image is None: return None, -1.0

    Faces, Image = Get_Primitive_Face(Image, Illumination_Check = True)

    if not Faces: return None, -1.0

    Face = Faces[0]

    Validation_Score = Validation(Face, Image)

    #print(Validation_Score)

    if Validation_Score == 0:
        return Face.embedding, float(Face.det_score)
    else: return None, -1.0


# 检索时使用的Embedding提取，输入文件路径，返回[（特征向量，置信度，脸部坐标框）]列表
# 提取失败时返回空列表。
def Get_Test_Embeddings(Image_Path: str) -> List[Tuple[Optional[np.ndarray], float, Optional[List[float]]]]:

    Image = cv2.imdecode(np.fromfile(Image_Path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # Image = cv2.imread(Image_Path)
    if Image is None: return []

    Faces, Image = Get_Primitive_Face(Image)
    if not Faces: return []

    # if len(Faces) == 1:
    #     Face_0 = Faces[0]

    #     Faces_1, _ = Get_Primitive_Face(cv2.flip(Image, 1))

    #     if not Faces_1:
    #         return [(Face_0.embedding, float(Face_0.det_score), Face_0.bbox.tolist())]

    #     Face_1 = Faces_1[0]

    #     Fusion_Vertex = Face_0.embedding + Face_1.embedding
    #     return [(Fusion_Vertex / np.linalg.norm(Fusion_Vertex), float((Face_0.det_score + Face_1.det_score) / 2.0), Face_0.bbox.tolist())]
    # else:
    #     Results = []
    #     for Face in Faces:
    #         Results.append((Face.embedding, float(Face.det_score), Face.bbox.tolist()))

    #     return Results

    # 另一个版本
    Faces = Application.get(Image)
    Faces_info = []
    for face in Faces:
        Faces_info.append((face.embedding, float(face.det_score), face.bbox.astype(int).tolist()))
    return Faces_info


if __name__ == "__main__":
    # Test_Image_Path = 'Provided_Dataset/Register/identity_07684/097810.jpg'
    Test_Image_Path = 'dataset/registered/p01/p01_r01.jpg'
    
    Vector, _ = Get_Embedding(Test_Image_Path)

    if Vector is not None:
        print(Vector.shape)
        print(Vector[: 5])
