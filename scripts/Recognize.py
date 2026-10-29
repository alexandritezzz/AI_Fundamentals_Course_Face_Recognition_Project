import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple
from Controller import controller
from Extract import Get_Test_Embeddings

recognize_control = controller()


def Load(Data_Path: str = recognize_control.getDataPath()) -> Optional[Dict[str, np.ndarray]]:

    try:
        with open(Data_Path, "rb") as File:
            Data: Dict[str, np.ndarray] = pickle.load(File)
        return Data
    except:
        return None


# 核心检索函数。输入图片路径，身份 - 特征向量字典和判断阈值（置信度低于阈值时，匹配的人物身份返回"Unknown."），
# 返回[（匹配的人物身份，匹配置信度，脸部坐标框）]列表。
def Face_Recognize(Image_Path: str, Data: Dict[str, np.ndarray], Threshold: float = 0.1) \
                   -> List[Tuple[str, float, Optional[List[float]]]]:

    Tuples = Get_Test_Embeddings(str(Image_Path))

    if not Tuples: return []

    Results = []

    for Query, _, BBox in Tuples:
        Norm = np.linalg.norm(Query)
        Query = Query / Norm

        Match_Identity = "Unknown."
        Match_Score = -1.0

        for Identity, Vector in Data.items():
            Score = np.dot(Query, Vector)

            if Score > Match_Score:
                Match_Score = float(Score)
                Match_Identity = Identity

        if Match_Score >= Threshold:
            Results.append((Match_Identity, Match_Score, BBox))
        else: Results.append(("Unknown.", Match_Score, BBox))
    
    return Results


# 交换遍历内外层的另一种实现，仅供参考
# 不影响测试，但是会导致前端脚本出现异常行为
# def Face_Recognize(Image_Path: str, Data: Dict[str, np.ndarray],\
#                    Threshold: float = 0.1) -> List[Tuple[str, float, Optional[List[float]]]]:
#         """识别人脸（计算其余弦相似度）"""

#         Embeddings = Get_Test_Embeddings(str(Image_Path))

#         if Embeddings is None:
#             return []

#         Results = []

#         # 对于每个人脸信息
#         for identity, Vec in Data.items():
#             best_similarity = -1
#             best_identity = "unknown"

#             # 对于提取的每个待测人脸
#             for embedding, _, BBox in Embeddings:
#                 # # 计算余弦相似度
#                 Norm = np.linalg.norm(embedding)
#                 embedding = embedding / Norm
#                 similarity = np.dot(embedding, Vec)

#                 if similarity > best_similarity:
#                     best_similarity = similarity
#                     best_identity = identity
            
#             if best_similarity >= Threshold:
#                 Results.append((best_identity, best_similarity, BBox))

#         Results.append(('Unknown.', best_similarity, BBox))
#         return Results