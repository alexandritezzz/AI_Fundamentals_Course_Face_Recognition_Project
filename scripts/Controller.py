# 集中控制项目相关参数
# 已经嵌入到各个脚本

# 使用例：
# from Controller import controller
# example_control = controller()
# Is_Customized_Dataset = example_controll.Is_Customized_Dataset

# 用法1：全局默认参数，直接在此修改
# 用法2：代码内修改，建议调用类方法

from pathlib import Path

class controller:

    # 可修改
    # True : 测试自建的 20人 数据集
    # False: 测试教师提供的 100人 CelebA 数据集
    Is_Customized_Dataset : bool = True


    # 可修改
    # 请检查拼写，模型名与给出选项不匹配将导致无法运行
    # 可选项：'buffalo_s', 'buffalo_l'
    # 'buffalo_s'：轻量、快速、均衡
    # 'buffalo_l'：更大、更慢、更准
    model_type : str = 'buffalo_l'


    def setModel(self, model: str = 'buffalo_s') -> None:
        """def setModel(self, model='buffalo_s') -> None\n
        设置模型类型\n
        可选项：\n
        buffalo_s（较小的模型）\n
        buffalo_l（较大的）
        """
        self.model_type = model

    def setDatasetType(self, dataset_type: bool = True) -> None:
        '''def setDatasetType(self, dataset_type=True) -> None\n
        设置数据集类型\n
        可选项：\n
        True：自定义20人数据集\n
        False：提供的100人数据集\n
        '''
        self.Is_Customized_Dataset = dataset_type

    def getDataPath(self) -> str:
        '''返回入库数据的路径'''
        data_path = f"Face_Data_Customized_{controller.model_type}.pkl" \
        if self.Is_Customized_Dataset else f"Face_Data_Provided_{controller.model_type}.pkl"

        return _resolvePath(data_path)

    
    def getRegPath(self) -> str:
        '''返回注册照片文件夹的路径'''
        reg_path = 'dataset/registered' if self.Is_Customized_Dataset \
            else 'Provided_Dataset/Register'

        return _resolvePath(reg_path)
    
    def getTestPath(self) -> str:
        '''返回测试照片文件夹的路径'''
        test_path = "dataset/test" if self.Is_Customized_Dataset else "Provided_Dataset/Test"

        return _resolvePath(test_path)
        

def _resolvePath(path : str):
        '''辅助函数，确保路径正确'''
        project_path = Path(__file__).parent.parent
        if Path.cwd() != project_path:
            return str( (project_path / Path(path)) )
        else:
            return path