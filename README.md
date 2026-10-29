# 使用方法

## 1. 环境准备

- 在带有 pip 的 Python 环境下（Python版本 >= 3.10.20），在终端执行：`pip install -r requirements.txt`
- **如非离线使用，首次运行时，Insightface 库将自动从网络下载预训练模型，需要保持网络畅通。**
- **模型较大，故未包含于项目文件夹中，离线使用的方法参见 1.1** 

### 1.1 离线使用 

- 下载模型：百度网盘链接 `https://pan.baidu.com/s/1f0OETYSumjx_fbAo1Tj-WQ?pwd=s4ek`（提取码: s4ek），内容为两个压缩包，分别为 `buffalos_s`（体积小运行快）和  `buffalo_l` （较大、较慢、精度较高）
- 下载后分别将压缩包内的 `*.onnx` 文件（每个模型5个）解压到：
  - Windows：用户文件夹下的 `.insightface\models\buffalo_s` 和 `.insightface\models\buffalo_l`
  - Linux：`~/.insightface/models/buffalo_s` 和 `~/.insightface/models/buffalo_l`
- 目录结构:
```
~/
  ...
  .insightface/
    models/
      buffalo_s/
        1k3d68.onnx
        2d106det.onnx
        det_500m.onnx
        genderage.onnx
        w600k_mbf.onnx
      buffalo_l/
        1k3d68.onnx
        2d106det.onnx
        det_10g.onnx
        genderage.onnx
        w600k_r50.onnx
```

## 2. 项目结构和功能

- **`dataset` 文件夹：自行收集的数据集**
- `Provided_Dataset` 文件夹：教师提供的数据集(从CelebA中随机抽取的100个身份)
- **`scripts/Test.py`**
  - **运行测试**
  - 给定一系列置信度阈值，在测试集上测试识别效果
  - 准确率将在命令窗口中输出
  - 在 `./Bad_Cases_{运行日期时间}` 中生成最高阈值的一组测试中无法匹配、匹配错误的照片
  - 由于本项目的模型在 CPU 上跑，运行时间较长，一般要几分钟
- **`scripts/Demonstrate.py`**
  - **运行基于 html 的示例 Demo**
  - 运行后会自动打开网页（可以关闭该功能），也可手动打开 `scripts/Frontend.html`
- **`scripts/Controller.py`：集中控制项目选用的模型和数据集**
- `scripts/Build.py`：遍历指定的数据集并建库
- `scripts/Recognize.py`：包含人脸识别相关的函数，对提供的图片根据注册数据进行 1: N 检索
- `scripts/Extract.py`：包含从图片中提取特征的函数
- `scripts/Frontend.html`：用于展示 “上传照片 - 1: N 检索” 功能的前端网页文件
- `scripts/Mark_Test_Imgs_Auto.py`：使用模型进行自动数据标注（作为辅助手段，需要人工微调）
- `scripts/Draw_Marker.py`：人工数据标注的辅助工具，根据 `dataset/test/annotations.jsonl` 给测试照片画标注框

## 3. 运行步骤

1. 请确保工作目录设置为项目根目录，且路径中没有中文字符，请勿移动项目内各种文件的位置
2. 在 `Controller.py` 中按指引选择模型和数据集
3. （非必需）运行 `Build.py`，会遍历指定的注册集文件夹下的全部图片，建造编号 - 特征向量数据集
4. 运行 `Test.py`，会遍历指定的测试集文件夹下的全部图片进行测试，输出结果并提供指定阈值下的错误样例
5. 运行 `Demonstrate.py` ，自动或手动在浏览器中打开 `Frontend.html`，可体验前端的上传照片 - 1: N 检索功能