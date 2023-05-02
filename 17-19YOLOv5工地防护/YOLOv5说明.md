## YOLO思想
输入图片--》卷积神经网络--》输出标注框结果
将图片分成4*4的格子，分别预测每个格子是否包含目标物体
 YOLO会预测出多个框，通过IOU（交并比）交集面积/并集面积，计算出最符合的框

## 安装pytorch GPU版  （macbook intel版不支持英伟达GPU加速，CPU训练太垃圾了，所以先用win笔记本）

## ubuntu 安装cuda10.1 笔记本的显卡是GT755M  对应的python版本是python3.5-3.8
https://blog.csdn.net/wwlswj/article/details/106364094

# 安装对应版本的pytorch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# 安装yolo v5
https://blog.csdn.net/xiaokai1999/article/details/114395430

# 测试yolo是否跑通
python detect.py --source "data/images/bus.jpg" --weights="weights/yolov5s.pt"
python detect.py --source "/home/miaojiawei/桌面/python_vision/8人脸检测/imgs/1.jpg" --weights="weights/yolov5s.pt" --conf-thres 0.3

将yolo中的detect.py 
parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
改成如下，即可直接调用笔记本摄像头  python detect.py
parser.add_argument('--source', type=str, '0', help='file/dir/URL/glob/screen/0(webcam)')

# 标注工具

使用LabelImg等标注工具（需要支持YOLO格式，使用的时候记得切换一下）标注图片：
LabelImg可以通过github下载：https://github.com/tzutalin/labelImg

# COCO数据集

在正确配置好环境后就可以检测自己的图片或视频了。YOLOv5已经在COCO数据集上训练好，COCO数据集一共有80个类别，如果您需要的类别也在其中的话，可以直接用训练好的模型进行检测。这80个类分别是：

[‘person’, ‘bicycle’, ‘car’, ‘motorcycle’, ‘airplane’, ‘bus’, ‘train’, ‘truck’, ‘boat’, ‘traffic light’, ‘fire hydrant’, ‘stop sign’, ‘parking meter’, ‘bench’, ‘bird’, ‘cat’, ‘dog’, ‘horse’, ‘sheep’, ‘cow’, ‘elephant’, ‘bear’, ‘zebra’, ‘giraffe’, ‘backpack’, ‘umbrella’, ‘handbag’, ‘tie’, ‘suitcase’, ‘frisbee’, ‘skis’, ‘snowboard’, ‘sports ball’, ‘kite’, ‘baseball bat’, ‘baseball glove’, ‘skateboard’, ‘surfboard’, ‘tennis racket’, ‘bottle’, ‘wine glass’, ‘cup’, ‘fork’, ‘knife’, ‘spoon’, ‘bowl’, ‘banana’, ‘apple’, ‘sandwich’, ‘orange’, ‘broccoli’, ‘carrot’, ‘hot dog’, ‘pizza’, ‘donut’, ‘cake’, ‘chair’, ‘couch’, ‘potted plant’, ‘bed’, ‘dining table’, ‘toilet’, ‘tv’, ‘laptop’, ‘mouse’, ‘remote’, ‘keyboard’, ‘cell phone’, ‘microwave’, ‘oven’, ‘toaster’, ‘sink’, ‘refrigerator’, ‘book’, ‘clock’, ‘vase’, ‘scissors’, ‘teddy bear’, ‘hair drier’, ‘toothbrush’]
# 训练电脑拉跨，直接用训练好了的模型做视频检测

# 模型评估标准 -- 重点

- precision 模型检测精度、查准率（预测的所有结果中，有多少是真的）。是评估预测的准不准  -- 越高越好

- recall 召回率、查全率  是评估找的全不全-- 是不是把所有真的都找出来了

- IOU 交并比 ，预测的框位置和标注框位置对比，看是否准确  -- IOU=1 表示完全匹配，可以设定一个IOU阈值，如果IOU>0.5，表示真阳性，越高说明越准确

- AP 衡量模型在每个类别上的好坏

- mAP 衡量模型在所有类型上的好坏,所有类别的好坏的平均值


# 检测速度

- 前传耗时，输入图像到输出结果的耗时

- FPS，每秒帧数，每秒钟能处理的图像的数量

- 浮点运算量，处理一张图片需要的浮点运算数量，与硬件无关


