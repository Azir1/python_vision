
## OCR 光学字符识别
1. 传统方法 -- 不推荐了
图像预处理--》字符分割--》字符识别

2. 基于深度学习的
图像矫正模块--》识别特征提取模块--》序列特征提取模块--》预测模块


## 目标：
1. 训练OCR文字检测模型
2. 训练OCR文字识别模型

## 安装paddle OCR
1. 根据官网教程安装
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html

## paddle自带的标注工具
1. 标注后，先是用 矩形框 标注出车牌位置 --- 作车牌目标检测
2. 
