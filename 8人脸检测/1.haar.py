'''
haar 检测人脸
1. 导入人脸图片
2. 使用haar模型检测
3. 将识别结果矩形框画出
'''

import cv2
import numpy as np
import glob
import os

# 导入opencv haar人脸检测模型(github直接下载)，正脸检测
face_detector = cv2.CascadeClassifier(
    '/Users/azir/Desktop/python_vision/8人脸检测/data/haarcascades/haarcascade_frontalface_default.xml')

for jpgPath in reversed(glob.glob('/Users/azir/Desktop/python_vision/8人脸检测/imgs/*.jpg')):

    img = cv2.imread(jpgPath)
    imgSize = cv2.resize(img, (620, 400))
    # haar 需要的是灰度图
    grey = cv2.cvtColor(imgSize, code=cv2.COLOR_BGR2GRAY)
    # 检测人脸 -- 调节参数，让检测更准确  scaleFactor调整图片尺寸，minNeighbors 候选人脸数量(影响准确率)  minSize 最小人脸尺寸  maxSize最大人脸尺寸
    detections = face_detector.detectMultiScale(
        grey, scaleFactor=1.18, minNeighbors=4, minSize=(20, 20), maxSize=(300, 300))
    # 绘制人脸框
    for rect in detections:
        x, y, w, h = rect
        cv2.rectangle(imgSize, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow('Demo', imgSize)
    cv2.waitKey(3000)

    print(jpgPath)
cv2.destroyAllWindows()
