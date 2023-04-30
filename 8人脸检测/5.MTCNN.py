'''
MTCNN卷积神经网络 检测人脸 -- 
1. 导入人脸图片
2. 使用hog模型检测
3. 将识别结果矩形框画出
'''

'''
# 安装dlib，装完cmake后要重启终端
# 确认电脑上有cmake brew install cmake    cmake --version
# pip install cmake
# pip install boots
# pip install dlib
'''


# 导入HOG人脸检测模型

import cv2
import numpy as np
import glob
# pip install tensorflow   深度学习库
# pip install  mtcnn
from mtcnn.mtcnn import MTCNN
hog_face_detector = MTCNN()

for jpgPath in reversed(glob.glob('/Users/azir/Desktop/python_vision/8-9人脸考勤机/imgs/*.jpg')):

    img = cv2.imread(jpgPath)
    # mtcnn需要转rgb
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 检测人脸 -- 调节参数，让检测更准确  scale调整图片尺寸，
    detections = hog_face_detector.detect_faces(img_cvt)
    print(detections)
    # 绘制人脸框
    for face in detections:
        [x, y, w, h] = face['box']

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow('Demo', img)
    cv2.waitKey(3000)

    print(jpgPath)
cv2.destroyAllWindows()
