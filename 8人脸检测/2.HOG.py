'''
hog 检测人脸
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
import dlib
hog_face_detector = dlib.get_frontal_face_detector()

for jpgPath in reversed(glob.glob('/Users/azir/Desktop/python_vision/8人脸检测/imgs/*.jpg')):

    img = cv2.imread(jpgPath)
    imgSize = cv2.resize(img, (620, 400))
    # HOG无需灰度图
    # 检测人脸 -- 调节参数，让检测更准确  scale调整图片尺寸，
    detections = hog_face_detector(imgSize, 1)
    print(detections)
    # 绘制人脸框
    for face in detections:
        x=face.left()
        y=face.top()
        r=face.right()
        b=face.bottom()
        
        cv2.rectangle(imgSize, (x,y), (r,b), (0, 255, 0), 1)
    cv2.imshow('Demo', imgSize)
    cv2.waitKey(3000)

    print(jpgPath)
cv2.destroyAllWindows()
