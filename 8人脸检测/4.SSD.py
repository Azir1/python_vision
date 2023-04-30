'''
CNN 检测人脸
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

import cv2
import numpy as np
import glob
import dlib
# 导入ssd人脸检测模型和权重文件
hog_face_detector = cv2.dnn.readNetFromCaffe('/Users/azir/Desktop/python_vision/8-9人脸考勤机/SSD_data/deploy.prototxt.txt','/Users/azir/Desktop/python_vision/8-9人脸考勤机/SSD_data/res10_300x300_ssd_iter_140000.caffemodel')

for jpgPath in reversed(glob.glob('/Users/azir/Desktop/python_vision/8-9人脸考勤机/imgs/*.jpg')):

    img = cv2.imread(jpgPath)
    img_height = img.shape[0]
    img_width = img.shape[1]
    # 缩放至模型输入尺寸
    imgSize = cv2.resize(img, (500, 300))
    
    # 将图像转为二进制
    img_blob = cv2.dnn.blobFromImage(imgSize,1.0,(500,300),(104.0,177.0,123.0))
    # HOG无需灰度图
    # 输入
    hog_face_detector.setInput(img_blob)
    # 检测人脸 
    detections = hog_face_detector.forward()
    # 人脸的数量
    num_detections =detections.shape[2]
    print(detections)
    # 绘制人脸框
    for index in range(num_detections) :
        # 置信度
        detection_confidence = detections[0,0,index,2]
        if detection_confidence>0.18:
          # 位置  因为缩放了，乘以原图的尺寸，恢复
          locations = detections[0,0,index,3:7]*np.array([img_width,img_height,img_width,img_height])
          lx,ly,rx,ry=locations.astype('int')
          print(detection_confidence)
        
        
          cv2.rectangle(img, (lx,ly), (rx,ry), (0, 255, 0), 1)
    cv2.imshow('Demo', img)
    cv2.waitKey(2000)

    print(jpgPath)
cv2.destroyAllWindows()
