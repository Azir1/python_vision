'''
1. 图片数据预处理
2. 加载模型
3. 提取图片的特征描述符
4. 预测图片：找到欧式距离最近的特征描述符
5. 评估测试数据集

主要是未找到合适的人脸数据集
'''


# hog检测出人脸区域
import cv2
import numpy as np
import dlib
# HOG模型
hog_face_detector = dlib.get_frontal_face_detector()
# 获取人脸的68个检测点
shape_predict = dlib.shape_predictor(
    '/Users/azir/Desktop/python_vision/9人脸识别/shape_predictor_68_face_landmarks.dat')
# resnet模型  人脸特征点提取
face_resnet_detector=dlib.cnn_face_detection_model_v1('/Users/azir/Desktop/python_vision/9人脸识别/dlib_face_recognition_resnet_model_v1.dat')
img = cv2.imread('/Users/azir/Desktop/python_vision/8人脸检测/imgs/4.jpg')

# 人脸检测
detections = hog_face_detector(img, 1)
print(detections)
# 绘制人脸框
for face in detections:
    x = face.left()
    y = face.top()
    r = face.right()
    b = face.bottom()
    points = shape_predict(img, face)
    print(points)
    # 绘制关键点
    for point in points.parts():

        cv2.circle(img, (point.x, point.y), 1, (0, 255, 0), -1)
    # 绘制矩形框
    cv2.rectangle(img, (x, y), (r, b), (0, 255, 0), 1)
cv2.imshow('Demo', img)
cv2.waitKey(4000)

cv2.destroyAllWindows()
