'''
人脸考勤
人脸注册：将人脸特征存进feature.csv
人脸识别：将检测的人脸特征与CSV中的人脸特征进行比较，如果命中，将考勤记录写入 attendance.csv
可参考文档
https://www.cnblogs.com/supersayajin/p/8489435.html
'''
import cv2
import numpy as np
import glob
import os
import dlib
import json


# 人脸目标检测+获取关键点

def get_face_rect(label_id, count, interval):

    # haar人脸目标检测
    face_detector = cv2.CascadeClassifier(
        '/Users/azir/Desktop/python_vision/8人脸检测/data/haarcascades/haarcascade_frontalface_default.xml')
    # landmark模型获取人脸的68个检测点
    shape_predict = dlib.shape_predictor(
        '/Users/azir/Desktop/python_vision/9人脸识别/shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)

    # resnet模型  人脸特征点提取
    face_resnet_detector = dlib.face_recognition_model_v1(
        '/Users/azir/Desktop/python_vision/9人脸识别/dlib_face_recognition_resnet_model_v1.dat')
    data = np.zeros((1, 128))  # 定义一个128维的空向量data
    label = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_flip = cv2.flip(frame, 1)
            frame_cvt = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
            # 1. 检测人脸 -- 调节参数，让检测更准确  scaleFactor调整图片尺寸，minNeighbors 候选人脸数量(影响准确率)  minSize 最小人脸尺寸  maxSize最大人脸尺寸
            detectionsRes = face_detector.detectMultiScale(
                frame_cvt, scaleFactor=1.2, minNeighbors=7, minSize=(20, 20))
            # 2. 截取人脸范围
            for rect in detectionsRes:
                x, y, w, h = rect
                # 3.绘制矩形框
                cv2.rectangle(frame_flip, (x, y), (w+x, h+y), (0, 255, 0), 2)

                # 4.计算68个关键点
                rec = dlib.rectangle(x, y, w+x, h+y)
                points = shape_predict(frame_flip, rec)
                for point in points.parts():

                    cv2.circle(frame_flip, (point.x, point.y),
                               1, (0, 255, 0), -1)
                # 5.renet提取人脸特征
                face_descriptor = face_resnet_detector.compute_face_descriptor(
                    frame_flip, points)  # 使用resNet获取128维的人脸特征向量
                faceArray = np.array(face_descriptor).reshape(
                    (1, 128))  # 转换成numpy中的数据结构
                print(faceArray)
                data = np.concatenate((data, faceArray))  # 拼接到事先准备好的data当中去
                label.append('jiawei')
            
            cv2.imshow('Demo', frame_flip)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    data = data[1:, :]                                                                                  #因为data的第一行是空的128维向量，所以实际存储的时候从第二行开始
    np.savetxt('faceData.txt', data, fmt='%f')                                                          #保存人脸特征向量合成的矩阵到本地

    labelFile=open('label.txt','w')                                      
    json.dump(label_id, labelFile)                                                                         #使用json保存list到本地
    labelFile.close()
# 人脸注册


def register_face(label_id, label, count, interval):
    '''
    label_id:人脸id
    label:人脸姓名
    count:采集数量
    interval:采集间隔
    '''
    # 获取视频流
    # haar检测人脸
    # 68个关键点获取
    # 提取特征
    # 特征保存到scv
    get_face_rect(label_id, count, interval)


cv2.destroyAllWindows()


def detect_face(img, count, interval):
    pass


if __name__ == "__main__":
    register_face(1, 2, 3, 4)
