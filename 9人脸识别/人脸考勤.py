'''
 TODO：欧式距离计算再看看
人脸考勤
人脸注册：将人脸特征存进feature.csv
人脸识别：将检测的人脸特征与CSV中的人脸特征进行比较，如果命中，将考勤记录写入 attendance.csv
可参考文档
https://www.cnblogs.com/supersayajin/p/8489435.html
'''
import cv2
import numpy as np
import dlib
import time
import csv

# 人脸目标检测+获取关键点
def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

# 人脸注册
# 获取视频流
# haar检测人脸
# 68个关键点获取
# 提取特征
# 特征保存到csv
# haar人脸目标检测

def register_face(label_id=2, label='qq', count=2, interval=2):
    '''
    label_id:人脸id
    label:人脸姓名
    count:采集数量
    interval:采集间隔interval单位是s
    '''

    face_detector = cv2.CascadeClassifier(
        '/Users/azir/Desktop/python_vision/8人脸检测/data/haarcascades/haarcascade_frontalface_default.xml')
    # landmark模型获取人脸的68个检测点
    shape_predict = dlib.shape_predictor(
        '/Users/azir/Desktop/python_vision/9人脸识别/shape_predictor_68_face_landmarks.dat')
    cap = cv2.VideoCapture(0)

    # resnet模型  人脸特征点提取
    face_resnet_detector = dlib.face_recognition_model_v1(
        '/Users/azir/Desktop/python_vision/9人脸识别/dlib_face_recognition_resnet_model_v1.dat')
    # 开始时间
    prev_time = time.time()
    # 采集次数
    collect_count = 0

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
                if (collect_count <= count) & (len(points.parts()) > 0):
                    if (time.time()-prev_time) > interval:
                        # 5.renet提取人脸特征描述符
                        face_descriptor = face_resnet_detector.compute_face_descriptor(
                            frame_flip, points)  # 使用resNet获取128维的人脸特征向量
                        # 转为列表
                        face_descriptor = [f for f in face_descriptor]
                        # 存入csv文件
                        faceArray = np.array(face_descriptor).reshape(
                            (1, 128))  # 转换成numpy中的数据结构
                        print(faceArray)
                        line = [label_id, label, faceArray]
                        write_csv(
                            '/Users/azir/Desktop/python_vision/9人脸识别/face_data/features.csv', line)

                        collect_count += 1
                        print('采集次数', collect_count)
                        prev_time = time.time()
                else:
                    print('采集完毕')
                    cv2.destroyAllWindows()

                    return
            cv2.imshow('Demo', frame_flip)

        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def formatarr(str_array="[[1,2,3],[4,5,6],[7,8,9]]"):
  # 首先去掉字符串两端的方括号
    str_rows = str_array.strip('[').strip(']')
    # 按逗号分割字符串中的每一个元素，返回一个列表
    str_rows = str_rows.replace('\n', '')

    # 定义一个空的二维数组
    matrix = str_rows.split(' ')
    newArr = []
    for i in matrix:
        if i:
            newArr.append(float(i))

    return newArr


# 人脸识别
# 1. 实时获取视频流中人脸的特征描述符
# 2. 将它与库里的特征做欧式距离判断
# 3. 找到预测的ID、NAME
# 4. 保存考勤信息


def detect_face():
    cap = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier(
        '/Users/azir/Desktop/python_vision/8人脸检测/data/haarcascades/haarcascade_frontalface_default.xml')
    # landmark模型获取人脸的68个检测点
    shape_predict = dlib.shape_predictor(
        '/Users/azir/Desktop/python_vision/9人脸识别/shape_predictor_68_face_landmarks.dat')

    # resnet模型  人脸特征点提取
    face_resnet_detector = dlib.face_recognition_model_v1(
        '/Users/azir/Desktop/python_vision/9人脸识别/dlib_face_recognition_resnet_model_v1.dat')
    # 读取csv
    csv_reader = csv.reader(
        open("/Users/azir/Desktop/python_vision/9人脸识别/face_data/features.csv"))
    name_list = []
    id_list = []
    features_list = None
    for row in csv_reader:
        label_id = row[0]
        label = row[1]

        face_data = formatarr(row[2])
        face_data = np.asarray(face_data, dtype=np.float64)
        face_data = np.reshape(face_data, (1, -1))
        print(face_data)
        name_list.append(label)
        id_list.append(label_id)
        if features_list is None:
            features_list = face_data
        else:
          # 特征拼接
            features_list = np.concatenate((features_list, face_data), axis=0)

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

                # 5.renet提取人脸特征描述符
                face_descriptor = face_resnet_detector.compute_face_descriptor(
                    frame_flip, points)  # 使用resNet获取128维的人脸特征向量
                # 转为列表
                face_descriptor = [f for f in face_descriptor]
                # 转换为np格式数组，然后计算欧式距离
                face_descriptor = np.asarray(face_descriptor, dtype=np.float64)
                # 设置阀值
                threshold = 0.5
                # 读取csv文件，欧式距离计算
                print(face_descriptor.shape, features_list.shape)
                temp = face_descriptor-features_list
                e = np.linalg.norm(temp, axis=1, keepdims=True)
                min_distance = e.min()
                print('distance: ', min_distance)
                if min_distance > threshold:
                    print('other')
                index = np.argmin(e)
                cv2.putText(frame_flip, str(
                    name_list[index]), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                print(name_list[index])

            cv2.imshow('Demo', frame_flip)

        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 人脸信息注册
    # register_face()
    # 人脸识别
    detect_face()
