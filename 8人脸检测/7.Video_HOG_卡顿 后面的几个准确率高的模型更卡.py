# 检测视频中的人脸


import cv2

from mtcnn.mtcnn import MTCNN
import dlib
# 对比后 视频中还是用haar，其他的都是过了，很卡

detections = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        detectionsRes = detections(frame, 1)

        for face in detectionsRes:
            x = face.left()
            y = face.top()
            r = face.right()
            b = face.bottom()

            cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 1)

        cv2.imshow('Demo', frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break


cv2.destroyAllWindows()
