# 检测视频中的人脸
# 对比后 视频中还是用haar，其他的都是过了，很卡
import cv2

face_detector = cv2.CascadeClassifier(
    '/Users/azir/Desktop/python_vision/8人脸检测/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_flip = cv2.flip(frame, 1)
        frame_cvt = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
        # 检测人脸 -- 调节参数，让检测更准确  scaleFactor调整图片尺寸，minNeighbors 候选人脸数量(影响准确率)  minSize 最小人脸尺寸  maxSize最大人脸尺寸
        detectionsRes = face_detector.detectMultiScale(
            frame_cvt, scaleFactor=1.2, minNeighbors=7, minSize=(20, 20))

        for rect in detectionsRes:
            x, y, w, h = rect

            cv2.rectangle(frame_flip, (x, y), (w+x, h+y), (0, 255, 0), 2)

        cv2.imshow('Demo', frame_flip)

        if cv2.waitKey(10) & 0xFF == 27:
            break


cv2.destroyAllWindows()
