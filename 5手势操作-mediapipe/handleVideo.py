from cgitb import grey
import cv2
import time
import numpy as np
from util_font import cv2ImgAddText
# 读取视频
cap = cv2.VideoCapture('/home/miaojiawei/桌面/workspace/pytools/out.mp4')
# 获取视频大小
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 获取总的帧率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(frame_height, frame_width, frame_count)
prev = time.time()

if not cap.isOpened():
    print('检查路径，文件打开失败')
    
while cap.isOpened():
    ret, frame = cap.read()
    # 播完自动退出
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cur = time.time()

    fps = int(1/(cur-prev))
    fps = '帧率：' + str(fps)
    # 画矩形
    txt = cv2ImgAddText(grey, fps, 50, 50, (255, 255, 255), 50)
    cv2.rectangle(txt, (80, 80), (300, 300), (0, 255, 0), 5)
    # cv2.putText(grey, fps, (50, 50), cv2.FONT_HERSHEY_COMPLEX,
    #             2.0, (255, 255, 255), 5)
    cv2.imshow('Demo', txt)
    # 计算实时帧率
    prev = time.time()
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
