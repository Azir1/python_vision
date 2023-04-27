# ```
# 注意mac环境暂时用不了conda，要在本机python下运行
# 推出conda  deactivate
# 5 手指关键点移动方块

# 1. Opencv获取视频流
# 2. 在画面上画一个方块
# 3. 通过mideapipe获得手指关键点坐标
# 4. 判断手指是否在方块上
# 5. 如果在方块上，方块跟随手指移动
# ```

import cv2
import numpy as np
import time
import mediapipe as mp
from util_font import cv2ImgAddText


def handleRect():
    # 定义并引用mediapipe中的hands模块
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    prev = time.time()
    # 读取视频流
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())

    pt1 = 500
    pt2 = 300
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # 获取灰度图
        grey = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2图像初始化
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # 关键点的位置

                    # print(id, cx, cy)
                    if id == 8:
                        print(id, cx, cy)
                        # 注意判断条件需要加上 括号！！！
                        if (pt1-200 < cx < pt1+100) & (pt2-200 < cy < pt2+100):
                            pt1 = cx
                            pt2 = cy

                    cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                # 绘制手部特征点：
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        cur = time.time()
        # 求帧率
        fps = int(1/(cur-prev))
        text = '帧率'+str(fps)

        # 画面左右倒置
        # 画实心正方形
        cv2.rectangle(frame, (pt1, pt2), (pt1+50, pt2+50), (0, 255, 0), -1)
        newFrame = cv2.flip(frame, 1)
        txt = cv2ImgAddText(newFrame, text, 30, 20)
        cv2.imshow('Demo', txt)
        prev = time.time()
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    handleRect()
