'''
1. 读取显示图像，灰度化，二值化
2. 侵蚀去噪点
3. 膨胀连接
4. 闭合孔洞
2. 获取轮廓
3. 轮廓坐标
4. 绘制矩形框
'''

import cv2
import numpy as np

img = cv2.imread(
    '/home/miaojiawei/桌面/workspace/pytools/python_vision/6maobi/demowenzi.png')

# 灰度化
imgGrey = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
# 二值化 -THRESH_BINARY_INV 黑白转换

thredBig = cv2.resize(imgGrey, None, fx=1.5, fy=1.5,
                      interpolation=cv2.INTER_CUBIC)
ret, thred = cv2.threshold(thredBig, 160, 255, cv2.THRESH_BINARY_INV)


kernel = np.ones((2, 2), dtype=np.int16)
# 开运算 侵蚀去噪点，后膨胀
# opening = cv2.morphologyEx(thred.copy(),cv2.MORPH_OPEN,kernel)

kernel2 = np.ones((3, 3), dtype=np.int16)
# 侵蚀-去噪点
ersion1 = cv2.erode(thred.copy(), kernel, iterations=1)

# 膨胀
dilation1 = cv2.dilate(ersion1.copy(), kernel2, iterations=1)
kernel3 = np.ones((3, 3), dtype=np.int16)
# 闭合孔洞，不然小框太多了
closing = cv2.morphologyEx(dilation1.copy(), cv2.MORPH_CLOSE, kernel3)
# canny轮廓
imgBorder = cv2.Canny(closing, 100, 250)

contours, hierarchy = cv2.findContours(
    imgBorder, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imgBig = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > 10 & h > 7:
        cv2.rectangle(imgBig, (x, y), (w+x, h+y), (0, 255, 0), 1)
while True:
    # cv2.imshow('Demo',np.hstack((thredBig,thred,ersion1,dilation1,closing,imgBorder)))
    cv2.imshow('Demo2', imgBig)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
