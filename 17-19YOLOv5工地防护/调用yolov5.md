1. pytorch hub调用yolo模型 ，详见main.py
加载训练好的YOLOv5n模型，并做简单识别应用

参考文档 -- 官网推荐的pytorch方式
https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/


2. tensorRT 加速

### 1. 转换pt-->wts
下载yolov5源码以及tensorrtx源码，并将yolov5s.pt转为.wts模型
将tensorrtx源码中的gen_wts.py复制到yolov5源码中并运行，生成.wts模型。

python gen_wts.py -w /home/miaojiawei/桌面/python_vision/17-19YOLOv5工地防护/yolov5/weights/yolov5n.pt

Writing into /home/miaojiawei/桌面/python_vision/17-19YOLOv5工地防护/yolov5/weights/yolov5n.wts


### 2. 构建教程
https://blog.csdn.net/qq_51331745/article/details/122251164

更改tensorrtx中yolov5 cmakelist  
改完pytorch路径后
注释以下行 ，不然make的时候报错
add_definitions(-std=c++11)


4.使用Yolov5
1）最开始时，使用命令：

sudo ./yolov5 -s yolov5s.wts yolov5s.engine s

    1

进行engine文件生成，但出现如下问题：

arguments not right!
./yolov5 -s  // serialize model to plan file
./yolov5 -d ../samples  // deserialize plan file and run inference

    1
    2
    3

原因是因为我使用的是yolov5-v3.0版本代码，具体情况参考问题
更换对应版本代码即可。
打开对应tensorrtx/yolov5文件夹下的READMI.md,参考对应指令即可。


### 2. 
3. deepstream 

英伟达® DeepStream软件开发工具包（SDK）是一个用于构建智能视频分析（IVA）管道的加速人工智能框架。DeepStream 可运行在 NVIDIA T4、NVIDIA Ampere 和 NVIDIA® Jetson™ Nano、NVIDIA® Jetson AGX Xavier™、NVIDIA® Jetson Xavier NX™、 NVIDIA® Jetson™ TX1 和 TX2。
————————————————
版权声明：本文为CSDN博主「许野平」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/quicmous/article/details/116190449