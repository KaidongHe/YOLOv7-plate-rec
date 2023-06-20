# YOLOv7-plate 车牌识别

本项目支持：  
1.本项目支持12种常见车牌识别，具体实现原理见“reference”。  
2.本项目使用PyQt5创建了一个简单的UI界面，具体实现见代码。  
  
自开源来，回开源去。  


# 文档介绍

同时使用高版本PyQt5和opencv-python包可能会无法运行，解决方案是使用本项目提供的requirements.txt，降低opencv-python版本，建议使用[anaconda](https://anaconda.org/anaconda)进行Python环境搭建。  

##  测试环境

>CPU：Intel(R) Core(R) i3-12100F CPU @ up to 4.30 GHz  
>GPU：GeForce GTX 1080Ti x 1  
>内存：16G   
>操作系统：Ubuntu20.04  


## 环境搭建

>git clone https://github.com/ultralytics/yolov5  # clone  
>cd yolov5  
>pip install -r requirements.txt  # install  

## 测试demo

>cd yolov7-plate-PyQt  
>python3 ui_yolo.py  

## 训练
>详细请参考references中的链接。  



# References

https://github.com/we0091234/yolov7_plate  
https://github.com/derronqi/yolov7-face  
https://github.com/we0091234/crnn_plate_recognition  
https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec  
