'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-06-15 19:10:25
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-06-19 22:18:52
FilePath: /my_file/file.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import os
from yolov7_detect_rec import *

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap

def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)   # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate),int(FrameNumber),int(duration)  



class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi("./ui_yolo.ui")
        # print(self.ui.__dict__)  # 查看ui文件中有哪些控件

        self.select_btn = self.ui.pushButton  # 登录按钮

        self.selectdir_btn = self.ui.dic_btn

        self.plate_label = self.ui.label_3

        self.videobtn=self.ui.video_btn

        self.cambtn = self.ui.video0_btn

        self.plateshow_label = self.ui.label

        self.file_list_widget = self.ui.pic_list
        self.file_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        # 绑定信号与槽函数
        self.select_btn.clicked.connect(self.openFile)
        self.selectdir_btn.clicked.connect(self.openDir)
        self.videobtn.clicked.connect(self.openvideo)
        self.cambtn.clicked.connect(self.opencam)
        self.file_list_widget.itemDoubleClicked.connect(self.show_select_item1)


    def show_select_item1(self,item):
        #QMessageBox.information(self, "ListWidget", "You clicked: "+item.text())
        pixmap = QPixmap(item.text())
        scaredPixmap = pixmap.scaled(self.plateshow_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.plateshow_label.setPixmap(scaredPixmap)
        

    '''打开文件'''
    def openFile(self):
    	#其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        # directory1 = QFileDialog.getExistingDirectory(self,
        #                                               "选取文件夹",
        #                                               "./")  # 起始路径
        file_th = QFileDialog.getOpenFileName(self, 
                                                 'Open file', 
                                                 '/home')[0]
        print(file_th)
        #os.system("python3 /home/antis/plate_yolov7/yolov7_plate-master/detect_rec_plate.py --detect_model /home/antis/plate_yolov7/yolov7_plate-master/weights/yolov7-lite-s.pt  --rec_model /home/antis/plate_yolov7/yolov7_plate-master/weights/plate_rec.pth --output /home/antis/plate_yolov7/yolov7_plate-master/result --source /home/antis/plate_yolov7/yolov7_plate-master/imgs/" )
        detect_model = 'weights/yolov7-lite-s.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        output = 'result'
        source = file_th
        img_size = 640
        # torch.save()
        plate_rec_model=init_model(device,rec_model) 
        if not os.path.exists(output):
            os.mkdir(output)


        time_b = time.time()

        print(file_th,end=" ")
        img = cv_imread(file_th)
        if img.shape[-1]==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        # img = my_letter_box(img)
        dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
        for result in dict_list:
            print('\n')
            print(result['plate_no'])
            self.plate_label.setText(result['plate_no'])
        ori_img=draw_result(img,dict_list)
        img_name = os.path.basename(file_th)
        save_img_path = os.path.join(output,img_name)
        cv2.imwrite(save_img_path,ori_img)
        pixmap = QPixmap(save_img_path)
        scaredPixmap = pixmap.scaled(self.plateshow_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
        self.plateshow_label.setPixmap(scaredPixmap)
        print(f"elasted time is {time.time()-time_b} s")

    def openDir(self):
        #其中self指向自身，"读取文件夹"为标题名，"./"为打开时候的当前路径
        directory1 = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "/home")  # 起始路径
        if(directory1 == None):
            return
        print(directory1)
        #os.system("python3 /home/antis/plate_yolov7/yolov7_plate-master/detect_rec_plate.py --detect_model /home/antis/plate_yolov7/yolov7_plate-master/weights/yolov7-lite-s.pt  --rec_model /home/antis/plate_yolov7/yolov7_plate-master/weights/plate_rec.pth --output /home/antis/plate_yolov7/yolov7_plate-master/result --source /home/antis/plate_yolov7/yolov7_plate-master/imgs/" )
        detect_model = 'weights/yolov7-lite-s.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        output = 'result'
        source = directory1
        img_size = 640
        # torch.save()
        plate_rec_model=init_model(device,rec_model) 
        if not os.path.exists(output):
            os.mkdir(output)

        file_list=[]
        out_list=[]
        allFilePath(source,file_list)
        print(file_list)
        time_b = time.time()
        for pic_ in file_list:
            print(pic_,end=" ")
            img = cv_imread(pic_)
            if img.shape[-1]==4:
                img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            # img = my_letter_box(img)
            dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
            for result in dict_list:
                print('\n')
                print(result['plate_no'])
                self.plate_label.setText(result['plate_no'])
            ori_img=draw_result(img,dict_list)
            img_name = os.path.basename(pic_)
            save_img_path = os.path.join(output,img_name)
            cv2.imwrite(save_img_path,ori_img)
            pixmap = QPixmap(save_img_path)
            scaredPixmap = pixmap.scaled(self.plateshow_label.size(), aspectRatioMode=Qt.KeepAspectRatio)
            self.plateshow_label.setPixmap(scaredPixmap)
        
        allFilePath(output,out_list)
        print(out_list)
        for pic_pth in out_list:
            self.file_list_widget.addItem(pic_pth)
        print(f"elasted time is {time.time()-time_b} s")
  

    def openvideo(self):
        video_name =  QFileDialog.getOpenFileName(self, 
                                                 'Open video', 
                                                 '/home')[0]
        print(video_name)
        capture=cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
        fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
        #out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频
        frame_count = 0
        fps_all=0
        #rate,FrameNumber,duration=get_second(capture)
        img_size = 640
        detect_model = 'weights/yolov7-lite-s.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        plate_rec_model=init_model(device,rec_model) 
        if capture.isOpened():
            while True:
                t1 = cv2.getTickCount()
                frame_count+=1
                print(f"第{frame_count} 帧",end=" ")
                ret,img=capture.read()
                if not ret:
                    break
                if img.shape[-1]==4:
                    img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
                # if frame_count%rate==0:
                dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
                ori_img=draw_result(img,dict_list)
                t2 =cv2.getTickCount()
                infer_time =(t2-t1)/cv2.getTickFrequency()
                fps=1.0/infer_time
                fps_all+=fps
                str_fps = f'fps:{fps:.4f}'
                
                cv2.putText(ori_img,str_fps,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imshow("video",ori_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #out.write(ori_img)                           
        else:
            print("失败")
            
        capture.release()
        #out.release()
        cv2.destroyAllWindows()
        print(f"all frame is {frame_count},average fps is {fps_all/frame_count} fps")

    def opencam(self):
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(0)
    
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        img_size = 640
        detect_model = 'weights/yolov7-lite-s.pt'
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        model = attempt_load(detect_model, map_location=device)
        rec_model = 'weights/plate_rec_color.pth'
        plate_rec_model=init_model(device,rec_model) 
        output = 'result'
        # Read until video is completed
        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            if img.shape[-1]==4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (img_size, img_size))
            dict_list=detect_Recognition_plate(model, img, device,plate_rec_model,img_size)
            ori_img=draw_result(img,dict_list)
            if output != None:
                cv2.imshow("post process result", ori_img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()#关闭视频文件或者摄像头
        cv2.destroyAllWindows()
        print("camrea is closed!!")
        


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 展示窗口
    w.ui.show()

    app.exec()
