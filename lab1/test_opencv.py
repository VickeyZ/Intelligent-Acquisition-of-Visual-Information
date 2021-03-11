import numpy as np 
import cv2 as cv
import os
from PIL import Image
from matplotlib import pyplot as plt
import csv

# bmp 转换为jpg
def bmpToJpg(file_path):
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = fileName[0:fileName.find(".")] + ".jpg"
        print(newFileName)
        im = Image.open(file_path + "\\" + fileName)
        im.save(file_path + "\\jpg\\" + newFileName)

def main():
    #bmpToJpg("D:\Courses\\2020\iavi\lab1\img\img_bmp")
    cal_hsv()


def cal_hsv():
    exposure = []
    gain = []
    average_v = []
    average_s = []
    average_h = []

    # 读入exposure & gain
    with open("D:\Courses\\2020\iavi\lab1\week1_raw_metadata.csv") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            exposure.append(row[0])
            gain.append(row[1])

    file_path = "D:\Courses\\2020\iavi\lab1\img\\img_bmp\\jpg\\"
    
    i = 0
    for fileName in os.listdir(file_path):
        img = cv.imread(fileName) 
        hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV) 
        H, S, V = cv.split(hsv)
        v = V.ravel()[np.flatnonzero(V)]   #亮度非零的值
        s = S.ravel()[np.flatnonzero(S)] 
        h = H.ravel()[np.flatnonzero(H)] 
        average_v.append(sum(v)/len(v))         #平均亮度
        average_s.append(sum(s)/len(s))
        average_h.append(sum(h)/len(h))
        # print(average_v)

    with open('D:\Courses\\2020\iavi\lab1\\hsv_statistic.csv','w')as f:
        f_csv = csv.writer(f)
        for i in range(0,17):
            row = [i+1,exposure[i],gain[i],average_h[i],average_s[i],average_v[i]]
            f_csv.writerow(row)

 
if __name__ == '__main__':
    main()