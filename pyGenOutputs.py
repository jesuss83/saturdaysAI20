import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import csv as csv
import cv2
import sys;
import os.path
import os
import statistics 

TOTAL_WIDTH = 288.00;
TOTAL_HEIGHT = 190.00;
TOTAL_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT;
TOTAL_COLORS = 16777216.00;
TILE_WIDTH = 16;
TILE_HEIGHT = 10;

def genImgCsv(center,imgFile):    
    img = cv2.imread(imgFile);
    print('Generating point for: ' + imgFile);
    row=0;
    col=0;
    tileCols = 16;
    tileRows = 10;
    index=0;
    centerX = int(center) % 288;
    centerY = int(center) // 288;
    print(str(centerX)+','+str(centerY));
    
    img.itemset((centerY,centerX,2),255);
    img.itemset((centerY,centerX,1),0);
    img.itemset((centerY,centerX,0),0);

    head, tail = os.path.split(imgFile);    
    pointFile = tail.split('.')[0]+'_p.png';
    pointFile = head + '/' + pointFile;
    
    cv2.imwrite(pointFile,img)

def getSprite(center, imgFile):
  img = cv2.imread(imgFile)
  centerX = int(center) % 288;
  centerY = int(center) // 288;
  sw = 100
  sh = 100

  head, tail = os.path.split(imgFile);    
  pointFile = tail.split('.')[0]+'_p.png';
  pointFile = head + '/' + pointFile;
  spr1File = tail.split('.')[0]+'_sp1.png';
  spr1File = head + '/' + spr1File;
  spr2File = tail.split('.')[0]+'_sp2.png';
  spr2File = head + '/' + spr2File;
  spr3File = tail.split('.')[0]+'_sp3.png';
  spr3File = head + '/' + spr3File;
  spr4File = tail.split('.')[0]+'_sp4.png';
  spr4File = head + '/' + spr4File;

  sprFiles = [spr1File,spr2File,spr3File,spr4File]  

  boxes = np.array(1)
  boxes2 = np.array(1)
  boxes3 = np.array(1)
  boxes4 = np.array(1)

  boxes = [[centerX-40,centerY-40,centerX+20,centerY+20]]  
  imgBox1 = cv2.rectangle(img,(boxes[0][0],boxes[0][1]),(boxes[0][2],boxes[0][3]),(0,0,255),1)  
  boxes2 = [[boxes[0][0]+60,boxes[0][1],boxes[0][2]+60,boxes[0][3]]]
  imgBox2 = cv2.rectangle(imgBox1,(boxes2[0][0],boxes2[0][1]),(boxes2[0][2],boxes2[0][3]),(255,0,0),1)
  boxes3 = [[boxes[0][0]-10,boxes[0][1]-20,boxes[0][2]+60,boxes[0][3]+10]]   
  imgBox3 = cv2.rectangle(imgBox2,(boxes3[0][0],boxes3[0][1]),(boxes3[0][2],boxes3[0][3]),(0,255,0),1)             
  boxes4 = [[boxes[0][0]-20,boxes[0][1]-10,boxes[0][2]+10,boxes[0][3]+10]]   
  imgBox4 = cv2.rectangle(imgBox3,(boxes4[0][0],boxes4[0][1]),(boxes4[0][2],boxes4[0][3]),(255,255,255),1)                   

  print('Generating boxes/point for: ' + imgFile)
  imgCircle = cv2.circle(imgBox4,(centerX,centerY), 2, (0,0,255), -1)
  cv2.imwrite(pointFile,imgCircle)

  print('Generating sprites for: ' + imgFile)

  bx = [
         [boxes[0][1], boxes[0][3], boxes[0][0], boxes[0][2]],
         [boxes2[0][1],boxes2[0][3],boxes2[0][0],boxes2[0][2]],
         [boxes3[0][1],boxes3[0][3],boxes3[0][0],boxes3[0][2]],
         [boxes4[0][1],boxes4[0][3],boxes4[0][0],boxes4[0][2]]

         ]

  for i in range(0,4):
    spr = cv2.imread(imgFile)  
    y1 = bx[i][0]
    y2 = bx[i][1]
    x1 = bx[i][2]
    x2 = bx[i][3]

    if (y1 < 0): y1 = 0
    if (y2 < 0): y2 = 0
    if (x1 < 0): x1 = 0
    if (x2 < 0): x2 = 0

    if (y2>0 and x2>0):
     spr = spr[y1:y2,x1:x2]
     cv2.imwrite(sprFiles[i],spr)
    else: 
     print('negative sprite: ' + sprFiles[i])

  


def getCenterPoint(centerPath): 
    center = 0.00;
    px = 0.00;
    py = 0.00;
    with open(centerPath, 'rU') as csvfile:
                csvline = csv.reader(csvfile);
                for row in csvline:
                             centerStr = row[0].replace('[','').replace(']','')                             
                             pxStr= row[1].replace('[','').replace(']','')
                             pyStr= row[2].replace('[','').replace(']','')
                             center = float(centerStr)
                             px = float(pxStr)
                             py = float(pyStr)
                    

    
    return center,px,py

def main(inputCenter,imgFile):
    center,px,py= getCenterPoint(inputCenter)
    print(str(center)+','+str(px)+','+str(py))
    getSprite(center,imgFile)
    
def main_entry():
    if (len(sys.argv)<1):
        print('set path');
        return

    dir_path = sys.argv[1];
    dir_path = dir_path.replace("\\","/");

    isdir = os.path.isdir(dir_path)  

    if (isdir == False):
     print('set existing path');
     return;

    print('Generatings outputs for:' + dir_path);
    processDir(dir_path);
    print('Generating outputs done for:' + dir_path);


def processDir(dir_path):

    imgFiles = [f for f in os.listdir(dir_path) if os.path.isfile(dir_path+'/'+f) and f.split('.')[1] == 'png'];       

    for imgFile in imgFiles:
        head, tail = os.path.split(imgFile);    
        fileName = tail.split('.')[0];
        dirForFile = dir_path+'/'+fileName;
        isdir = os.path.isdir(dirForFile)        
        inputPath= dirForFile+'/'+tail
        inputCenter = dirForFile + '/' + fileName + '_p.csv';
        inputFile = dirForFile+'/'+imgFile;

        if (isdir!=False):         
            print('Generating outputs for: ' + imgFile);
            
            main(inputCenter,inputFile);
        else:
            print('path not found: ' + dirForFile)

       

main_entry();

