import numpy as np
import pandas as pd
import csv as csv
import cv2
import sys;
import os.path
import os
import statistics 
import shutil

TOTAL_WIDTH = 288.00;
TOTAL_HEIGHT = 190.00;
TOTAL_PIXELS = TOTAL_WIDTH * TOTAL_HEIGHT;
TOTAL_COLORS = 16777216.00;
TILE_WIDTH = 16;
TILE_HEIGHT = 10;
GENERATE_COLOR_TILES = 0;   
   
def hex_to_rgb(value):
    value = value.lstrip('#')    
    lv = len(value)    
    if (lv // 3 == 0):
        return (0,0,0);
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))    

def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

def _getColorTile(tile,tileNum):
    
    tileCols = 16;
    tileRows = 10;    
    color = 0;
    index=0;
    rgbs = [];    
    
    #RGB = (R*65536)+(G*256)+B , (when R is RED, G is GREEN and B is BLUE)
    for row in range(tileRows):          
            for col in range(tileCols):                        
                        r = tile[row-1,col-1][0];
                        g = tile[row-1,col-1][1];
                        b = tile[row-1,col-1][2];
                        rgb =  (r*65536) + (g*256) + b;                          
                        rgbs.append(rgb);                        
                        index = index + 1;               
                     
    
    avgColor = find_max_mode(rgbs) #statistics.mode(rgbs);    
    
    return avgColor;
    
    
def _getTile(img, tileX,tileY):
    
    startingX = (tileX)*TILE_WIDTH;
    startingY = (tileY)*TILE_HEIGHT;
    
    endingX = startingX+TILE_WIDTH;
    endingY = startingY+TILE_HEIGHT;
    
    if (endingX >= 288):
            endingX = 287;
    
    if (endingY >= 190):
            endingY = 189;        
    
    tile = img[startingY:endingY,startingX:endingX];
    
    return tile;
    
    
def _getTiles(img):               
    cols = 18;
    rows = 19;
    tiles = [];
        
    for r in range(rows):            
        for c in range(cols):        
            tile = _getTile(img,c,r);
            tiles.append(tile);                            
        
    
    return tiles;    


def getTilesForInput(imgFile):
    tiles = [];        
    img = cv2.imread(imgFile);
    
    tiles = _getTiles(img);    
    tileNum = 0;    
    
    tileLine = '';
    tileData = '';
    
    for tile in tiles:        
        tileAvgColor = _getColorTile(tile,tileNum);        
        tileData = tileData+ str(tileAvgColor) + ",";        
        tileNum = tileNum + 1;
    
    #print ('Total tiles for: ' + imgFile + ':' + str(tileNum));
    tileLine = tileLine+tileData[0:len(tileData)-1];
    
    return tileLine;


def genImgFromInputValues(imgColors,imgFile):    
    img = cv2.imread(imgFile);    
    row=0;
    col=0;
    tileCols = 16;
    tileRows = 10;
    index=0;        
    
    for avgColor in imgColors: 
            avghx = hex(int(avgColor));                        
            avghx = str(avghx);                       
            avghx = '#'+avghx[2:len(avghx)];      
            avgrgb = hex_to_rgb(avghx);
            ravg = avgrgb[0];
            gavg = avgrgb[1];
            bavg = avgrgb[2];
            
            row = index // 18;
            col = index % 18;
            
            startingX = (col)*TILE_WIDTH;
            startingY = (row)*TILE_HEIGHT;
    
            endingX = startingX+TILE_WIDTH;
            endingY = startingY+TILE_HEIGHT;                       
    
            if (endingX >= 288):
                endingX = 287;
    
            if (endingY >= 190):
                endingY = 189;        
            
            tile = img[startingY:endingY,startingX:endingX];
            
            for row in range(tileRows):
                   for col in range(tileCols):
                            tile[row-1,col-1][0] = ravg;
                            tile[row-1,col-1][1] = gavg;
                            tile[row-1,col-1][2] = bavg;                       
            
            img[startingY:endingY,startingX:endingX] = tile;      
            index = index + 1;
            
    
    return img;                

def getImgFromInputs(inputFile,imgFile):
    total=0;
    head, tail = os.path.split(inputFile);    
    with open(inputFile, 'rU') as csvfile:
                csvline = csv.reader(csvfile);
                for row in csvline:
                    if (total==1):                             
                             img=genImgFromInputValues(row,imgFile);
                             cv2.imwrite(head+'/'+tail.split('.')[0]+'_t.png',img); 
                    total=total+1;

def genImgTilesetInput(imgFile):
    total = 0;
    head, tail = os.path.split(imgFile);    
    inputFile = head+'/'+tail.split('.')[0]+'.csv';

    fileHdr = '';

    for i in range(0,342):
        fileHdr = fileHdr + 'Tile' + str(i) + ',';

    ftileSet = open(inputFile,'w');
    tileLine=getTilesForInput(imgFile);
    ftileSet.write(fileHdr[0:len(fileHdr)-1]+'\r');
    ftileSet.write(tileLine+'\r');    
    ftileSet.close();            
    return inputFile;

def main(imgFile):    
    inputFile = genImgTilesetInput(imgFile);    
    getImgFromInputs(inputFile,imgFile);    

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

    print('Generatings inputs for:' + dir_path);
    processDir(dir_path);
    print('Generating inputs done for:' + dir_path);

def create_dir(dir_path,imgFile):
    
    head, tail = os.path.split(imgFile);    
    fileName = tail.split('.')[0];
    dirForFile = dir_path+'/'+fileName;
    isdir = os.path.isdir(dirForFile)  

    if (isdir==False):
      os.mkdir(dirForFile)

    shutil.copyfile(dir_path+'/'+imgFile,dirForFile+'/'+imgFile);

    return dirForFile;


def processDir(dir_path):

    imgFiles = [f for f in os.listdir(dir_path) if os.path.isfile(dir_path+'/'+f) and f.split('.')[1] == 'png'];       


    for imgFile in imgFiles:
        dirForFile = create_dir(dir_path,imgFile);
        print('Generating inputs/tilesets for: ' + dir_path+'/'+imgFile);
        inputFile = dirForFile+'/'+imgFile;
        main(inputFile);

        

main_entry();
