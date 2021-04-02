import numpy as np
import cv2
import logging
import torch.nn as nn
import torch

def showpic(pic):              #顯示圖片
    cv2.imshow('RGB', pic)     #顯示 RGB 的圖片
    cv2.waitKey(0)             #有這段才不會有bug

def readpic(p):                #讀入圖片
    return cv2.imread(p)
    
def savepic(img, p):           #儲存圖片
    cv2.imwrite(p, img)
    
#讀取多張照片
def path2pic(img):
    pic = []
    for i in img:
        p = readpic(i)
        pic.append(p)
    return pic

# 處理多張圖片變成 tensor
def pic2tensor(img_list):
    adap = nn.AdaptiveMaxPool2d((1024, 1024))
    pic = []
    for i in img_list:
        a, b, c = i.shape
        p1 = torch.tensor(i, dtype=torch.float).permute(2, 0, 1)
        p2 = p1.view(1, 3, a, b)
        p3 = adap(p2)
        pic.append(p3)
    outputs = torch.cat(pic, dim=0)
    return outputs

def sumlist(l, n):
    c = 0
    for i in l:
        c += i
    return c / n

def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)
    
    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)
    
    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    
    return my_logger

#logger.disabled = True  #暫停 logger
#logger.handlers  # logger 內的紀錄程序
#logger.removeHandler  # 移除紀錄程序
#logger.info('xxx', exc_info=True)  # 紀錄堆疊資訊

# top  n  %  accuracy
# both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
def top_n_accuracy(preds_array, truths, n):
    N = preds_array.shape[0]
    best_n = np.argsort(preds_array, axis=1)[:, -n:]
    successes = 0
    for i in range(preds_array.shape[0]):
      if truths[i] in best_n[i, :]:
        successes += 1
    return float(successes / N)

def sigmoid(m):
    e = np.exp((-1) * m)
    output = 1 / (1 + e)
    return output