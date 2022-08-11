# Conv Feature with perception learning

# 1.  training dataset
此任務為圖片分類任務，資料皆為狗的照片，並可以分為50種不同的狗，下方照片為舉例圖片

![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111277_160.JPEG)![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111500_113.jpg)



資料可分為:

63325張訓練資料

450張測試資料

450張驗證資料



# 2. Image feature extractor
本篇使用U-net的結構搭配Convolution訓練模型提取圖片特徵，實作時以pytorch為主要套件。

U-net結構為先將圖片轉成較低維度的向量(encoder)，再由此向量還原回原圖(decoder)，模型目標為使原圖與還原後的圖盡量接近。

訓練完後已encoder後的向量作為圖片的特徵提取。

參考自:

[U-net]: https://heartbeat.comet.ml/deep-learning-for-image-segmentation-u-net-architecture-ff17f6e4c1cf	"Deep Learning for Image Segmentation: U-Net Architecture"

![U-net](https://github.com/ss9636970/convFeature-perception_learning/blob/main/readme/U-net.png)

# 3. Perception

本篇使用Perception作為圖片分類器，perception為線性分類器，可以視為單層的神經網路演算法，並在最後接上softmax激活函式。

本篇使用cross entropy loss為模型優化的目標函式的標準值，並用梯度下降法更新模型參數。



# 4. 程式碼說明

moduleClass.py 為Perception模型定義程式碼

CNN_encoder.py 為特徵提取模型

funcion.py 為程式中運用到的函式。

CNN_encoder.ipynb為訓練特徵提取執行程式

main.ipynb為執行perception模型訓練程式，當中包括讀取資料及特徵提取的執行程式碼





