import numpy as np
import cv2
from sklearn.cluster import KMeans
import time
import math
def MSE(image,newImage):
    """
    計算均方差
    """
    h,w,d=np.shape(image)
    image=image.reshape(h*w,d).astype(int)
    newImage=newImage.reshape(h*w,d).astype(int)
    total=0
    for i in range(h*w):
        for j in range(d):
            total+=(image[i][j]-newImage[i][j])**2
    total/=(h*w*d)
    return total

def PSNR(image,newImage):
    """
    計算峰值訊噪比
    image:原圖
    newImage:壓縮後的圖
    """
    mse=MSE(image,newImage)
    maxi=255
    return   round(10*math.log10(maxi**2/mse),2)
    

image=cv2.imread("./lena.png")
h,w,d=np.shape(image)
vector=np.array(image.reshape(h*w,d))



cv2.imshow("Image",image)

#
instance=[["2",KMeans(2)],["8",KMeans(8)],["256",KMeans(256)]]
for name,kmeans in instance:
    t=time.time()#記錄生成編碼簿到生成壓縮圖的時間
    kmeans.fit(vector)#使用Kmeans演算法來產生編碼簿
    codeBook=np.array(kmeans.cluster_centers_).astype(np.uint8)#取出編碼簿
    pred=kmeans.predict(vector)#對圖片編碼
    np.savetxt(f"{name}_CodeBook.csv", codeBook,fmt='%d',delimiter=",")#將編碼簿存成CSV檔

    #下面四行解壓縮
    newImage=[]
    for i in pred:
        newImage.append(codeBook[i])
    newImage=np.array(newImage).reshape(h,w,d)

    print(f"{name} PSNR={PSNR(image,newImage)} dB")#顯示峰值訊噪比
    print(f"RunTime:{round(time.time()-t,1)} sec\n")#顯示花費總時間
    
    cv2.imshow(f"{name}_newImage",newImage)#顯示壓縮圖
    cv2.imwrite(f"{name}_newImage.png",newImage)#儲存壓縮圖


cv2.waitKey(0)
cv2.destroyAllWindows()