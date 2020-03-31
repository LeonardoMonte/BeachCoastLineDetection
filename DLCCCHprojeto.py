import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

def printImage(img):
    fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.axis("off")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def printImageGray(img):
    fig=plt.figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()

def conect8(m, i,j,value):
    if i == 0 and j == 0:
        if m[i][j+1] == value:
            return True
        elif m[i+1][j+1] == value:
            return True
        elif m[i+1][j+1] == value:
            return True
        else:
            return False
    elif i == 0 and j == m.shape[1]-1:
        if m[i][j-1] == value:
            return True
        elif m[i+1][j-1] == value:
            return True
        elif m[i+1][j] == value:
            return True
        else:
            return False
    elif i == m.shape[0]-1 and j == 0:
        if m[i-1][j-1] == value:
            return True
        elif m[i-1][j+1] == value:
            return True
        elif m[i][j+1] == value:
            return True
        else:
            False
    elif i == m.shape[0]-1 and j == m.shape[1]-1:
        if m[i-1][j] == value:
            return True
        elif m[i-1][j-1] == value:
            return True
        elif m[i][j-1] == value:
            return True
        else:
            return False
    elif i == 0 and (j > 0 and j < m.shape[1]-1):
        if m[i][j-1] == value:
            return True
        elif m[i+1][j-1] == value:
            return True
        elif m[i+1][j] == value:
            return True
        elif m[i+1][j+1] == value:
            return True
        elif m[i][j+1] == value:
            return True
        else:
            return False
    elif (i > 0 and i < m.shape[0]-1) and j ==0:
        if m[i-1][j] == value:
            return True
        elif m[i-1][j+1] == value:
            return True
        elif m[i][j+1] == value:
            return True
        elif m[i+1][j+1] == value:
            return True
        elif m[i+1][j] == value:
            return True
        else:
            return False
    elif (i > 0 and i < m.shape[0]-1) and j == m.shape[1]-1:
        if m[i-1][j] == value:
            return True
        elif m[i-1][j-1] == value:
            return True
        elif m[i][j-1] == value:
            return True
        elif m[i+1][j-1] == value:
            return True
        elif m[i+1][j] == value:
            return True
        else:
            return False
    elif i == m.shape[0]-1 and (j > 0 and j < m.shape[1]-1):
        if m[i][j-1] == value:
            return True
        elif m[i-1][j-1] == value:
            return True
        elif m[i-1][j] == value:
            return True
        elif m[i-1][j+1] == value:
            return True
        elif m[i][j+1] == value:
            return True
        else:
            return False
    else:
        if m[i][j-1] == value:
            return True
        elif m[i+1][j-1] == value:
            return True
        elif m[i+1][j] == value:
            return True
        elif m[i+1][j+1] == value:
            return True
        elif m[i][j+1] == value:
            return True
        elif m[i-1][j-1] == value:
            return True
        elif m[i-1][j] == value:
            return True
        elif m[i-1][j+1] == value:
            return True
        else:
            return False

def linhaCosta(img):

    #img = pipeline(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#converte para hsv
    hsv = cv2.GaussianBlur(hsv, (5, 5), 5)
    b, g ,r = cv2.split(hsv)
    zeros = np.zeros(img.shape[:2], dtype = "uint8")
    b_3d = cv2.merge([b, zeros, zeros])
    ret1,th1 = cv2.threshold(b,65,255,cv2.THRESH_BINARY)#65
    im2, contours, hierarchy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    areas = []
    image_saida = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    for i in range(0,len(contours)):
        area = cv2.contourArea(contours[i])
        areas.append(area)
    indiceAreaMaior = areas.index(max(areas))
    saida = cv2.drawContours(image_saida, contours, indiceAreaMaior, (255), -1)
    s = cv2.bitwise_not(saida)
    im3, contours2, hierarchy2 = cv2.findContours(s,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas2 = []
    image_saida2 = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    contours2F = []
    for i in range(0,len(contours2)):
        area2 = cv2.contourArea(contours2[i])
        if area2 > 3000:
            contours2F.append(contours2[i])
        areas2.append(area2)
    indiceAreaMaior2 = areas2.index(max(areas2))
    saida2 = cv2.drawContours(image_saida2, contours2F, -1, (255), -1)
    mask_inv = cv2.bitwise_not(saida2)
    imgOut = img.copy()
    for i in range(mask_inv.shape[0]):
        for j in range(mask_inv.shape[1]):
            if(mask_inv[i][j] == 0 and conect8(mask_inv,i,j, 255)):
                imgOut[i][j] = [0,255,0]
    return imgOut


def experimentos(imgs, labels,type = 'lower'):

    if type == 'media':
        array = [diferencaCostaMedia(im,lb) for im,lb in zip(imgs,labels)]

    if type == 'lower':
        array = [diferencaCostalower(im,lb) for im,lb in zip(imgs,labels)]

    return np.array(array).ravel(),np.mean(array),np.std(array)

def diferencaCostaMedia(imggerada, label):
    linhaV = []
    linhaE = []

    for j in range(imggerada.shape[1]):
        cont = 0
        aux = []
        for i in range(imggerada.shape[0]):
            if np.all(imggerada[i][j] == [0, 255, 0], axis=0):
                cont += 1
                aux.append(i)
        if cont == 1:
            linhaV.append(aux[-1])
        else:
            value = int(round(np.mean(aux)))
            linhaV.append(value)

    for j in range(label.shape[1]):
        for i in range(label.shape[0]):
            if np.all(label[i][j] == [0, 0, 0], axis=0):
                linhaE.append(i)
                break

    soma = 0
    for i in range(len(linhaV)):
        soma = soma + abs(linhaE[i] - linhaV[i]) #np.sum([abs(le - lv) for le, lv in zip(linhaE, linhaV)])

    return soma / len(linhaE)

def diferencaCostalower(imggerada, label):
    linhaV = []
    linhaE = []

    for j in range(imggerada.shape[1]):
        cont = 0
        aux = []
        for i in range(imggerada.shape[0]):
            if np.all(imggerada[i][j] == [0, 255, 0], axis=0):
                cont += 1
                aux.append(i)
        if cont == 1:
            linhaV.append(aux[-1])
        else:
            value = aux[np.argmax(aux)]
            linhaV.append(value)

    for j in range(label.shape[1]):
        for i in range(label.shape[0]):
            if np.all(label[i][j] == [0, 0, 0], axis=0):
                linhaE.append(i)
                break

    soma = 0
    for i in range(len(linhaV)):
        soma = soma + abs(linhaE[i] - linhaV[i]) #np.sum([abs(le - lv) for le, lv in zip(linhaE, linhaV)])

    return soma / len(linhaE)

def equalize_hist(img2):

    img = img2.copy()
    for c in range(0, 2):
       img[:,:,c] = cv2.equalizeHist(img[:,:,c])


    return img

def histogram_equalize(img):
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        value = int(-value)
        lim = 0 + value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def pipeline(img):

    img = increase_brightness(img, value=-102)
    img = equalizeRGB(img)
    #img = cv2.medianBlur(img,3)
    #img = cv2.blur(img,(3,3))

    return img

def equalizeRGB(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def generateresults():

    path = 'DLCCCH/projeto/X/*'
    pathout = 'DLCCCH/projeto/Y/*'
    path2 = 'DLCCCH/projeto/resultado/'

    X = [cv2.imread(img) for img in sorted(glob.glob(path))]
    names = [os.path.basename(img) for img in sorted(glob.glob(path))]
    nameslabel = [os.path.basename(img) for img in sorted(glob.glob(path2))]
    label = [cv2.imread(img) for img in sorted(glob.glob(path2))]

    # for img,name in zip(X,names):
    #     cv2.imwrite(pathout+name.replace('.jpg','_out.jpg'),linhaCosta(img))

    return [linhaCosta(im) for im in X],names,label,nameslabel


imgs,names,label,nameslabel = generateresults()


print(names)
print(nameslabel)


array,media,std = experimentos(imgs,label,type='lower')

print(array)
print(media)
print(std)
