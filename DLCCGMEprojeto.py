import numpy as np
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import glob
import os

def printImage(img):
    fig = plt.figure(figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.axis("off")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def printImageGray(img):
    fig = plt.figure(figsize=(8, 8), dpi=120, facecolor='w', edgecolor='k')
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.show()


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


def getIdNo(i, j, img):
    g = (i * img.shape[1]) + j + 2
    return g


def init_vertices(grafo, im):
    grafo.add_node(0, i=-1, j=-1)  # Nó inicial s
    grafo.add_node(1, i=-1, j=-1)  # Nó Final t
    cont = 2
    N = im.shape[1] - 1
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            # y = functionCostY(i, j, N)
            grafo.add_node(cont, i=i, j=j, weight=im[i][j])
            cont += 1


def init_link(grafo, im, t, u):
    for j in range(im.shape[1]):
        for h in range(im.shape[0]):
            # t = 5
            g1 = getIdNo(h, j, im)
            if h > 0 and h < im.shape[0] - 1 and j < im.shape[1] - 1:

                g2 = getIdNo(h - 1, j + 1, im)
                g3 = getIdNo(h, j + 1, im)
                g4 = getIdNo(h + 1, j + 1, im)

                c12 = im[h][j] * t + im[h - 1][j] * t + u
                c13 = im[h][j] * t + im[h][j] * t
                c14 = im[h][j] * t + im[h + 1][j] * t + u
                grafo.add_edge(g1, g2, weight=c12)
                grafo.add_edge(g1, g3, weight=c13)
                grafo.add_edge(g1, g4, weight=c14)
                if j == 0:
                    c = (h - im.shape[1] // 2) ** 2
                    grafo.add_edge(0, g1, weight=c)
            elif h == 0 and j < im.shape[1] - 1:

                g3 = getIdNo(h, j + 1, im)
                g4 = getIdNo(h + 1, j + 1, im)

                c13 = im[h][j] * t + im[h][j] * t
                c14 = im[h][j] * t + im[h + 1][j] * t + u
                grafo.add_edge(g1, g3, weight=c13)
                grafo.add_edge(g1, g4, weight=c14)
                if j == 0:
                    c = (h - im.shape[1] // 2) ** 2
                    grafo.add_edge(0, g1, weight=c)
            elif h == im.shape[0] - 1 and j < im.shape[1] - 1:
                g2 = getIdNo(h - 1, j + 1, im)
                g3 = getIdNo(h, j + 1, im)
                c12 = im[h][j] * t + im[h - 1][j] * t + u
                c13 = im[h][j] * t + im[h][j] * t
                grafo.add_edge(g1, g2, weight=c12)
                grafo.add_edge(g1, g3, weight=c13)
                if j == 0:
                    c = (h - im.shape[1] // 2) ** 2
                    grafo.add_edge(0, g1, weight=c)
            elif j == im.shape[1] - 1:
                c = (h - im.shape[1] // 2) ** 2
                grafo.add_edge(g1, 1, weight=c)

def equalizeRGB(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

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


def pipeline(img):
    img = increase_brightness(img, value=-100)
    img = equalizeRGB(img)

    return img

def encontra_linha(_img, t, u):

    _img = pipeline(_img)
    hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)  # converte para hsv
    hsv = cv2.GaussianBlur(hsv, (9, 9), 0)  # melhor resultado com 9x9
    h, s, v = cv2.split(hsv)
    edge = cv2.Canny(h, 50, 40)  # melhor resultado com 50 x40
    edge_inv = cv2.bitwise_not(edge)
    DG = nx.DiGraph()
    init_vertices(DG, edge_inv)
    init_link(DG, edge_inv, t, u)
    img2 = _img.copy()
    length, path = nx.single_source_dijkstra(DG, 0, 1)
    newPath = path[1:len(path) - 1]
    for i in newPath:
        l = DG.nodes[i]['i']
        c = DG.nodes[i]['j']
        img2[l][c] = [0, 255, 0]
    return img2

def generateresults():

    path = 'DLCCGME/projeto/X/*'
    path2 = 'projeto/Y/*'
    pathout = 'DLCCGME/projeto/resultado/'

    X = [cv2.imread(img) for img in sorted(glob.glob(path))][6:]
    label = [cv2.imread(img) for img in sorted(glob.glob(path2))][6:]
    names = [os.path.basename(img) for img in sorted(glob.glob(path))][6:]
    labelnames = [os.path.basename(img) for img in sorted(glob.glob(path2))][6:]

    # for img,name in zip(X,names):
    #     cv2.imwrite(pathout+name.replace('.jpg','_out.jpg'),encontra_linha(img,10,100))

    return [encontra_linha(img,10,100) for img in X],names,label,labelnames

imgs,names,label,nameslabel = generateresults()
cv2.imshow('a',imgs[3])
cv2.waitKey()

print(names)
print(nameslabel)


array,media,std = experimentos(imgs,label,type = 'lower')

print(array)
print(media)
print(std)