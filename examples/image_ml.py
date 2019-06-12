#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np
import pandas as pd
import numpy.linalg as LA

from PIL import Image, ImageMath
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift

import PIL_ext

def imsvd(im, n_components=2, *args, **kwargs):
    """svd for image
    
    Arguments:
        filename {str} -- an image
        n_components {int} -- number of components
    
    Keyword Arguments:
        output {bool} -- output the transformed data (default: {False})

    Example:
    Scripts/DataAnalysis/pca.py --filename Folders/临时文件夹/光谱值数据.xlsx --n_components 2
    """

    if isinstance(im, str):
        im = Image.open(im)

    if im.mode in 'RGB':
        data = np.asarray(im, dtype=np.float64)
        for k in (0,1,2):
            datak = data[:,:, k]
            pca = PCA(n_components=n_components)
            pca.fit(datak)
            data[:,:, k] = pca.inverse_transform(pca.transform(datak))
        im = Image.fromarray(data.astype('uint8'))
        im = im.convert('RGB')
    else:
        data = np.asarray(Image.open(path).convert('L'))
        pca = PCA(n_components=n_components)
        pca.fit(data)
        y = pca.inverse_transform(pca.transform(data))
        im = Image.fromarray(y).convert('L')
    im.save('_' + path.name)


def eigenimages(images, n_components=10, n_eigen=10):
    """Eigen images
    
    Arguments:
        images {List[Image]} -- list of images
    
    Keyword Arguments:
        n_components {number} -- number of components (default: {10})
        n_eigen {number} -- number of eigen images (default: {10})
    """

    size = images[0].size
    mode = images[0].mode
    data = PIL_ext.tomatrix(images, 'col')
    pca = PCA(n_components=n_components)
    pca.fit(data)
    eigens = pca.transform(data)
    return [PIL_ext.toimage(eigens[:, i], size, mode=mode) for i in range(n_eigen)]


def ncimages(images, n_components=10, n_eigen=10):
    """Non-negtive images

    apply NMF
    
    Arguments:
        images {List[Image]} -- list of images
    
    Keyword Arguments:
        n_components {number} -- number of components (default: {10})
        n_eigen {number} -- number of eigen images (default: {10})
    """

    size = images[0].size
    mode = images[0].mode
    data = PIL_ext.tomatrix(images, 'col')
    nmf = NMF(n_components=n_components)
    nmf.fit(data)
    eigens = nmf.transform(data)
    eigens *= 256
    return [PIL_ext.toimage(eigens[:, i], size, mode=mode) for i in range(n_eigen)]


@PIL_ext.imageOp(mode='L')
def quantize(data, N=2):
    def convert_(x, M = 255):
        for k in range(N):
            if (k / N) * M <x <= ((k+1)/N) * M:
                return (k / N) * M
        return 0
    convert = np.frompyfunc(convert_, 1, 1)
    return convert(data)


import cv2

def imageconv(image, fil=None):
    image = np.asarray(image, dtype=np.float64)
    res = cv2.filter2D(image, -1, fil)
    return Image.fromarray(res.astype('uint8'))


sobel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
sobel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
sobel = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
prewitt_y = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]))
prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
mean = np.ones((3, 3)) / 9
mean_1 = mean - 1
roberts_x = np.array([[1,0],[0,-1]])
roberts_y = np.array([[0,-1],[1,0]])

# with Image.open("test/1.jpg") as im:
#     im = imageconv(im, fil=laplacian)
#     im.save("res.jpg")

def preprocess(data):
    N = data.shape[1]
    a = int(np.sqrt(N))
    a = 50
    ft = PIL_ext.FiltTransformer(shape=(a,a), channels=['a','d'])
    ft.fit(data)
    return ft.transform(data)

from sklearn.mixture import *
def image_clustering(images, preprocess=None, method=KMeans, *args, **kwargs):
    data = PIL_ext.tomatrix(images)
    if preprocess:
        data = preprocess(data)

    n_clusters=5
    km = method(n_clusters, *args, **kwargs)

    km.fit(data)
    # size = images[0].size
    # centers = [PIL_ext.toimage(c, size) for c in km.cluster_centers_]
    aa=km.predict(data)
    for c in range(n_clusters):
        ims=[im.resize((50,50)) for im, a in zip(images, aa) if a==c]
        im = PIL_ext.sqstack(ims)
        im.save(p/ ('clusters/class %d.jpg' % c))

p =  pathlib.Path('faces01')
images = PIL_ext.get_images(folder=p, op=lambda x:x.resize((50,50)))
image_clustering(images, preprocess, GaussianMixture)

# for k, a in enumerate(images):
#     a = quantize(a)
#     a.save('faces01/face%d.jpg'%k)

# eigens = ncimages(images, n_eigen=5)
# eigens = [eigen.resize((50,50)) for eigen in eigens]
# for k, e in enumerate(eigens):
#     e.save(p/('eigen/nc %d.jpg' % k))

