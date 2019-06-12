#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pathlib

import PIL.Image as Image

import numpy as np
import numpy.linalg as LA


from PIL_ext import *


if __name__ == '__main__':

    def inside(point):
        if point == (0,0):
            return True
        else:
            x, y = point
            if 16 >= x >= 0:        
                if y >= 0:
                    t = np.arcsin((x / 16) ** (1/3))
                    return y <= np.round(13*np.cos(t) - 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))
                else:
                    t = np.pi - np.arcsin((x /16) ** (1/3))
                    return y >= np.round(13*np.cos(t) - 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))
            elif x > 16:
                return False
            else:
                return inside((-x, y))


    # def heart(t):
    #     return (16*(np.sin(t))**3, (13*np.cos(t) - 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)))

    user = pathlib.Path('faces')

    nrow = 15
    ncol = 18

    toImage = Background(nrow, ncol, 60)  # 先生成头像集模板

    imgs = []
    for f in user.iterdir():
        if f.suffix == '.jpg':
            try:
                im = Image.open(f)
                imgs.append(im)
            except:
                pass

    coords = []

    for x in range(ncol):
        for y in range(nrow):
            point = (x-ncol//2 + 1, nrow//2-y - 2)
            if inside(point):
                coords.append((x,y))

    # coords = tuple((int(x), int(y)) for x, y in zip(xs, ys))

    toImage.draw(imgs, coords)
    toImage.save(user / "fx.jpg")
