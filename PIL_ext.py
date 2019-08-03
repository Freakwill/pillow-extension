# -*- coding: utf-8 -*-

import itertools
import pathlib

import numpy as np
import numpy.linalg as LA
from PIL import Image


def tovector(image, k=None):
    # image -> vector
    data = np.asarray(image, dtype=np.float64)
    if k:
        return data[:,:, k].flatten()
    else:
        return data.flatten()

def tomatrix(images, way='row'):
    # image -> matrix
    if way in {'r', 'row'}:
        return np.row_stack([tovector(image) for image in images])
    elif way in {'c', 'col', 'column'}:
        return np.column_stack([tovector(image) for image in images])

def toimage(vector, size, mode='RGB'):
    # vector -> image
    if mode == 'RGB':
        if len(size)==2:
            size += (3,)
        return Image.fromarray(vector.reshape(size).astype('uint8')).convert(mode)
    else:
        return Image.fromarray(vector.reshape(size).astype('uint8')).convert(mode)

# from sklearn.preprocessing import FunctionTransformer
# class FiltTransformer(FunctionTransformer):
#     '''Transform images to vectors
#     '''
#     def __init__(self, shape, channels, *args, **kwargs):
#         def func(X):
#             return np.column_stack([np.row_stack([filt(x.reshape(shape), channel).flatten() for x in X])
#                 for channel in channels])
#         super(FiltTransformer, self).__init__(func=func, *args, **kwargs)

#     def fit(self, X):
#         """
#         Transform images to vectors

#         Arguments:
#             X {list|array} -- images or a folder where images are stored
#         """
#         if isinstance(X, pathlib.Path):
#             images = []
#             for f in X.iterdir():
#                 try:
#                     img = Image.open(f).resize((200, 200))
#                 except IOError:
#                     print("Warning: 没有找到文件 <%s> 或读取文件失败"%f)
#                 else:
#                     images.append(img)
#         else:
#             images = X
#         size = images[0].size
#         assert np.all(im.size==size for im in X)
#         return size


def get_images(files=None, folder=None, op=None, exts=('.jpg','.jpeg','.png')):
    """[summary]
    
    [description]
    
    Keyword Arguments:
        files {List[Path]} -- jpg or jpeg files (default: {None})
        folder {Path} -- the folder of images (default: {None})
        op {Function} -- operating each image (default: {None})
        exts {tuple[str]} -- (default: {('.jpg','.jpeg','.png')})
    
    Returns:
        List[Image] -- list of images
    
    Raises:
        Exception -- Provide files or a folder
        LookupError -- A file name is invalid
    """
    images = []
    if files:
        if folder:
            files += [f for f in pathlib.Path(folder).iterdir()]
    elif folder:
        files = pathlib.Path(folder).iterdir()
    else:
        raise Exception('Must provide files or a folder')

    for f in files:
        if isinstance(f, str):
            f = pathlib.Path(f)
        if f.suffix == '':
            for ext in exts:
                f = pathlib.Path(f).with_suffix(ext)
                if f.exists():
                    images.append(Image.open(f))
                    break
        elif f.exists() and f.suffix in exts:
            im = Image.open(f)
            images.append(im)
        else:
            raise LookupError('Invalid file name %s' % f)

    if op:
        images = [op(image) for image in images]

    return images


def lrmerge(im1, im2, loc=None):
    '''Merge left part of `im1` and right part of `im2`

    Example
    ------
    im1 = Image.open(imagepath / 'go1.jpg')
    im2 = Image.open(imagepath / 'go2.jpg')
    im = lrmerge(im1, im2)
    '''
    xsize1, ysize1 = im1.size
    xsize2, ysize2 = im2.size
    if loc is None:
        loc = xsize1 // 2
    elif loc <1:
        loc = int(xsize1 * loc)
    box1 = (0, 0, loc, ysize1)
    im1 = im1.crop(box1)
    im2.paste(im1, box1)
    return im2


def tbmerge(im1, im2, loc=None):
    '''Merge top part of `im1` and bottum part of `im2`

    See also lrmerge
    '''
    xsize1, ysize1 = im1.size
    xsize2, ysize2 = im2.size
    if loc is None:
        loc = ysize1 // 2
    elif loc <1:
        loc = int(ysize1 * loc)
    box1 = (0, 0, xsize1, loc)
    im1 = im1.crop(box1)
    im2.paste(im1, box1)
    return im2


def resize_as(im1, im2):
    im1.resize(im2.size)


def cover(im1, im2, w=10):
    '''w: width of the gap
       im1.size == im2.size
    '''
    xsize1, ysize1 = im1.size
    xsize2, ysize2 = im2.size
    im = Image.new(im2.mode, (xsize2, ysize1+w))
    box2 = (0, ysize1+w-ysize2, xsize2, ysize1+w)
    im.paste(im2, box2)
    box1 = (0, 0, xsize1, ysize1) 
    im.paste(im1, box1)
    return im


def stackup(images, size=None, w=200):
    # stack up images
    if size is None:
        size = images[0].size
    im0 = images[0]
    for im in images[1:]:
        im = im.resize(size)
        im0 = cover(im0, im, w=w)
    return im0


def scale(image, k, l=None):
    if l is None:
        l = k
    s = (image.size[0] * k, image.size[1] * l)
    return image.resize(s)

def scale_w(image, width):
    # scale an image according to a fixed width
    w, h = image.size
    h = int(h * width / w)
    return image.resize((width, h))

def scale_h(image, height):
    # scale an image according to a fixed height
    w, h = image.size
    w = int(w * height / h)
    return image.resize((w, height))


def hstack(images, height=None):
    '''Stack images horizontally
    
    Arguments:
        images {[Image]} -- list of images
    
    Keyword Arguments:
        height {Int} -- the common height of images (default: {None})
    
    Returns:
        Image -- the result of image stacking
    '''
    if height is None:
        height = images[0].size[1]

    images = [scale_h(im, height) for im in images]
    stack = Image.new(images[0].mode, (sum(im.size[0] for im in images), height))
    stack.paste(images[0])
    shift = images[0].size[0]
    for im in images[1:]:
        stack.paste(im, (shift, 0))
        shift += im.size[0]
    return stack


def vstack(images, width=None):
    # See also hstack
    if width is None:
        width = images[0].size[0]

    images = [scale_w(im, width) for im in images]
    stack = Image.new(images[0].mode, (width, sum(im.size[1] for im in images)))
    stack.paste(images[0])
    shift = images[0].size[1]
    for im in images[1:]:
        stack.paste(im, (0, shift))
        shift += im.size[1]
    return stack


def tile(layout, vh=True):
    # vh: vstack then hstack
    if vh:
        imageList = [vstack(images) for images in layout]
        return hstack(imageList)
    else:
        imageList = [hstack(images) for images in layout]
        return vstack(imageList)

def sqstack(images, n=None, *args, **kwargs):
    N = len(images)
    if N ==1:
        return images[0]
    if n is None:
        n = int(np.ceil(np.sqrt(N)))
    layout = []
    k = 0
    while True:
        if k+n<N:
            layout.append(images[k:k+n])
        elif k+n >= N:
            layout.append(images[k:])
            break
        k += n
    return tile(layout)


def palace(images):
    assert len(images) == 9, 'exactly 9 images'
    return tile([imags[:3], images[3:6], images[6:9]])


def center_paste(image, other):
    # put other onto the center of the image 
    width1, height1 = image.size
    width2, height2 = other.size
    image.paste(other, ((width1-width2)//2, (height1-height2)//2))
    return image


def fill_image(image):  
    width, height = image.size      
    #选取长和宽中较大值作为新图片的; 生成新图片 
    a = max(width, height)
    new_image = Image.new(image.mode, (a, a), color='white') 
    #将之前的图粘贴在新图上，居中   
    if width > height:
        new_image.paste(image, (0, (a - height) // 2))
    else:  
        new_image.paste(image, ((a - width) // 2, 0))      
    return new_image

def cut_image(image):
    # make 9-palace
    width, height = image.size
    item_width = width // 3
    box_list = [(j*item_width, i*item_width, (j+1)*item_width, (i+1)*item_width) for i in range(3) for j in range(3)]
    image_list = [image.crop(box) for box in box_list]
    return image_list

def replace(image, small, box=None):
    if box is None:
        box = (0,0, *small.size)
    elif len(box)==2:
        box += small.size
    image.paste(small, box)

def replaceOp(image, op, box):
    """Operate only part of the image
    
    Arguments:
        image {Image} -- the image
        op {function} -- operation on images
        box {tuple} -- an square area of the image
    """

    small = op(image.crop(box))
    replace(image, small, box)

 
def save_images(image_list, name=''):  
 
    for index, image in enumerate(image_list, 1):  
        image.save('%s%d.png' % (name, index), 'PNG')


class Background:
    '''The background where you paint.
    
    You can paint with pictures instead of pixels.
    '''

    def __init__(self, nrow, ncol, size=(50, 50), mode='RGB', *args, **kwargs):
        '''
        Arguments:
            nrow {int} -- [the number of rows]
            ncol {int} -- [the number of columns]
        
        Keyword Arguments:
            size {tuple|int} -- [the size of pictures] (default: {(50, 50)})
            mode {str} -- [mode of image] (default: {'RGB'})
        '''
        if isinstance(size, int):
            size = (size, size)
        self.image = Image.new(mode, (ncol*size[0], nrow*size[1]))
        self.nrow = nrow
        self.ncol = ncol
        self.size = size
        self.mode = mode

    def paste(self, img, coord=(0, 0), scale=None):
        '''Embed a image into the background
        
        Embed an image `img` into the background at coordinate `coord`, 
        Like ploting a point in an convas.
        
        Arguments:
            img {Image} -- [a small picture]
        
        Keyword Arguments:
            coord {tuple} -- [coordinate] (default: {(0, 0)})
            scale {int} -- scaling the small image
        '''
        x, y = coord
        if scale:
            size = self.size[0] * scale, self.size[1] * scale
            img = img.resize(size)
        else:
            img = img.resize(self.size)
        self.image.paste(img, (x * self.size[0], y * self.size[1]))

    def save(self, *args, **kwargs):
        self.image.save(*args, **kwargs)

    def draw(self, imgs, coords):
        '''Embed several images into the background.

        imgs[k] will be loacated at coords[k]

        Arguments:
            imgs {[Image]} -- list of images
            coords {[tuple]} -- list of coordinates
        '''

        for img, coord in zip(itertools.cycle(imgs), coords):
            self.paste(img, coord)

def patch(imgs, big):
    """
    Patch a big image with small images

    Arguments:
        imgs {List[Image]} -- list of images
        big {Image} -- big image
    
    Returns:
        an image
    """
    bg = Background(big.size[1], big.size[0], size=imgs[0].size)
    a = np.asarray(big, dtype=np.float64)
    means = [np.asarray(img, dtype=np.float64).mean(axis=(0,1)) for img in imgs if img.mode=='RGB']
    for i in range(big.size[1]):
        for j in range(big.size[0]):
            p = a[i, j, :]
            k = np.argmin([LA.norm(p-m) for m in means])
            bg.paste(imgs[k], (j, i))
    return bg.image


def rgbimage(a, b, c):
    assert a.shape == b.shape == c.shape
    d = np.zeros(a.shape + (3,))
    d[:,:,0]=a
    d[:,:,1]=b
    d[:,:,2]=c
    return Image.fromarray(d.astype('uint8')).convert('RGB')

class imageOp:
    """Decorator for operation on images
    
    Examples:
        @imageOp(mode='RGB')
        def f ...

        @imageOp
        def f ...
    """

    def __new__(cls, f=None, mode=None):
        obj = super(imageOp, cls).__new__(cls)
        if mode is None and f is not None:
            obj.mode = None
            return obj(f)
        else:
            obj.mode = mode
            return obj

    def __call__(self, f):
        def ff(image, *args, **kwargs):
            array = np.asarray(image, dtype=np.float64)    
            if self.mode is None:
                mode = image.mode
            else:
                mode = self.mode
            return Image.fromarray(f(array, *args, **kwargs).astype('uint8')).convert(self.mode)
        return ff


def cross(image1, image2, func):
    # as imageOP but it is a binary function
    a1 = np.asarray(image1, dtype=np.float64)
    a2 = np.asarray(image2, dtype=np.float64)
    return Image.fromarray(func(a1, a2).astype('uint8')).convert(image1.mode)


def filt(array, channel='a'):
    import pywt  
    if array.ndim == 3:
        abc=[]
        for k in range(3):
            wp = pywt.WaveletPacket2D(data=array[:,:,k], wavelet='db1', mode='symmetric')
            abc.append(wp[channel].data)
        return np.stack(abc, axis=2)
    else:
        wp = pywt.WaveletPacket2D(data=array, wavelet='db1', mode='symmetric')
        return wp[channel].data

# folder=pathlib.Path('/Users/william/Programming/Python/toy/faces/eigen/')
# images=get_images(folder=folder)

# image=sqstack(images[:6])
# image.save(folder / '1.jpg')
# image=sqstack(images[6:])
# image.save(folder / '2.jpg')
# 
