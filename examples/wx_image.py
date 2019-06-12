#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import itchat
import pathlib
import PIL.Image as Image
import math
import image


itchat.auto_login(hotReload=True)
friends = itchat.get_friends(update=True)[0:]   # 核心：得到friends列表集，内含很多信息

# print("授权微信用户为："+ user)
# user = '@7a854d8fbfbb165660f6fb2b92765866bf0715e45dc5f683c11378a267bf7962'

user = pathlib.Path(user)
if not user.exists():
    user.mkdir()  # 创建文件夹用于装载所有好友头像
num = 0
for i in friends:
    img = itchat.get_head_img(userName=i["UserName"])
    with open(user / ("%d.jpg" % num), 'wb') as fileImage:
        fileImage.write(img)
    num += 1

print("所有好友头像数：%d" % num)
eachsize = int(math.sqrt(640 * 640 / num))    # 先圈定每个正方形小头像的边长，如果嫌小可以加大
print("小正方形头像边长：%d" % eachsize)
numrow = int(640 / eachsize)
print("一行小头像数：%d" % numrow)
numcol = int(math.ceil(num / numrow))   # 向上取整
toImage = image.Background(numrow, numcol, eachsize)  # 先生成头像集模板
x = 0   # 小头像拼接时的左上角横坐标
y = 0   # 小头像拼接时的左上角纵坐标
for i in range(num):
    try:
        #打开图片
        img = Image.open(user / ("%d.jpg"%i))
    except IOError:
        print("Error: 没有找到文件或读取文件失败")
    else:
        img = img.resize((eachsize, eachsize), Image.ANTIALIAS)
        toImage.paste(img, (x, y))
        x += 1
        if x == numrow:
            x = 0
            y += 1
toImage.save(user / "x.jpg")
# itchat.send_image(str(user / "x.jpg"), 'filehelper')