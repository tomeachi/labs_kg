import math
import numpy as np
from PIL import Image, ImageOps

from main import draw_correct

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

f = open('model_1.obj')
"""
for s in f:
    if (s[0] == 'v' and s[1] == ' '): print (s)
"""
list = []
listf = []
for s in f:
    splitted = s.split()
    if(splitted[0] == 'v'): list.append([float(x) for x in splitted[1:]])
    if (splitted[0] == 'f'): listf.append([int(x.split('/')[0]) for x in splitted[1:]])
#for v in list:
    #img_mat[int(10000*v[1])+1000, int(10000*v[0])+1000] = 255

for face in listf:
    x0 = int(10000*list[face[0]-1][0])+1000
    y0 = int(10000*list[face[0]-1][1])+1000
    x1 = int(10000*list[face[1]-1][0])+1000
    y1 = int(10000*list[face[1]-1][1])+1000
    x2 = int(10000*list[face[2]-1][0])+1000
    y2 = int(10000*list[face[2]-1][1])+1000
    draw_correct(img_mat, x0, y0, x1, y1, (0, 255, 255))
    draw_correct(img_mat, x1, y1, x2, y2, (0, 255, 255))
    draw_correct(img_mat, x0, y0, x2, y2, (0, 255, 255))

img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img2.png')