import math
import random
import numpy as np
from PIL import Image, ImageOps

from main import draw_correct

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)
z_buf = np.full((1000, 1000), np.inf)

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

z_buf = [[math.inf for i in range(img_mat.shape[0])] for j in range(img_mat.shape[1])]

q=0.1
w= 10000*q
def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    d = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / d
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / d
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2
def draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    u0 = w*x0/z0 + img_mat.shape[1]/2
    v0 = w*y0/z0 + img_mat.shape[0]/2
    u1 = w*x1/z1 + img_mat.shape[1]/2
    v1 = w*y1/z1 + img_mat.shape[0]/2
    u2 = w*x2/z2 + img_mat.shape[1]/2
    v2 = w*y2/z2 + img_mat.shape[0]/2
    xmin = math.floor(min(u0, u1, u2))
    xmax = math.ceil(max(u0, u1, u2))
    ymin = math.floor(min(v0, v1, v2))
    ymax = math.ceil(max(v0, v1, v2))
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > img_mat.shape[1]: xmax = img_mat.shape[1]
    if ymax > img_mat.shape[0]: ymax = img_mat.shape[0]

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = barycentric_coordinates(x, y, u0, v0, u1, v1, u2, v2)
            if (l0>=0 and l1>=0 and l2>=0):
                z_ish = z0*l0 + z1*l1 + z2*l2
                if z_ish < z_buf[y][x]:
                    img_mat[y][x] = color
                    z_buf[y][x] = z_ish

def norm(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    x = (y1 - y2) * (z1 - z0) - (y1 - y0) * (z1- z2)
    y = -((x1 - x2) * (z1 - z0) - (x1 - x0) * (z1- z2))
    z = (x1 - x2) * (y1 - y0) - (x1 - x0) * (y1- y2)
    return x, y, z

def cos_l(x, y, z):
    return z / (math.sqrt(x**2 + y**2 + z**2))
def rotate_and_trans(x, y, z, a, b, g):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(a), np.sin(a)],
                   [0, -np.sin(a), np.cos(a)]])

    Ry = np.array([[np.cos(b), 0, -np.sin(b)],
                   [0, 1, 0],
                   [np.sin(b), 0, np.cos(b)]])

    Rz = np.array([[np.cos(g), np.sin(g), 0],
                   [-np.sin(g), np.cos(g), 0],
                   [0, 0, 1]])
    mat = np.array([[x],
                   [y],
                   [z]])
    R = Rx@Ry@Rz@[x, y, z] + [0, -0.03, q]
    return R[0], R[1], R[2]

z_buf = [[math.inf for i in range(img_mat.shape[0])] for j in range(img_mat.shape[1])]
for i in list:
    i[0], i[1], i[2] = rotate_and_trans(i[0], i[1], i[2], 0, np.pi/2, 0)

for face in listf:
    x0 = list[face[0] - 1][0]
    y0 = list[face[0] - 1][1]
    z0 = list[face[0] - 1][2]
    x1 = list[face[1] - 1][0]
    y1 = list[face[1] - 1][1]
    z1 = list[face[1] - 1][2]
    x2 = list[face[2] - 1][0]
    y2 = list[face[2] - 1][1]
    z2 = list[face[2] - 1][2]

    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosl = cos_l(xn, yn, zn)
    if cosl < 0:
        draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, (-123*cosl,-104*cosl, -238*cosl))




img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.save("triangle5.png")