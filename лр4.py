import math
import numpy as np
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)
z_buf = np.full((1000, 1000), np.inf)
tjpg = Image.open('bunny-atlas.jpg')
tjpg = ImageOps.flip(tjpg)
tnp = np.asarray(tjpg)

tweight = tnp.shape[1]
theight = tnp.shape[0]

f = open('bunny.obj')
list = [] #заяц
listf = [] #номера
listv = [] #текстура
listc = [] #номера текстуры
for s in f:
    splitted = s.split()
    if(splitted[0] == 'v'): list.append([float(x) for x in splitted[1:]])
    if(splitted[0] == 'f'): listf.append([int(x.split('/')[0]) for x in splitted[1:]])
    if(splitted[0]=='f'): listc.append([int(x.split('/')[1]) for x in splitted[1:]])
    if(splitted[0]=='vt'): listv.append([float(x) for x in splitted[1:]])

z_buf = [[math.inf for i in range(img_mat.shape[1])] for j in range(img_mat.shape[0])]

q=1
w= 10000*q

def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    d = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / d
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / d
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2
def draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, I, H, W):
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
            l0, l1, l2 = barycentric_coordinates(x,y, u0, v0, u1, v1, u2, v2)
            if(l0>=0 and l1>=0 and l2>=0):
                z = l0*z0 + l1*z1 + l2*z2
                In = (l0*I[0] + l1*I[1] + l2*I[2])
                w_t = int(tweight * (l0*W[0] + l1*W[1] + l2*W[2]))
                h_t = int(theight * (l0*H[0] + l1*H[1] + l2*H[2]))
                if z < z_buf[y][x] and In <= 0:
                    img_mat[y, x] = tnp[w_t, h_t] * -In
                    z_buf[y][x] = z

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

vn = [[0,0,0] for i in range(len(list))]

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
    n = math.sqrt(xn**2 + yn**2 + zn**2)

    for i in range(3):
        vn[face[i]-1][0] += xn/n
        vn[face[i]-1][1] += yn/n
        vn[face[i]-1][2] += zn/n

for vec in vn:
    n = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    for i in range(3):
        vec[i] /= n

for (face, texture) in zip(listf, listc):
    x0 = list[face[0] - 1][0]
    y0 = list[face[0] - 1][1]
    z0 = list[face[0] - 1][2]
    x1 = list[face[1] - 1][0]
    y1 = list[face[1] - 1][1]
    z1 = list[face[1] - 1][2]
    x2 = list[face[2] - 1][0]
    y2 = list[face[2] - 1][1]
    z2 = list[face[2] - 1][2]

    w0 = listv[texture[0] - 1][0]
    h0 = listv[texture[0] - 1][1]
    w1 = listv[texture[1] - 1][0]
    h1 = listv[texture[1] - 1][1]
    w2 = listv[texture[2] - 1][0]
    h2 = listv[texture[2] - 1][1]

    n0 = vn[face[0] - 1]
    n1 = vn[face[1] - 1]
    n2 = vn[face[2] - 1]

    I0 = cos_l(n0[0], n0[1], n0[2])
    I1 = cos_l(n1[0], n1[1], n1[2])
    I2 = cos_l(n2[0], n2[1], n2[2])

    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    cosl = cos_l(xn, yn, zn)

    if cosl < 0:
        draw_triangle(img_mat, x0, y0, z0, x1, y1, z1, x2, y2, z2, (I0, I1, I2), (w0,w1,w2), (h0,h1,h2))

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.save("ttt.png")