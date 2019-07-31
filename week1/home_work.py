#!/usr/bin/env python
# coding: utf-8
"""
crop, color shift, rotation and perspective transform
"""

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def random_crop(img):
    width = img.shape[0]
    height = img.shape[1]
    x1 = random.randint(0, width)
    x2 = random.randint(x1, width)
    y1 = random.randint(0, height)
    y2 = random.randint(y1, height)
    return img[x1: x2, y1: y2]

def random_color_shift(img):
    B, G, R = cv2.split(img)
    random_b = random.randint(0, 255)
    B = B - random_b
    random_g = random.randint(0, 255)
    G = G - random_g
    random_r = random.randint(0, 255)
    R = R - random_r
    return cv2.merge((R, B, G))

def random_rotation(img):
    random_angle = random.randint(0, 360)
    M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), random_angle, 1)
    return cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))

def random_perspctive_transform(img):
    random_margin = 60
    width = img.shape[0]
    height = img.shape[1]
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M_warp, (width, height))


choise = input()
img = cv2.imread('Lenna.jpg')
img_crop = random_crop(img)
img_color_shift = random_color_shift(img)
img_rotation = random_rotation(img)
img_transform = random_perspctive_transform(img)
cv2.imshow('test', img_transform)
key = cv2.waitKey()
cv2.destroyAllWindows()
