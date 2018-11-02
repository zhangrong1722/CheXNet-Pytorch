import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt

# create by Feng, edit by Feng 2017 / 08 / 14
# this project is doing image data augmentation for any DL/ML algorithm
#

# avg blur minimum filter size is 3

def avg_blur(img, max_filiter_size = 3) :
	img = img.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		out = cv2.blur(img, (filter_size, filter_size))
	return out

# gaussain blur minimum filter size is 3
# when sigma = 0 gaussain blur weight will compute by program
# when the sigma is more large the blur effect more obvious

def gaussain_blur(img, max_filiter_size = 3, sigma = 0) :
	img = img.astype(np.uint8)
	if max_filiter_size >= 3 :
		filter_size = random.randint(3, max_filiter_size)
		if filter_size % 2 == 0 :
			filter_size += 1
		#print ('size = %d'% filter_size)
		out = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
	return out

def gaussain_noise(img, mean = 0, var = 0.1) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	sigma = var ** 0.5
	gauss = np.random.normal(mean, sigma, (h, w, c))
	gauss = gauss.reshape(h, w, c).astype(np.uint8)
	noisy = img + gauss
	return noisy

# fill_pixel is 0(black) or 255(white)

def img_shift(img, x_min_shift_piexl = -1, x_max_shift_piexl = 1, y_min_shift_piexl = -1, y_max_shift_piexl = 1, fill_pixel = 0) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	out = np.zeros(img.shape)
	if fill_pixel == 255 :
		out[:, :] = 255
	out = out.astype(np.uint8)
	move_x = random.randint(x_min_shift_piexl, x_max_shift_piexl)
	move_y = random.randint(y_min_shift_piexl, y_max_shift_piexl)
	#print (('move_x = %d')% (move_x))
	#print (('move_y = %d')% (move_y))
	if move_x >= 0 and move_y >= 0 :
		out[move_y:, move_x: ] = img[0: (h - move_y), 0: (w - move_x)]
	elif move_x < 0 and move_y < 0 :
		out[0: (h + move_y), 0: (w + move_x)] = img[ - move_y:, - move_x:]
	elif move_x >= 0 and move_y < 0 :
		out[0: (h + move_y), move_x:] = img[ - move_y:, 0: (w - move_x)]
	elif move_x < 0 and move_y >= 0 :
		out[move_y:, 0: (w + move_x)] = img[0 : (h - move_y), - move_x:]
	return out

# In img_rotation func. rotation center is image center

def img_rotation(img, min_angel = 0, max_angel = 0, min_scale = 1, max_scale = 1, fill_pixel = 0) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	_angel = random.randint(min_angel, max_angel)
	_scale = random.uniform(min_scale, max_scale)
	rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), _angel, _scale)
	out = cv2.warpAffine(img, rotation_matrix, (w, h))
	if fill_pixel == 255 :
		mask = np.zeros(img.shape)
		mask[:, :, :] = 255
		mask = mask.astype(np.uint8)
		mask = cv2.warpAffine(mask, rotation_matrix, (w, h))
		for i in range (h) :
			for j in range(w) :
				if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 0 :
					out[i, j, :] = 255
	return out

# In img_flip func. it will random filp image
# when flip factor is 1 it will do hor. flip (Horizontal)
#					  0            ver. flip (Vertical)
#					 -1			   hor. + ver flip

def img_flip(img) :
	img = img.astype(np.uint8)
	flip_factor = random.randint(-1, 1)
	out = cv2.flip(img, flip_factor)
	return out

# Zoom image by scale

def img_zoom(img, min_scale = 1, max_scale = 1) :
	img = img.astype(np.uint8)
	h, w, c = img.shape
	scale = random.uniform(min_scale, max_scale)
	h = int(h * scale)
	w = int(w * scale)
	out = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
	return out

# change image contrast by hsv

def img_contrast(img, min_s, max_s, min_v, max_v) :
	img = img.astype(np.uint8)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	_s = random.randint(min_s, max_s)
	_v = random.randint(min_v, max_v)
	if _s >= 0 :
		hsv_img[:, :, 1] += _s
	else :
		_s = - _s
		hsv_img[:, :, 1] -= _s
	if _v >= 0 :
		hsv_img[:, :, 2] += _v
	else :
		_v = - _v
		hsv_img[:, :, 2] += _v
	out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	return out

# change image color by hsv

def img_color(img, min_h, max_h) :
	img = img.astype(np.uint8)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	_h = random.randint(min_h, max_h)
	hsv_img[:, :, 0] += _h
	out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	return out




























