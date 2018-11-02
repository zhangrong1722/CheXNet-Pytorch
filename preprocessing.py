import os
import cv2
from PIL import Image, ImageEnhance

from utils.data_augument import img_contrast, img_shift, img_rotation, gaussain_blur, gaussain_noise, avg_blur


_max_filiter_size = 7  # for avg_blur and gaussain_blur
_sigma = 0  # for gaussain_blur

_mean = 0  # for gaussain_noise
_var = 0.1  # for gaussain_noise

_x_min_shift_piexl = -20  # for img_shift
_x_max_shift_piexl = 20  # for img_shift
_y_min_shift_piexl = -20  # for img_shift
_y_max_shift_piexl = 20  # for img_shift
_fill_pixel = 0  # for img_shift and img_rotation: black

_min_angel = -20  # for img_rotation
_max_angel = 20  # for img_rotation
_min_scale = 0.9  # for img_rotation
_max_scale = 1.1  # for img_rotation

_min_s = -10  # for img_contrast
_max_s = 10  # for img_contrast
_min_v = -10  # for img_contrast
_max_v = 10  # for img_contrast

_min_h = -30  # for img_color
_max_h = 30  # for img_color

_generate_quantity = 10

data_dir = 'data/LESION_DATA'
img_lst = os.listdir(data_dir)
for name in img_lst:
    abs_path = os.path.join(data_dir, name)
    img = cv2.imread(abs_path)
    prefix, suffix = abs_path.split('.')
    cv2.imwrite('%s_%s.%s' % (prefix, 'blur1', suffix), gaussain_blur(img, _max_filiter_size, _sigma))
    cv2.imwrite('%s_%s.%s' % (prefix, 'blur2', suffix), gaussain_blur(img, _max_filiter_size, _sigma))
    cv2.imwrite('%s_%s.%s' % (prefix, 'noise1', suffix), gaussain_noise(img, _mean, _var))
    cv2.imwrite('%s_%s.%s' % (prefix, 'noise2', suffix), gaussain_noise(img, _mean, _var))
    cv2.imwrite('%s_%s.%s' % (prefix, 'shift1', suffix),
                img_shift(img, _x_min_shift_piexl, _x_max_shift_piexl, _y_min_shift_piexl, _y_max_shift_piexl,
                          _fill_pixel))
    cv2.imwrite('%s_%s.%s' % (prefix, 'shift2', suffix),
                img_shift(img, _x_min_shift_piexl, _x_max_shift_piexl, _y_min_shift_piexl, _y_max_shift_piexl,
                          _fill_pixel))
    cv2.imwrite('%s_%s.%s' % (prefix, 'rotation1', suffix),
                img_rotation(img, _min_angel, _max_angel, _min_scale, _max_scale, _fill_pixel))
    cv2.imwrite('%s_%s.%s' % (prefix, 'rotation2', suffix),
                img_rotation(img, _min_angel, _max_angel, _min_scale, _max_scale, _fill_pixel))

    img02 = Image.open(abs_path)
    ImageEnhance.Brightness(img02).enhance(0.5).save('%s_%s.%s' % (prefix, 'brightness1', suffix))
    ImageEnhance.Brightness(img02).enhance(1.5).save('%s_%s.%s' % (prefix, 'brightness2', suffix))
    ImageEnhance.Contrast(img02).enhance(0.6).save('%s_%s.%s' % (prefix, 'contrast1', suffix))
    ImageEnhance.Contrast(img02).enhance(1.5).save('%s_%s.%s' % (prefix, 'contrast2', suffix))
