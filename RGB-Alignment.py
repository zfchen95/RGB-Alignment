import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.linalg import norm
from skimage import color
from scipy.misc import imresize
from time import time


def separate(image):
    height = int(np.floor(image.shape[0] / 3))
    blue = image[1: height, :]
    green = image[height+1: height*2, :]
    red = image[height*2+1: height*3, :]

    print(blue.shape, green.shape, red.shape)
    # plt.figure()
    # plt.imshow(image)
    return blue, green, red


def crop_original(image):
    if image.shape[1] > 3000:
        return image[int(image.shape[0]*0.02): int(image.shape[0]*0.98), int(image.shape[1]*0.05): int(image.shape[1]*0.95)]
    img_grey = image
    upper_bound = 0
    left_bound = 0
    lower_bound = img_grey.shape[0]
    right_bound = img_grey.shape[1]
    print(upper_bound, lower_bound, left_bound, right_bound)
    height = img_grey.shape[0]
    width = img_grey.shape[1]
    partition = 10
    # high_p = 80
    # low_p = 20
    # threshold = 5
    # for i in range(1, int(height / partition)):
    #     if np.percentile(img_grey[i, :], high_p) - np.percentile(img_grey[i, :], low_p) < threshold:
    #         upper_bound = i
    #     if np.percentile(img_grey[height - i, :], high_p) - np.percentile(img_grey[height - i, :], low_p) < threshold:
    #         lower_bound = height - i
    # for i in range(1, int(width / partition)):
    #     if np.percentile(img_grey[:, i], high_p) - np.percentile(img_grey[:, i], low_p) < threshold:
    #         left_bound = i
    #     if np.percentile(img_grey[:, width - i], high_p) - np.percentile(img_grey[:, width - i], low_p) < threshold:
    #         right_bound = width - i
    threshold_h = 250
    threshold_l = 30
    p = 50
    for i in range(1, int(height / partition)):
        if np.percentile(img_grey[i, :], p) < threshold_l or np.percentile(img_grey[i, :], p) > threshold_h:
                upper_bound = i
        if np.percentile(img_grey[height - i, :], p) < threshold_l or np.percentile(img_grey[height - i, :], p) > threshold_h:
            lower_bound = height - i
    for i in range(1, int(width / partition)):
        if np.percentile(img_grey[:, i], p) < threshold_l or np.percentile(img_grey[:, i], p) > threshold_h:
            left_bound = i
        if np.percentile(img_grey[:, width - i], p) < threshold_l or np.percentile(img_grey[:, width - i], p) > threshold_h:
            right_bound = width - i
    print(upper_bound, lower_bound, left_bound, right_bound)
    # plt.figure()
    # plt.imshow(image[upper_bound: lower_bound, left_bound: right_bound])
    return image[upper_bound: lower_bound, left_bound: right_bound]


def crop(image):
    img_grey = color.rgb2gray(image)
    # img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_grey)
    upper_bound = 0
    left_bound = 0
    lower_bound = img_grey.shape[0]
    right_bound = img_grey.shape[1]
    print(upper_bound, lower_bound, left_bound, right_bound)
    height = img_grey.shape[0]
    width = img_grey.shape[1]
    partition = 10
    high_p = 80
    low_p = 20
    threshold = 0.01
    for i in range(1, int(height / partition)):
        if np.percentile(img_grey[i, :], high_p) - np.percentile(img_grey[i, :], low_p) < threshold:
            upper_bound = i
        if np.percentile(img_grey[height-i, :], high_p) - np.percentile(img_grey[height-i, :], low_p) < 0.05:
            lower_bound = height-i
    for i in range(1, int(width / partition)):
        if np.percentile(img_grey[:, i], high_p) - np.percentile(img_grey[:, i], low_p) < threshold:
            left_bound = i
        if np.percentile(img_grey[:, width-i], high_p) - np.percentile(img_grey[:, width-i], low_p) < 0.05:
            right_bound = width-i
    print(upper_bound, lower_bound, left_bound, right_bound)
    return image[upper_bound: lower_bound, left_bound: right_bound]


def find_shift(img, ref, window_size=10, partition=5):
    x_shift = 0
    y_shift = 0
    # NCC
    if True:
        best = 0
        height = int(img.shape[0] / partition)
        width = int(img.shape[1] / partition)
        ref_dif = ref[height: int((partition-1)*height), width: int((partition-1)*width)]
        ref_dif = ref_dif - np.mean(ref_dif)
        img_dif = img[height: int((partition-1)*height), width: int((partition-1)*width)]
        img_dif = img_dif - np.mean(img_dif)

        print(ref_dif.shape, img_dif.shape)

        for i in range(-window_size, window_size):
            tmp = np.roll(img_dif, i, 0)
            for j in range(-window_size, window_size):
                tmp2 = np.roll(tmp, j, 1)
                score = np.sum(tmp2 * ref_dif) / (norm(tmp2) * norm(ref_dif))
                if score > best:
                    x_shift = i
                    y_shift = j
                    best = score
    # SSD
    else:
        best = float('inf')
        for i in range(-window_size, window_size):
            tmp = np.roll(img, i, 0)
            for j in range(-window_size, window_size):
                score = np.sum((np.roll(tmp, j, 1) - ref) ** 2)
                if score < best:
                    x_shift = i
                    y_shift = j
                    best = score
    return [x_shift, y_shift]


def find_shift_multiscale(image, ref):
    shift_tot = [0, 0]
    prev = 128
    while image.shape[0] * 2 > prev:
        new_img = image
        new_ref = ref
        scale = 2
        while new_img.shape[0] > prev:
            new_img = imresize(new_img, 50)
            new_ref = imresize(new_ref, 50)
            scale = scale * 2
        shift = find_shift(new_img, new_ref, window_size=5, partition=3)
        print(prev, shift, scale)
        image = np.roll(image, shift[0] * scale, 0)
        image = np.roll(image, shift[1] * scale, 1)
        prev = prev * 2
        shift_tot[0] = shift_tot[0] + shift[0] * scale
        shift_tot[1] = shift_tot[1] + shift[1] * scale
    print(shift_tot)
    return shift_tot


def align_image(image):
    blue, green, red = separate(image)
    if green.shape[0] > 512:
        red_shift = find_shift_multiscale(red, green)
        blue_shift = find_shift_multiscale(blue, green)
    else:
        red_shift = find_shift(red, green)
        blue_shift = find_shift(blue, green)
    print(red_shift, blue_shift)
    red = np.roll(red, red_shift[0], 0)
    red = np.roll(red, red_shift[1], 1)
    blue = np.roll(blue, blue_shift[0], 0)
    blue = np.roll(blue, blue_shift[1], 1)
    return cv2.merge([red, green, blue])


def main():
    img_list = ['00125v.jpg', '00149v.jpg', '00153v.jpg', '00351v.jpg', '00398v.jpg', '01112v.jpg']
    # img_list = ['00153v.jpg']
    # img_list = ['01047u.tif']
    # img_list = ['01657u.tif']
    # img_list = ['01861a.tif']
    for img_name in img_list:
        img = plt.imread(img_name)
        print(img.shape)
        img = crop_original(img)
        print(img.shape)
        start = time()
        new_img = align_image(img)
        end = time()
        print('running time', end - start)
        print(new_img.shape)
        # new_img = crop(new_img)
        fig = plt.figure()
        plt.imshow(imresize(new_img, 100))
        # fig.savefig(img_name.replace('u.tif', '.jpg'), format='jpg', dpi=1200)
        # fig.savefig(img_name.replace('a.tif', '.jpg'), format='jpg', dpi=1200)
        # fig.savefig(img_name.replace('v.jpg', '.jpg'))
    plt.show()


main()