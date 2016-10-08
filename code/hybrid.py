# -*- coding: utf-8 -*-
import numpy as np
import cv2

'''
visualize a hybrid image by progressively downsampling the image and
concatenating all of the images together.
'''
def vis_hybrid_image(hybrid_image):

    scales = 5
    scale_factor = 0.5
    padding = 5

    original_height = hybrid_image.shape[0]
    num_colors = hybrid_image.shape[2]

    output = hybrid_image
    cur_image = hybrid_image
    
    
    for i in range(2, scales + 1):
        output = np.concatenate((output, np.ones((original_height, padding, num_colors))), axis=1)
        cur_image = cv2.resize(cur_image, (0,0), fx=scale_factor, fy=scale_factor)
        if cur_image.ndim==2:
            cur_image = cur_image.reshape(cur_image.shape+(1,))
        tmp = np.concatenate((np.ones((original_height - cur_image.shape[0], cur_image.shape[1], num_colors)), cur_image), axis=0)
        output = np.concatenate((output, tmp), axis=1)

    return output


def my_imfilter(image, kernel, padding_type=0):

    iw,il,ic = image.shape

    kw,kl = kernel.shape
    pad_w = (kw-1)/2
    pad_l = (kl-1)/2
    new_row = np.zeros((pad_w,il,ic))
    new_col = np.zeros((iw+2*pad_w,pad_l,ic))

    if padding_type == 0:
        new_row = np.zeros((pad_w,il,ic))
        new_col = np.zeros((iw+2*pad_w,pad_l,ic))
        image = np.vstack((image,new_row))
        image = np.vstack((new_row,image))
        image = np.hstack((new_col,image))
        image = np.hstack((image,new_col))

    elif padding_type == 1:
        new_row1 = np.zeros((pad_w,il,ic))
        new_row2 = np.zeros((pad_w,il,ic))
        for i in range(0,pad_w):
            new_row1[i, :, :]+=image[0, :, :]
            new_row2[i, :, :]+=image[iw-1, :, :]
        image = np.vstack((new_row1, image))
        image = np.vstack((image, new_row2))
        new_col1 = np.zeros((iw+2*pad_w,pad_l,ic))
        new_col2 = np.zeros((iw+2*pad_w,pad_l,ic))
        for i in range(0,pad_l):
            new_col1[:, i, :]+=image[:, 0, :]
            new_col2[:, i, :]+=image[:, il-1,:]
        image = np.hstack((new_col1, image))
        image = np.hstack((image, new_col2))

    elif padding_type == 2:
        new_row1 = np.zeros((pad_w,il,ic))
        new_row2 = np.zeros((pad_w,il,ic))
        for i in range(0,pad_w):
            new_row1[pad_w-1-i, :, :]+=image[i, :, :]
            new_row2[i, :, :]+=image[iw-1-i, :, :]
        image = np.vstack((new_row1, image))
        image = np.vstack((image, new_row2))
        new_col1 = np.zeros((iw+2*pad_w,pad_l,ic))
        new_col2 = np.zeros((iw+2*pad_w,pad_l,ic))
        for i in range(0,pad_l):
            new_col1[:, pad_l-1-i, :]+=image[:, i, :]
            new_col2[:, i, :]+=image[:, il-1-i,:]
        image = np.hstack((new_col1, image))
        image = np.hstack((image, new_col2))

    newImage = np.zeros(image.shape)

    for i in range(pad_w, pad_w+iw):
        for j in range(pad_l, pad_l+il):
            for m in range(-pad_w, pad_w+1):
                for n in range(-pad_l, pad_l+1):
                    newImage[i][j][:] += image[i+m][j+n][:] * kernel[pad_w-m][pad_l-n]

    newImage = newImage[pad_w:(pad_w+iw), pad_l:(pad_l+il), :]

    return newImage


def fft_imfilter(image, kernel):

    result = np.zeros(image.shape,np.complex128)
    for i in range(0,image.shape[2]):
        im = image[:,:,i]
        image_fre = np.fft.fft2(im)
        image_fre = np.fft.fftshift(image_fre)
        f = np.fft.fft2(kernel,im.shape)
        fshift = np.fft.fftshift(f)
        newImage_fre = image_fre * fshift
        newImage = np.fft.ifft2(np.fft.ifftshift(newImage_fre))
        result[:,:,i] = newImage
    return np.abs(result)

#	return

if __name__ == '__main__':
    image1 = cv2.imread('../data/einstein.bmp') / 255.0
    image2 = cv2.imread('../data/marilyn.bmp') / 255.0
    
    
    if image1.ndim == 2:
        image1 = image1.reshape(image1.shape+(1,))
    if image2.ndim == 2:
        image2 = image2.reshape(image2.shape+(1,))

    cutoff_frequency = 5

    kernel = cv2.getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency)
    kernel = cv2.mulTransposed(kernel, False)


    low_frequencies = my_imfilter(image1, kernel)
    high_frequencies = image2 - my_imfilter(image2, kernel)
    hybrid_image = low_frequencies + high_frequencies

    vis = vis_hybrid_image(hybrid_image)

    cv2.imshow("low", low_frequencies)
    cv2.imshow("high", high_frequencies + 0.5)
    cv2.imshow("vis", vis)
    cv2.waitKey(0)

    cv2.imwrite('low_frequencies.jpg', low_frequencies*255)
    cv2.imwrite('high_frequencies.jpg', (high_frequencies + 0.5)*255)
    cv2.imwrite('hybrid_image.jpg', (hybrid_image)*255)
    cv2.imwrite('hybrid_image_scales.jpg', vis*255)
