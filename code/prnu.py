import cv2
import numpy as np

def prnu(image_path):

    """
    first convert to gray scale

    IM_out = (I_ones + Noise_cam).IM_in + Noise_add // noise add prob optional 

    . = pixel wise product

    W = IM_out - denoise(IM_out)

    Output = W / IM_in
    """

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    denoised_image = cv2.GaussianBlur(original_image, (3, 3), 0)

    prnu_residual = original_image - denoised_image

    prnu = prnu_residual / original_image

    return np.array(prnu)