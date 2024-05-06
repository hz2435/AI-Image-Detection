import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt 
from scipy.signal import wiener

def denoise(image):
    """ 
    well this entire function is straight up copied but it's meant to be a Wavelet Transform-based Wiener filter
    """

    image = image.astype(np.float32)
    
    wavelet = 'db8'
    max_level = pywt.dwt_max_level(data_len=min(image.shape), filter_len=pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec2(image, wavelet, level=max_level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(image.size))

    denoised_coeffs = [coeffs[0]] 
    for detail_level in coeffs[1:]:
        denoised_level = []
        for band in detail_level:
            band_thresholded = pywt.threshold(band, threshold, mode='soft')
            denoised_level.append(band_thresholded)
        denoised_coeffs.append(tuple(denoised_level))

    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)

    denoised_image = np.clip(denoised_image, 0, 255)

    return denoised_image.astype(np.uint8)

    # return wiener(image, (50, 50))
    # return cv2.GaussianBlur(image, (3, 3), 0)


def prnu(image_path):

    """
    first convert to gray scale
    IM_out = (I_ones + Noise_cam).IM_in + Noise_add // noise add prob optional 
    . = pixel wise product

    W = IM_out - denoise(IM_out)

    Output = W / IM_in
    """
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    denoised_image = denoise(original_image)
    if denoised_image.shape != original_image.shape: #reshaping stuff
        denoised_image = cv2.resize(denoised_image, (original_image.shape[1], original_image.shape[0]))

    prnu_residual = original_image.astype(np.float32) - denoised_image.astype(np.float32)
    prnu_clipped = np.clip(prnu_residual, 0, 255).astype(np.uint8)

    # prnu_normalized = (prnu_residual*180).astype(np.uint8)
    # prnu_enhanced = cv2.equalizeHist((prnu_normalized * 255).astype(np.uint8))

    epsilon = 1e-10  # prevent divide by zero
    prnu = (prnu_clipped + epsilon) / (original_image + epsilon) #lmao


    # f_final = cv2.equalizeHist((prnu * 255).astype(np.uint8)) #equilizehist is just high pass contrast enhancer for visualization
    f_final =(prnu * 255).astype(np.uint8)


    # plt.imshow(f_final, cmap='gray') 
    # plt.axis('off') 
    # plt.show()
    
    return np.array(f_final)

# prnu("dog.jpg")
# prnu("Gemini_Generated_Image_nhr5xknhr5xknhr5.jpeg")


# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# denoised_image = cv2.GaussianBlur(original_image, (3, 3), 0)

# prnu_residual = original_image - denoised_image

# prnu = prnu_residual / original_image

# return np.array(prnu