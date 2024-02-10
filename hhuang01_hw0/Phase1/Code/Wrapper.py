#!/usr/bin/env python

"""
CMSC733 Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Marco Huang (hhuang01@umd.edu)
Master Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from scipy.ndimage import rotate
from scipy.signal import convolve2d
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace
from sklearn.cluster import KMeans

def plot_images(images, x_len, y_len, image_name, fig_size = -1):
    if fig_size != -1:
        plt.figure(figsize = fig_size)
    total_plots = x_len * y_len
    for i, image in enumerate(images[:total_plots]):
        plt.subplot(y_len, x_len, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(image_name)
    plt.close()

def dog_filter(sigma_values, filter_size, num_orientations):
    dog_filters = []
    orientation_angles = np.linspace(0, 360, num_orientations, endpoint=False)
    sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    for sigma in sigma_values:
        gaussian_kernel = gauss2d(sigma, filter_size)
        sobel_gaussian_convolution = convolve2d(gaussian_kernel, sobel_operator, mode='same')
        for angle in orientation_angles:
            rotated_filter = rotate(sobel_gaussian_convolution, angle, reshape=False)
            dog_filters.append(rotated_filter)
    return dog_filters

def gauss2d(sigma, size):
    interval = (size - 1) / 2
    x, y = np.ogrid[-interval:interval:size*1j, -interval:interval:size*1j]
    gauss = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    gauss_normalized = gauss / np.sum(gauss)
    return gauss_normalized

def gauss1d(sigma, x, order):
    gauss = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
    if order == 0:
        return gauss
    elif order == 1:
        return -gauss*(x/sigma**2)
    elif order == 2:
        return gauss*((x**2-sigma**2)/sigma**4)
    return gauss

def laplacian(sigma, size):
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    gauss = gauss2d(sigma, size)
    log = convolve2d(gauss, laplacian_filter, mode='same')
    return log

def lm_filter(sigma, size):
    filters = []
    orientations = np.linspace(0, 180, 6)
    for s in sigma[:-1]:
        for order in [1, 2]:
            interval = size / 2.5
            x, y = np.meshgrid(np.linspace(-interval, interval, size), np.linspace(-interval, interval, size))
            gauss_x = gauss1d(3*s, x[0, :], order)
            gauss_y = gauss1d(s, y[:, 0], order)
            for orientation in orientations:
                gauss_kernel = np.outer(gauss_y, gauss_x)
                filter = rotate(gauss_kernel, orientation, reshape=False)
                filters.append(filter)
    for s in sigma:
        filters.append(laplacian(s, size))
        filters.append(laplacian(3*s, size))
        filters.append(gauss2d(s, size))
    return filters

def gabor_filter(sigma, size, num_orientations, wavelength, phase_offset, aspect_ratio):
    gabor_filters = []
    orientation_angles = np.linspace(0, 360, num_orientations, endpoint=False)

    for sigma in sigma:
        sigma_x = sigma
        sigma_y = sigma / aspect_ratio
        n = size // 2
        x, y = np.meshgrid(np.linspace(-n, n, size), np.linspace(-n, n, size))
        x_theta = x
        y_theta = y
        gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / wavelength * x_theta + phase_offset)
        
        for angle in orientation_angles:
            rotated_filter = rotate(gb, angle, reshape=False)
            gabor_filters.append(rotated_filter)

    return gabor_filters
    
def half_disc_masks(scales, orientations=8):
    masks = []
    for scale in scales:
        radius = scale // 2
        for orientation in np.linspace(0, 180, orientations, endpoint=False):
            mask = np.zeros((scale, scale), dtype=np.uint8)
            # Create a white filled half-circle
            cv2.ellipse(mask, (radius, radius), (radius, radius), 0, -orientation, -orientation + 180, 255, -1)
            masks.append(mask)
    return masks

def filter_image(image, filters):
    filtered_images = [cv2.filter2D(image, -1, f) for f in filters]
    return filtered_images

def create_feature_vectors(filtered_images):
    h, w = filtered_images[0].shape[:2]
    feature_vectors = np.zeros((h, w, len(filtered_images)), dtype=np.float32)
    for i, filtered_image in enumerate(filtered_images):
        feature_vectors[:, :, i] = filtered_image
    return feature_vectors.reshape((-1, len(filtered_images)))

def generate_texton_map(image, filters, n_clusters=20):
    filtered_images = filter_image(image, filters)
    feature_vectors = create_feature_vectors(filtered_images)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(feature_vectors)
    labels = kmeans.labels_
    texton_map = labels.reshape(image.shape[:2])
    return texton_map

def generate_brightness_map(image, n_clusters=16):
    pixels = image.reshape((-1, 1))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    brightness_map = kmeans.labels_.reshape(image.shape)
    return brightness_map

def generate_color_map(image, n_clusters=16, color_space='RGB'):
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'YCrCb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    elif color_space == 'Lab':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    color_map = kmeans.labels_.reshape(image.shape[:2])
    return color_map

def gradient(image, bins, hdBank):
    height, width = image.shape
    gradient_accumulator = np.zeros((height, width), dtype=np.float32)
    epsilon = 1e-5
    
    for i in range(0, len(hdBank), 2):
        mask = hdBank[i]
        inv_mask = hdBank[i + 1]
        chi_sqr_dist_pair = np.zeros((height, width), dtype=np.float32)

        for bin_value in range(bins):
            bin_image = (image == bin_value).astype(np.float32)

            g_i = cv2.filter2D(bin_image, -1, mask)
            h_i = cv2.filter2D(bin_image, -1, inv_mask)

            chi_sqr_dist_pair += np.divide(
                (g_i - h_i) ** 2,
                g_i + h_i + epsilon,
                out=np.zeros_like(g_i), where=(g_i + h_i) != 0
            )

        gradient_accumulator += chi_sqr_dist_pair / 2

    gradient_accumulator -= gradient_accumulator.min()
    gradient_accumulator /= (gradient_accumulator.max() + epsilon)

    return gradient_accumulator


    
def main():

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    sigma1 = [5, 10]
    size = 50
    orientations1 = 16
    dog_filters = dog_filter(sigma1, size, orientations1)
    number_of_filters = len(dog_filters)
    y_len = 2
    x_len = (number_of_filters + 1) // y_len
    plot_images(dog_filters, x_len, y_len, image_name='results/Filter/DoG.png')


    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """
    sigma2 = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
    lm_filters = lm_filter(sigma2, size)
    plot_images(lm_filters, x_len = 12, y_len = 4, image_name = 'results/Filter/LM.png', fig_size = (12,4))


    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    sigma3 = [3,5,7,9,12]
    num_orientations = 8
    wavelength = 5
    phase_offset = 0
    aspect_ratio = 0.5
    gabor_filters = gabor_filter(sigma3, size, num_orientations, wavelength, phase_offset, aspect_ratio)
    x_len = num_orientations
    y_len = len(sigma3)
    plot_images(gabor_filters, x_len, y_len, image_name='results/Filter/Gabor.png')
    

    """
    Generate Half-disk masks
    Display all the Half-disk masks and save image as HDMasks.png,
    use command "cv2.imwrite(...)"
    """
    half_disc = half_disc_masks(scales = [5, 10, 15, 20, 25])
    plot_images(half_disc, x_len = 8, y_len = 6, image_name = 'results/Filter/HDMasks.png')

    
    for i in range(1,11):
        image = cv2.imread('../BSDS500/Images/' + str(i) + '.jpg')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        """
        Generate Texton Map
        Filter image using oriented gaussian filter bank
        """
        texton_map = generate_texton_map(gray_image, gabor_filters, n_clusters=20)


        """
        Generate texture ID's using K-means clustering
        Display texton map and save image as TextonMap_ImageName.png,
        use command "cv2.imwrite('...)"
        """
        plt.imshow(texton_map)
        plt.title('Texton Map')
        plt.axis('off')
        plt.savefig('results/Texton/texton_map' + str(i) + '.png')


        """
        Generate Texton Gradient (Tg)
        Perform Chi-square calculation on Texton Map
        Display Tg and save image as Tg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        texton_gradient = gradient(texton_map, 64, half_disc)
        plt.imshow(texton_gradient)
        plt.axis('off')
        plt.savefig('results/Texton/tg' + str(i) + '.png')

        """
        Generate Brightness Map
        Perform brightness binning 
        """
        brightness_map = generate_brightness_map(gray_image, n_clusters=16)
        plt.imshow(brightness_map)
        plt.axis('off')
        plt.savefig('results/Brightness/brightness_map' + str(i) + '.png')


        """
        Generate Brightness Gradient (Bg)
        Perform Chi-square calculation on Brightness Map
        Display Bg and save image as Bg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        brightness_gradient = gradient(brightness_map, 64, half_disc)
        plt.imshow(brightness_gradient)
        plt.axis('off')
        plt.savefig('results/Brightness/bg' + str(i) + '.png')

        """
        Generate Color Map
        Perform color binning or clustering
        """
        color_map = generate_color_map(image, n_clusters=16, color_space='RGB')
        plt.imshow(color_map)
        plt.title('Color Map')
        plt.axis('off')
        plt.savefig('results/Color/color_map' + str(i) + '.png')
        

        """
        Generate Color Gradient (Cg)
        Perform Chi-square calculation on Color Map
        Display Cg and save image as Cg_ImageName.png,
        use command "cv2.imwrite(...)"
        """
        color_gradient = gradient(color_map, 64, half_disc)
        plt.imshow(color_gradient)
        plt.axis('off')
        plt.savefig('results/Color/cg' + str(i) + '.png')


        """
        Read Sobel Baseline
        use command "cv2.imread(...)"
        """
        sobel = cv2.imread('../BSDS500/SobelBaseline/' + str(i) + '.png')
        sobel_gray = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY) if sobel.ndim == 3 else sobel


        """
        Read Canny Baseline
        use command "cv2.imread(...)"
        """
        canny = cv2.imread('../BSDS500/CannyBaseline/' + str(i) + '.png')
        canny_gray = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY) if canny.ndim == 3 else canny


        """
        Combine responses to get pb-lite output
        Display PbLite and save image as PbLite_ImageName.png
        use command "cv2.imwrite(...)"
        """
        pb = (texton_gradient + brightness_gradient + color_gradient) / 3
        pb_normalized = (pb - pb.min()) / (pb.max() - pb.min())
        PbEdges = pb_normalized * (0.5 * sobel_gray + 0.5 * canny_gray) / 255

        plt.imshow(PbEdges)
        plt.axis('off')
        plt.savefig('results/Pb-lite/PbLite_' + str(i) + '.png')

        
if __name__ == '__main__':
    main()