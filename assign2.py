################################################################################
# COMP3317 Computer Vision
# Assignment 2 - Corner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d


################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color):
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    trans_coeff = np.array([0.299, 0.587, 0.114])
    img_gray = img_color @ trans_coeff
    return img_gray


################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma):
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result

    # TODO: form a 1D horizontal Guassian filter of an appropriate size
    kernel_size = np.int(np.sqrt(np.log(1000) * 2 * sigma))
    kernel_range = np.arange(-1 * kernel_size, kernel_size+1)
    gaussian_kernel = np.exp((kernel_range**2)/-2/(sigma**2))
    print(gaussian_kernel)

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border
    filtered_image = convolve1d(img, gaussian_kernel, 1, np.float64, 'constant', 0, 0)
    partial_weights_kernel = np.ones(np.shape(img))
    img_weights = convolve1d(partial_weights_kernel, gaussian_kernel, 1, np.float64, 'constant', 0, 0)
    img_smoothed = np.divide(filtered_image, img_weights)

    return img_smoothed


################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma):
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result

    # TODO: smooth the image along the vertical direction
    smoothed_y = smooth1D(img, sigma)
    # TODO: smooth the image along the horizontal direction
    img_smoothed = (smooth1D(smoothed_y.T, sigma)).T

    return img_smoothed


################################################################################
#   perform Harris corner detection
################################################################################
def harris(img, sigma, threshold):
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    # TODO: compute Ix & Iy

    # TODO: compute Ix2, Iy2 and IxIy

    # TODO: smooth the squared derivatives

    # TODO: compute cornesness functoin R

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy

    # TODO: perform thresholding and discard weak corners

    return sorted(corners, key=lambda corner: corner[2], reverse=True)


################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners):
    try:
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners:
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except:
        print('Error occurs in writting output to \'%s\'' % outputfile)
        sys.exit(1)


################################################################################
#   load corners from a file
################################################################################
def load(inputfile):
    try:
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc):
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except:
        print('Error occurs in writting output to \'%s\'' % outputfile)
        sys.exit(1)


################################################################################
#   main
################################################################################
def main():
    parser = argparse.ArgumentParser(description='COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type=str, default='grid1.jpg', help='filename of input image')
    parser.add_argument('-s', '--sigma', type=float, default=1.0, help='sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type=float, default=1e6, help='threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type=str, help='filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try:
        # img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except:
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    plt.imshow(np.uint8(img_color))
    plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    plt.imshow(np.float32(img_gray), cmap='gray')
    plt.show()

    plt.imshow(np.float64(smooth2D(img_gray, 1)), cmap='gray')
    plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)

    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap='gray')
    plt.plot(x, y, 'r+', markersize=5)
    plt.show()

    # save corners to a file
    if args.outputfile:
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)


if __name__ == '__main__':
    main()
