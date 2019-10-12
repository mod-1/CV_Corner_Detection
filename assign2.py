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

    # using the Y channel of the YIQ model to perform the conversion
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

    # form a 1D horizontal Gaussian filter of an appropriate size
    kernel_size = np.int(np.sqrt(np.log(1000) * 2 * sigma))
    kernel_range = np.arange(-1 * kernel_size, kernel_size + 1)
    gaussian_kernel = np.exp((kernel_range ** 2) / -2 / (sigma ** 2))

    # convolve the 1D filter with the image;
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

    # smooth the image along the vertical direction
    smoothed_y = smooth1D(img.T, sigma).T
    # smooth the image along the horizontal direction
    img_smoothed = smooth1D(smoothed_y, sigma)

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

    # compute Ix & Iy
    finite_diff_kernel = np.array([0.5, 0, -0.5])
    Ix = convolve1d(img, finite_diff_kernel, 1, np.float64, 'reflect', cval=0)
    Ix[:, 0] = img[:, 1] - img[:, 0]
    Ix[:, (Ix.shape[1] - 1)] = img[:, img.shape[1] - 1] - img[:, img.shape[1] - 2]
    Iy = convolve1d(img, finite_diff_kernel, 0, np.float64, 'reflect', cval=0)
    Iy[0] = img[1] - img[0]
    Iy[Iy.shape[0] - 1] = img[img.shape[0] - 1] - img[img.shape[0] - 2]
    # compute Ix2, Iy2 and IxIy
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    IxIy = np.multiply(Ix, Iy)
    # smooth the squared derivatives
    Ix_smooth = smooth2D(Ix2, sigma)
    Iy_smooth = smooth2D(Iy2, sigma)
    IxIy_smooth = smooth2D(IxIy, sigma)
    # compute cornesness function R
    R = ((Ix_smooth * Iy_smooth) - (IxIy_smooth * IxIy_smooth)) - (
                0.04 * ((Ix_smooth + Iy_smooth) * (Ix_smooth + Iy_smooth)))
    # mark local maxima as corner candidates;
    # perform quadratic approximation to local corners upto sub-pixel accuracy
    # perform thresholding and discard weak corners
    corners = []
    for row in range(1, len(R) - 1):
        for col in range(1, len(R[0]) - 1):
            if R[row][col] > R[row - 1][col - 1] and R[row][col] > R[row - 1][col] and R[row][col] > R[row - 1][
                col + 1] and R[row][col] > R[row][col - 1] and R[row][col] > R[row][col + 1] and R[row][col] > \
                    R[row + 1][col - 1] and R[row][col] > R[row + 1][col] and R[row][col] > R[row + 1][col + 1]:
                e = R[row][col]
                a = (R[row - 1][col] + R[row + 1][col] - (2 * e)) / 2
                b = (R[row][col - 1] + R[row][col + 1] - (2 * e)) / 2
                c = (R[row + 1][col] - R[row - 1][col]) / 2
                d = (R[row][col + 1] - R[row][col - 1]) / 2
                x = -c / (2 * a)
                y = -d / (2 * b)
                cornerness = (a * (x ** 2)) + (b * (y ** 2)) + (c * x) + (d * y) + e
                if cornerness > threshold:
                    corners.append((col + x, row + y, cornerness))

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
        print('Error occurs in writing output to \'%s\'' % outputfile)
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
        print('Error occurs in writing output to \'%s\'' % outputfile)
        sys.exit(1)


################################################################################
#   main
################################################################################
def main():
    parser = argparse.ArgumentParser(description='COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type=str, default='grid1.jpg', help='filename of input image')
    parser.add_argument('-s', '--sigma', type=float, default=1.0, help='sigma value for Gaussian filter')
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
    # plt.imshow(np.uint8(img_color))
    # plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    # plt.imshow(np.float32(img_gray), cmap='gray')
    # plt.show()

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
