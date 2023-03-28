import numpy as np
import pyefd
from skimage.feature import hog
from skimage.feature import canny as cn
from skimage.measure import find_contours as fd
from skimage.measure import centroid as cen
from scipy.special import comb
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import skimage.measure as measure

def historgram_feature_1(image_array):  # shape of array is fine
    horizontal_histogram = []
    vertical_histogram = []

    for i in image_array:
        horizontal_histogram.append(np.sum(i, axis=0))
        vertical_histogram.append(np.sum(i, axis=1))

    with open("Histogram/hist_data_horiztonal_test.npy", 'wb') as f:
        np.save(f, horizontal_histogram)
    with open("Histogram/hist_data_vertical_test.npy", "wb") as f:
        np.save(f, vertical_histogram)

    skifihwd = "done"

    return skifihwd


def HOG_feature_2(images):  # shape fixed
    # step 1 calcualte the image gradeint
    hog_feature = []

    hog_feature_array_temp = []
    for i in images:

        hog_feature_tt = hog(i, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(2, 2))
        #print(hog_feature_tt, "in loop")
        hog_feature_array_temp.append(hog_feature_tt)


    hog_feature = hog_feature_array_temp
    print(np.shape(hog_feature))

    # print(hog_features.shape)

    with open("HOG/HOG_test.npy", "wb") as f:
        np.save(f, hog_feature)

    return hog_feature


def feature_3_EFD(images, n):
    # compute EFDs for the images in the dataset
    efd_coeffs = []
    i = 0
    for image in images:
        image_grey = image / 255

        # Apply the Canny edge detector to the image to extract its contour
        edges = cn(image_grey, high_threshold=0.3)

        # print(edges)
        # Find the contours in the edge image
        contours = fd(image, 0.5)
        # print(contours)
        # Calculate the elliptic Fourier descriptors of the contour
        efd = pyefd.elliptic_fourier_descriptors(contours[0], order=n)
        # pyefd.plot_efd((efd))

        # Store the EFD coefficients for the image
        efd_coeffs.append(efd)

    efd_coeffs = np.array(efd_coeffs)

    with open("efd/efd_test.npy", "wb") as f:
        np.save(f, efd_coeffs)
    cccccc = "done"
    return cccccc


def feature_4_pixels(images):
    pixels=[]
    for image in images:
        image=image/255
        #image=image.flatten()
        pixels.append(image)

    # print(feature_images.shape[0], "shape zero")
    #scores, pvalues = f_classif(feature_images, labels)
    # print(np.shape(scores), "scores")
    print(pixels[0])
    mean = np.mean(pixels[0], axis=0)
    print(mean, "mean")

    with open("Pixels/pixels_train.npy", "wb") as f:
        np.save(f, pixels)
    return pixels



def extract_features(img):
    # Load the template image
    template = Image.open('template.png')

    # Convert the input image and template to grayscale
    img_gray = Image.fromarray(img).convert('L')
    template_gray = template.convert('L')

    # Get the size of the input image and template
    img_width, img_height = img_gray.size
    template_width, template_height = template_gray.size

    # Find the best match using template matching
    result = np.zeros((img_height - template_height, img_width - template_width))
    for y in range(img_height - template_height):
        for x in range(img_width - template_width):
            # Extract the region of interest
            roi = img_gray.crop((x, y, x + template_width, y + template_height))

            # Calculate the mean squared error between the template and the ROI
            mse = np.mean(np.square(np.array(roi) - np.array(template_gray)))

            # Store the result
            result[y, x] = mse

    # Return the minimum value of the result
    return np.min(result)

def get_lbp_feature(image, radius, n_points):
    rows, cols = image.shape
    lbp_feature = np.zeros(n_points, dtype=np.uint8)

    # center of circle
    cx, cy = radius, radius

    # compute LBP feature
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        px = int(cx + x)
        py = int(cy + y)

        # calculate difference between center pixel and neighboring pixel
        if px >= cols or py >= rows or px < 0 or py < 0:
            lbp_code = 0
        else:
            lbp_code = int(image[py, px] >= image[cy, cx])

        # add the binary code to the LBP feature
        lbp_feature[i] = lbp_code

    # convert LBP feature to decimal value
    decimal_value = 0
    for i in range(n_points):
        decimal_value += lbp_feature[i] * 2 ** i

    return decimal_value


def extract_lbp_features(images, radius, n_points):
    lbp_features = []
    for image in images:
        #image = np.reshape(image, (28, 28))

        # pad image to avoid border effects
        padded_image = np.pad(image, radius, mode='edge')

        # compute LBP feature for each pixel in image
        rows, cols = image.shape
        lbp_image = np.zeros((rows, cols), dtype=np.uint32)
        for row in range(rows):
            for col in range(cols):
                lbp_feature = get_lbp_feature(padded_image[row:row + 2 * radius + 1, col:col + 2 * radius + 1], radius,
                                              n_points)
                lbp_image[row, col] = lbp_feature

        # histogram of LBP features
        histogram, _ = np.histogram(lbp_image, bins=2 ** n_points, range=(0, 2 ** n_points))
        lbp_features.append(histogram)

    #lbp_features = np.array(lbp_features)

    with open("LBP/lpb_train.npy", "wb") as f:
        np.save(f, lbp_features)
    cccccc = "done"
    return cccccc

    return lbp_features


def feature_4_zernike(images, n_moments):
    zen_moments = []
    for image in images:
        image = image/255
        #image_2d = np.reshape(image, (28,28))
        thresh = 0.5  # choose an appropriate threshold value
        binary = image > thresh #converts to binary image

    # Compute the centroid of the binary image
        r, c = measure.centroid(binary)
        cy, cx = r, c

    # Translate the image so that the centroid is at the center
        rows, cols = binary.shape
        tx, ty = cols/2 - cx, rows/2 - cy
        binary_translated = np.roll(np.roll(binary, int(tx), axis=1), int(ty), axis=0)

    # Npipormalize the image so that it has unit radius
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols))
        rr = rr - rows/2
        cc = cc - cols/2
        r = np.sqrt(rr**2 + cc**2)
        rmax = r.max()
        r[rmax == 0] = 1
        rr = rr / rmax
        cc = cc / rmax
        binary_normalized = binary_translated

    # Compute the Zernike moments of the normalized image
        moments = []
        nmax =  n_moments # maximum degree of Zernike moments to compute
        mag_moments=[]
        for n in range(nmax+1):
            for m in range(-n, n+1, 2):
                if m == 0:
                    p = lambda x, y: np.sqrt(n + 1) * np.pi * comb(n, n // 2) * rr ** n
                elif m > 0:
                    p = lambda x, y: np.sqrt(2 * (n + 1)) * np.pi * comb(n, (n - m) // 2) * rr ** (n - m) * np.cos(m * np.arctan2(cc, rr))
                else:
                    p = lambda x, y: np.sqrt(2 * (n + 1)) * np.pi * comb(n, (n + m) // 2) * rr ** (n + m) * np.sin(-m * np.arctan2(cc, rr))
                moments.append(abs(np.sum(p(rr, cc) * binary_normalized)))
        #print(np.shape(moments))
        zen_moments.append(moments)
        #print(np.shape(zen_moments), "zen")

    print(np.shape(zen_moments[0]))
    with open("zen/zen_moments_test.npy", "wb") as f:
        np.save(f, np.array(zen_moments))
    return zen_moments

def feature_5_lpb(images):
    images = images
    radius = 3
    n_points = 8 * radius
    X_lbp = []
    j=0
    for i in images:
        lbp = local_binary_pattern(i, n_points, radius, method='uniform')
        # Compute the histogram of LBP patterns
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype(np.float32)
        hist /= np.sum(hist)
        X_lbp.append(hist)
    print(np.shape(X_lbp))

    with open("LBP/lpb_import_train_230325.npy", "wb") as f:
        np.save(f, X_lbp)


def main():
    with open("image data 2828/images_test_28.npy", 'rb') as f:
        train_data = np.load(f)
    #feature_5_lpb(train_data)

    # with open("image data/images_test_784.npy", 'rb') as f:
    #    test_data = np.load(f)

    #HOG_feature_2(train_data)
    #historgram_feature_1(train_data)
    #feature_3_EFD(train_data, 10)
    #extract_lbp_features(train_data, 3, 24)
    #feature_4_pixels(train_data)

    #feature_4_zenrike(train_data, 10)
    #with open("zen_moments_train.npy", "wb") as f:
    #    np.save(f, np.array(zen))
    #    print(len(np.array(zen)))
    #with open("zen/zen_moments_test.npy", "wb") as f:
    #    np.save(f, np.array(zen))

    with open("Histogram/hist_data_vertical_train.npy", "rb") as f:
       zen = np.load(f)
    x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    print(np.shape(zen[0]))
    plt.figure()
    plt.plot(x,zen[0])
    plt.title("Vertical histogram feature of the first image in MNIST training set")
    plt.show()
    #print((zen[0]))
    #print(zen[0:10])


main()
