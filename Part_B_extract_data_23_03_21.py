import matplotlib.pyplot as plt
import numpy as np
import struct as st


def extract_data_784(filename, path, leng):
    train_imagesfile = open(filename['images'], 'rb')  # rb ==> read as bytes
    magic, n, rows, cols = st.unpack('>IIII', train_imagesfile.read(16))
    images = np.fromfile(train_imagesfile, dtype=np.uint8).reshape((leng), 784)
    with open(path, "wb") as f:
        np.save(f, np.array(images))
    #return images

def extract_labels(filename, path):
    train_lablesfile = open(filename['labels'], 'rb')
    magic, n = st.unpack('>II', train_lablesfile.read(8))
    labels = np.fromfile(train_lablesfile, dtype=np.uint8)
    with open(path, "wb") as f:
        np.save(f, np.array(labels))

    # print(labels[1])

    return len(labels)

def extract_data(filename, path):
    #filename = {'images': 'train-images.idx3-ubyte'}
    train_imagesfile = open(filename['images'], 'rb')  # rb ==> read as bytes

    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))  # maigc number which is the first four bytes of the training images file

    nImg = st.unpack('>I', train_imagesfile.read(4))[0]  # num of images
    nR = st.unpack('>I', train_imagesfile.read(4))[0]  # num of rows
    nC = st.unpack('>I', train_imagesfile.read(4))[0]  # num of column

    # print(nImg, nR, nC)

    # images_array = np.zeros((nImg, nR, nC))  # creates an array to store all the images at indexs 0-59999
    # not neccessary as the declaration happens at line 24 which accounts for the above

    nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte
    images_array = np.asarray(st.unpack('>' + 'B' * nBytesTotal, train_imagesfile.read(nBytesTotal))).reshape(
        (nImg, nR, nC))
    # print(len(images_array[1]))
    norm_image_arr = []
    #for j in range(len(images_array) - 1):
    #    norm_image_arr.append([i / 255 for i in images_array[j]])
    with open(path, 'wb') as f:
        np.save(f, images_array)
    #return images_array

def main():

    #filename_train_labels = {'labels': 'idx data/train-labels.idx1-ubyte'}
    #filename_test_labels = {'labels': 'idx data/t10k-labels.idx1-ubyte'}

    #train_label_path = "label data/labels_train.npy"
    #test_label_path = "label data/labels_test.npy"

    #length = extract_labels(filename_train_labels, train_label_path)
    #length2 = extract_labels(filename_test_labels, test_label_path)
    #print(length)

    filename_train_images = {'images': 'idx data/train-images.idx3-ubyte'}
    filename_test_images = {'images': 'idx data/t10k-images.idx3-ubyte'}

    train_image_path = "image data 2828/images_train_28.npy"
    test_image_path = "image data 2828/images_test_28.npy"

    #extract_data_784(filename_train_images, train_image_path, length)
    #extract_data_784(filename_test_images, test_image_path, length2)

    #extract_data(filename_train_images, train_image_path)
    #extract_data(filename_test_images, test_image_path)
    with open("image data 2828/images_train_28.npy", "rb") as f:
        images = np.load(f)

    plt.figure
    plt.imshow(images[0])
    plt.title("Plot of first MNIST image")
    plt.show()

main()