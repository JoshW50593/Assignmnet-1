import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_data(filename):
    data=[]
    with open(filename, 'rb') as f:
        data = np.load(f)
    return data


def extract_class_feature_data(labels, X):
    classes = np.unique(labels)
    separated_by_class = [[] for _ in range(len(classes))]
    for i in range(len(labels)):
        separated_by_class[labels[i]].append(X[i])

    return separated_by_class


def mean_func(labels, X_in):
    X = extract_class_feature_data(labels, X_in)
    #print(X.)
    mean=[]
    for i in range(len(X)):
        mean.append(np.mean(X[i], axis=0))
    #print(np.shape(mean))
    return mean

def std_func(labels, X_in):
    X = extract_class_feature_data(labels, X_in)
    std=[]
    for i in range(len(X)):
        #print(np.shape(X[i]))
        std_temp = np.std(X[i], axis=0)
        if np.any(std_temp == 0):
            zeros_mask = std_temp == 0
            std_temp[zeros_mask] = 1e-2

        std.append(std_temp)

    #print(np.shape(std))
    return std

def anova(feature_images, labels):
    scores, pvalues = f_classif(feature_images, labels)
    print(len(scores), "scores")
    X_aaxis = np.arange(len(feature_images[0]))
    #plt.bar(X_aaxis, scores)

    #plt.show()
    nan_indices = np.isnan(scores)


    sorted_indices = np.argsort(scores)[::-1]

    sorted_f_scores = scores[sorted_indices]
    nan_mask = np.isnan(sorted_f_scores)

    sorted_indices = sorted_indices[~nan_mask]
    sorted_f_scores = sorted_f_scores[~nan_mask]

    num_features = len(sorted_f_scores)
    accuracies = []
    for k in range(1, num_features+1):
        selected_features = sorted_indices[:k]
        selected_images = feature_images[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(selected_images, labels, test_size=0.2, random_state=42)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(np.arange(1, num_features+1), accuracies)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Accuracy')
    plt.show()

def features_means_std_write_to_file(labels):
    hog_train = get_data("HOG/HOG_train.npy")
    hog_test = get_data("HOG/HOG_test.npy")
    # print(np.shape(hog_train), "hog shape")
    efd_train = get_data("efd/efd_train.npy")
    efd_test = get_data("efd/efd_test.npy")
    # print(np.shape(efd_train), "efd shaape")
    efd_train = np.reshape(efd_train, (60000, 40))
    efd_test = np.reshape(efd_test, (10000, 40))

    His_ver_train = get_data("Histogram/hist_data_vertical_train.npy")
    # print(np.shape(His_ver_train), "his ver train shape")
    His_hor_train = get_data("Histogram/hist_data_horiztonal_train.npy")
    # print(np.shape(His_hor_train), "his hor train shape")
    His_ver_test = get_data("Histogram/hist_data_vertical_test.npy")
    His_hor_test = get_data("Histogram/hist_data_horiztonal_test.npy")

    lbp_train = get_data("LBP/lpb_import_train.npy")
    # print(np.shape(lbp_train), "shape lbp trina")
    lbp_train = np.reshape(lbp_train, (60000, 784))
    lbp_test = get_data("LBP/lpb_import_test.npy")
    lbp_test = np.reshape(lbp_test, (10000, 784))

    pixels_train = get_data("Pixels/pixels_train.npy")
    pixels_test = get_data("Pixels/pixels_test.npy")
    pixels_train = np.reshape(pixels_train, (60000, 784))
    pixels_test = np.reshape(pixels_test, (10000, 784))
    zen_train = get_data("zen/zen_moments_train.npy")
    # print(np.shape(zen_train), "zen train shape")

    zen_test = get_data("zen/zen_moments_test.npy")

    train_sep_mean = mean_func(labels, zen_train)
    train_sep_std = std_func(labels, zen_train)


    efd_train_sep_mean = mean_func(labels, zen_train)
    efd_train_sep_std = std_func(labels, zen_train)

    with open("zen/zen_train_class_mean.npy", "wb") as f:
        np.save(f, efd_train_sep_mean)
    with open("zen/zen_train_class_std.npy", "wb") as f:
        np.save(f, efd_train_sep_std)

    return "complete"

def anova(feature_images, labels):
    scores, pvalues = f_classif(feature_images, labels)
    X_aaxis = np.arange(len(feature_images[0]))
    plt.bar(X_aaxis, scores)
    plt.title("ANOVA Scores for the Vertical Project Histogram features")
    plt.xlabel("Feature Index")
    plt.ylabel("Score")
    plt.show()



def my_naive_bayes(X_in, priors, mean, std):

    log_priors = np.log(priors)
    predict = []
    #gaussian_log_likelihood = (pdf(X_in, std, mean, labels))
    #print(len(gaussian_log_likelihood))
    #print(len(X_in))
    #print(len(X_in), "xin len")
    std = np.array(std)
    mean = np.array(mean)
    for i in range(len(X_in)):
        post = []
        for j in range(len(log_priors)):
            # if np.any(std[j]==0):
            #     std[j]=np.where(std[j]==0, std[j], 1e-6)
            scale = 1 / (std[j] * np.sqrt(2 * np.pi))
            gaus = (np.exp(-0.5*(((X_in[i])-mean[j])/std[j])**2))
            guas_dis = np.log(scale)+np.log(gaus)
            guas_dis = (np.sum(guas_dis))
            post.append(np.exp(guas_dis + log_priors[j]))
        predict.append(np.argmax(post))

        #print(len)
    print((predict[0]))
    return predict

def predict_josh(X_in, priors, mean, std):
    predictions = my_naive_bayes(X_in, priors, mean, std)
    return predictions



def main():


    labels = get_data("label data/labels_train.npy")
    labels_test = get_data("label data/labels_test.npy")

    #hog_train_sep = extract_class_feature_data(labels, "HOG/HOG_train.npy")


    priors = np.bincount(labels)/len(labels)
    #print(efd_priors)

    efd_train = get_data("efd/efd_train.npy")
    efd_test = get_data("efd/efd_test.npy")
    efd_train = np.reshape(efd_train, (60000, 40))
    efd_test = np.reshape(efd_test, (10000, 40))

    hog_train = get_data("HOG/HOG_train.npy")
    hog_test = get_data("HOG/HOG_test.npy")


    pixel_train = get_data("Pixels/pixels_train.npy")
    pixel_test = get_data("Pixels/pixels_test.npy")
    pixel_train=pixel_train/255
    pixel_test=pixel_test/255

    pixel_train = np.reshape(pixel_train, (60000, 784))
    pixel_test = np.reshape(pixel_test, (10000, 784))

    pixel_meen = np.mean(pixel_train, axis=0)
    pixel_stdee = np.std(pixel_train, axis=0)



    pixel_train_norm = (pixel_train - pixel_meen) / pixel_stdee
    pixel_test_norm = (pixel_test - pixel_meen) / pixel_stdee

    his_hor_train = get_data("Histogram/hist_data_horiztonal_train.npy")
    his_ver_train = get_data("Histogram/hist_data_vertical_train.npy")
    his_train = np.concatenate([his_hor_train, his_ver_train], axis=1)
    his_train_reshape = np.reshape(his_train, (60000, 28 * 2))


    his_hor_test = get_data("Histogram/hist_data_horiztonal_test.npy")
    his_ver_test = get_data("Histogram/hist_data_vertical_test.npy")

    his_test = np.concatenate([his_hor_test, his_ver_test], axis=1)
    his_test = np.reshape(his_test, (10000, 28*2))



    lbp_train = get_data("LBP/lpb_import_train_230325.npy")
    lbp_test = get_data("LBP/lpb_import_test.npy")

    zen_train = get_data("zen/zen_moments_train.npy")
    zen_test = get_data("zen/zen_moments_test.npy")


    zen_std = std_func(labels, zen_train)
    zen_mean = mean_func(labels, zen_train)

    #X_train =

    nb_classifier = GaussianNB()
    nb_classifier.fit(zen_train, labels)



    predict = nb_classifier.predict(zen_test)
    predict_mine = predict_josh(zen_test, priors, zen_mean, zen_std)

    acr_sklearn = accuracy_score(predict, labels_test)
    acr_mine = accuracy_score(predict_mine, labels_test)

    print("acr skleanr:",acr_sklearn,"acr mine:", acr_mine)

    cm = confusion_matrix(labels_test, predict_mine)


    # plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title("Zernike moments confusion matrix before feature selection")
    plt.show()


    #efd: sk=85.45, mine=85.45 fine
    #HOG: sk=83.52 mine=83.52 fine
    #Pxiels: sk=65.51 mine =9.8
    #histogram: sk=60.17 mine = 58.9 fine
    #LBP: sk=45.31 mine=45.31 fine
    #zen: sk=17.45 mine=17.45 fine







main()