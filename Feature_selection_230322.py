#PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB


def PCA(X, y):
    # Apply PCA to the dataset
    pca = PCA()
    X_pca = pca.fit_transform(X)

    # Calculate the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_

    # Plot the cumulative sum of the explained variance ratio
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(cumulative_explained_variance_ratio)

    # Label the axes
    plt.xlabel('Number of features')
    plt.ylabel('Cumulative explained variance ratio')
    plt.title('Number of features needed to describe an image')

    # Show the plot
    plt.show()


# CFS



def anova(feature_images, labels):
    scores, pvalues = f_classif(feature_images, labels)
    print(len(scores), "scores")
    X_aaxis = np.arange(len(feature_images[0]))

    plt.bar(X_aaxis, scores)
    #plt.title("ANOVA Scores for the Projection Histogram features")
    plt.xlabel("Feature Index")
    plt.ylabel("Score")
    plt.show()



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

def anova_best_k(feature_images, labels):
    scores, pvalues = f_classif(feature_images, labels)


    nan_indices = np.isnan(scores)

    sorted_indices = np.argsort(scores)[::-1]

    sorted_f_scores = scores[sorted_indices]
    nan_mask = np.isnan(sorted_f_scores)

    sorted_indices = sorted_indices[~nan_mask]
    sorted_f_scores = sorted_f_scores[~nan_mask]

    num_features = len(sorted_f_scores)

    best_k = 0
    best_accuracy = 0

    for k in range(1, num_features+1):
        selected_features = sorted_indices[:k]
        selected_images = feature_images[:, selected_features]
        accuracies = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(selected_images, labels):
            X_train, X_test = selected_images[train_index], selected_images[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            y_pred = gnb.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy - best_accuracy > 0.02:  # increase in accuracy is greater than 0.5%
            best_accuracy = mean_accuracy
            best_k = k
            best_features = sorted_indices[:best_k]

    print("Best k indicies: ", best_k)
    # fig, ax = plt.subplots(figsize=(15, 8))
    # ax.plot(np.arange(1, num_features+1), accuracies)
    # ax.set_xlabel('Number of Features')
    # ax.set_ylabel('Accuracy')
    # plt.show()



def read_in_data(path):
    with open(path, "rb") as f:
        data = np.load(f)

    return data

def read_in_lables():
    with open("label data/labels_train.npy", "rb") as f:
        labels = np.load(f)
    return labels

def main():
    Hog = read_in_data("HOG/HOG_train.npy")


    #Hog = np.reshape(Hog, (60000, 784))

    efd = read_in_data("efd/efd_train.npy")
    efd = np.reshape(efd, (60000, 40))


    lbp = read_in_data("LBP/lpb_import_train_230325.npy")


    pixel = read_in_data("Pixels/pixels_train.npy")


    zen = read_in_data("zen/zen_moments_train.npy")

    hor_his = read_in_data("Histogram/hist_data_horiztonal_train.npy")
    ver_his = read_in_data("Histogram/hist_data_vertical_train.npy")

    his = np.concatenate((hor_his, ver_his), axis=1)
    his = np.reshape(his, (60000, 28 * 2))


    train_features = np.concatenate((efd, Hog, his, lbp, zen), axis=1)
    # test_features = np.concatenate((efd_test, hog_test, his_test, lbp_test, zen_test), axis=1)


    pixel = np.reshape(pixel, (60000, 784))
    labels = read_in_lables()
    class_data=[[] for _ in range(10)]

    for i in range(len(labels)):
        class_data[labels[i]].append(Hog[i])

    anova_best_k(ver_his, labels)

    #anova(his, labels)

    #anova(efd, labels) of thw wrong shape
    #anova(lbp, labels)
    #anova(pixel, labels)
main()