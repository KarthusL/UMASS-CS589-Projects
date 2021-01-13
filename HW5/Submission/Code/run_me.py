import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import misc
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os


def read_scene():
    data_x = misc.imread('../../Data/umass_campus.jpg')
    return (data_x)


def K_Means(flattened_image, unfalttened_image):
    image = []
    image.append(reconstruct(unfalttened_image))
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    for k in tqdm(ks):
        model = KMeans(n_clusters = k, n_jobs = -1, max_iter = 1000).fit(flattened_image)
        labels = model.predict(flattened_image)
        empty_image = np.zeros_like(flattened_image)
        for i in range(empty_image.shape[0]):
            empty_image[i] = model.cluster_centers_[labels[i]]
        image.append(reconstruct(empty_image))
    kmeans_plot(image, 3, 3)


def reconstruct(flattened_image):
    return flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])


def kmeans_plot(images, m, n):
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    f, axarr = plt.subplots(m, n, figsize=(15, 15))
    for i, img in tqdm(enumerate(images)):
        axarr[int(i // m), i % n].imshow(img)
        axarr[int(i // m), i % n].axis("off")
        axarr[int(i // m), i % n].set_title("Kmeans with {} clusters".format(ks[i - 1]))
    axarr[0, 0].set_title("Original")
    f.savefig("../Figures/kmeans.png")
    plt.show()


def HAC(flattened_image, unfalttened_image):
    image = []
    image.append(reconstruct(flattened_image))
    ls = ["average", "complete'"]
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    a_s = ["euclidean", "manhattan", "cosine", "l1", "l2"]
    for a in tqdm(a_s):
        for l in tqdm(ls):
            for k in tqdm(ks):
                model = AgglomerativeClustering(n_clusters = k, affinity = a, linkage = l).fit_predict(flattened_image)
                empty_image = np.zeros_like(flattened_image)
                centers = np.zeros((k, 3))
                for i in range(k):
                    points = flattened_image[model == i]
                    mean = np.mean(points, axis = 0)
                    centers[i, :] = mean
                for i in range(empty_image.shape[0]):
                    empty_image[i] = centers[model[i]]
                image.append((reconstruct(empty_image), a, l, k))
    HAC_plot(image, 9, 9)


def HAC_plot(images, m, n):
    print(len(images))
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    f, axarr = plt.subplots(m, n, figsize=(25, 25))
    for i, img in enumerate(images):
        print(i)
        axarr[i // n, i % n].imshow(img[0])
        axarr[i // n, i % n].axis("off")
        axarr[i // n, i % n].set_title("{},{},{}".format(img[1][:2], img[2][:2], img[3]))
    axarr[0, 0].set_title("Original")
    f.savefig("../Figures/HAC.png")
    plt.show()


def HAC_error(flattened_image):
    images = []
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    for k in tqdm(ks):
        #model = KMeans(n_clusters=k, n_jobs=-1, max_iter=1000).fit_predict(flattened_image)
        model = AgglomerativeClustering(n_clusters = k, affinity = "euclidean", linkage= "complete").fit_predict(flattened_image)
        empty_image = np.zeros_like(flattened_image)
        centers = np.zeros((k, 3))
        for i in range(k):
            points = flattened_image[model == i]
            mean = np.mean(points, axis=0)
            centers[i, :] = mean
        for i in range(empty_image.shape[0]):
            empty_image[i] = centers[model[i]]
        print(k)
        images.append(empty_image)
    for image in images:
        difference = flattened_image - image
        diff_sq = np.square(difference)
        mean = np.mean(diff_sq)
        sqrt = np.sqrt(mean)
        print(sqrt)


# def elbow(flattened_image):
#     results = []
#     a = np.arange(15).reshape(3, 5)
#     b = np.arange(15).reshape(3, 5)
#     print(a - b)
#     print(flattened_image)
#     ks = [2, 5, 10, 25, 50, 75, 100, 200]
#     for k in ks:
#         model = KMeans(n_clusters = k, n_jobs = -1, max_iter = 1000, random_state = 0).fit(flattened_image)
#         labels = model.predict(flattened_image)
#         empty_image = np.zeros_like(flattened_image)
#         for i in range(empty_image.shape[0]):
#             empty_image[i] = model.cluster_centers_[labels[i]]
#         print("empty image", empty_image)
#         variance(flattened_image, empty_image)
#
#
# def variance(flattened_image, model_image):
#     flattened_sum = np.array(np.average(flattened_image, axis = 0))
#     print("---", flattened_sum)


def plot():
    ks = [2, 5, 10, 25, 50, 75, 100, 200]
    kmean_data = [9.91209698634, 9.44031602578, 8.74204972151, 7.77224763716, 6.8962187707, 6.27056084679, 5.87201839234, 4.81488317615]
    HAC_data = [9.95085590959, 9.8329327602, 9.31725996918, 8.4213973504, 7.7536915939, 7.29576132651, 6.89471295027, 5.77561829302]
    plt.plot(ks, kmean_data, '--', label = "kmeans")
    plt.plot(ks, HAC_data, '-', label = "HAC")
    plt.legend(loc = "best")
    plt.show()
    plt.savefig("../Figures/compare.png")



if __name__ == '__main__':
    # K-Means
    data_x = read_scene()
    flattened_image = data_x.ravel().reshape(data_x.shape[0] * data_x.shape[1], data_x.shape[2])
    print('Implement k-means here ...')
    #K_Means(flattened_image, data_x)
    print('Implement HAC here ...')
    #HAC(flattened_image, data_x)
    print("implement Error here ...")
    #HAC_error(flattened_image)
    #elbow(flattened_image)
    #plot()
    reconstructed_image = flattened_image.ravel().reshape(data_x.shape[0], data_x.shape[1], data_x.shape[2])

