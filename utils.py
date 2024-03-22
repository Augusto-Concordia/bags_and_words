from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist


def computeHistogram(img_file, F, textons):

    # YOUR CODE HERE

    img = img_as_float(io.imread(img_file))
    img = rgb2gray(img)

    filter_qty = F.shape[2]
    k = textons.shape[0]

    assert img.shape[0] == img.shape[1], 'Image is not square'
    assert filter_qty == textons.shape[1], 'Filter qty does not match textons column count'

    # compute filter responses
    responses = np.zeros((img.shape[0], img.shape[1], filter_qty))

    for filter_index in range(filter_qty):
        responses[:, :, filter_index] = correlate(
            img, F[:, :, filter_index], mode='constant', cval=0)

    # compute histogram
    histogram = np.zeros(k)

    # for each pixel, find the closest texton and increment the corresponding histogram bin
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distances = cdist(responses[i, j, :].reshape(1, -1), textons)
            histogram[np.argmin(distances)] += 1

    # END YOUR CODE
    return histogram


def createTextons(F, file_list, K):

    # YOUR CODE HERE

    PIXEL_SAMPLE_SIDE_COUNT = 75

    file_qty = len(file_list)
    filter_qty = F.shape[2]

    collected_sampled_responses = np.zeros(
        (file_qty * PIXEL_SAMPLE_SIDE_COUNT * PIXEL_SAMPLE_SIDE_COUNT, filter_qty))

    for file_index, file_name in enumerate(file_list):
        img = img_as_float(io.imread(file_name))
        img = rgb2gray(img)

        assert img.shape[0] == img.shape[1], 'Image is not square'

        # randomly sample pixels, choose a starting point to sample a 10x10 patch (or whatever the size is)
        if img.shape[0] - PIXEL_SAMPLE_SIDE_COUNT <= 0:
            random_pixels = np.zeros((filter_qty, 2), dtype=np.uint8)
        else:
            random_pixels = np.random.randint(
                0, img.shape[0] - PIXEL_SAMPLE_SIDE_COUNT, size=(filter_qty, 2))

        # image responses for each filter
        responses = np.zeros((img.shape[0] * img.shape[1]))

        # compute filter responses
        for filter_index in range(filter_qty):
            responses = correlate(
                img, F[:, :, filter_index], mode='constant', cval=0)

            collected_sampled_responses[
                file_index * PIXEL_SAMPLE_SIDE_COUNT * PIXEL_SAMPLE_SIDE_COUNT: (file_index + 1) * PIXEL_SAMPLE_SIDE_COUNT * PIXEL_SAMPLE_SIDE_COUNT,
                filter_index] = responses[
                    random_pixels[filter_index, 0]:random_pixels[filter_index, 0] + PIXEL_SAMPLE_SIDE_COUNT,
                    random_pixels[filter_index, 1]:random_pixels[filter_index, 1] + PIXEL_SAMPLE_SIDE_COUNT].reshape(-1)

    # K-means clustering
    k_means = sklearn.cluster.KMeans(n_clusters=K)
    k_means.fit(collected_sampled_responses)

    # END YOUR CODE
    return k_means.cluster_centers_
