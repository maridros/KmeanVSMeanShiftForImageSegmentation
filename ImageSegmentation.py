# Drosou Maria
# Department of Informatics And Computer Engineering, University of West Attica
# e-mail: cs151046@uniwa.gr
# A.M.: 151046

import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans
from skimage.color import label2rgb
from sklearn.metrics import f1_score
import pandas as pd


def add_random_noise(image, rows, columns, channels):
    noisy = np.zeros((columns, rows, channels))
    for c in range(channels):
        for x in range(columns):
            for y in range(rows):
                noisy[x, y, c] = abs(round(image[x, y, c]
                                           + 0.05 * np.random.rand() * image[x, y, c]
                                           - 0.05 * np.random.rand() * image[x, y, c]))
    noisy = np.array(noisy, dtype='uint8')
    return noisy


def calc_scores(segmented_image, mask_image, print_info="Image"):

    # detect cluster of target area

    # 1. keep colors only in target area
    region = cv2.bitwise_and(segmented_image, segmented_image, mask=mask_image)

    # 2. get list of colors and frequency of them
    flat_region = np.reshape(region, [-1, 3])
    list_of_colors, counts_of_colors = np.unique(flat_region, axis=0, return_counts=True)

    # 3. keep only non black colors (colors of target area)
    non_black_idx = np.nonzero(list_of_colors)
    region_colors = []
    region_colors_counts = []
    for j in non_black_idx[0]:
        region_colors.append(list_of_colors[j])
        region_colors_counts.append(counts_of_colors[j])

    # 4. get dominant color which defines the predicted target region
    pred_color_idx = np.argmax(region_colors_counts)
    pred_color = region_colors[int(pred_color_idx)]

    # 5. change the segmented image assigning white to the target color-segment and black to the rest
    predicted_image = cv2.inRange(segmented_image, pred_color, pred_color)

    # and illustrate the produced black and white image
    cv2.namedWindow(print_info, cv2.WINDOW_NORMAL)
    cv2.imshow(print_info, predicted_image.astype(np.uint8))
    cv2.resizeWindow(print_info, width, height)
    cv2.waitKey()

    # 7. compare produced image with target image using 2 metrics
    # Metric 1: Intersection Over Union
    intersection = np.logical_and(mask_image, predicted_image)
    union = np.logical_or(mask_image, predicted_image)
    iou_score = np.sum(intersection) / np.sum(union)

    print('. . Intersection Over Union score = ' + str(iou_score))

    # Metric 2: F1 Score
    flat_mask_image = mask_image.flatten()
    flat_predicted_image = predicted_image.flatten()
    f1score = f1_score(flat_mask_image, flat_predicted_image, pos_label=255, average='binary')

    print('. . F1 score = ' + str(f1score))

    return iou_score, f1score


# Create a dataframe for the results
resultsDf = pd.DataFrame(columns=['Noise percentage', 'Algorithm', 'IOU score', 'F1 score'])

# Loading original image in BGR
originImg = cv2.imread('InputData/aeroplane.jpg')

# Loading target image
target = cv2.imread('InputData/mask_annotated_image.png')
mask = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Shape of original image
originShape = originImg.shape
height = originShape[0]
width = originShape[1]

noisy_img = np.copy(originImg)
np.random.seed(6)

for i in range(5):
    noise = str(i * 5) + "%"
    if i > 0:
        # add noise
        noisy_img = add_random_noise(noisy_img, width, height, 3)

    # display image
    title = "Original image with " + noise + " noise"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, noisy_img)
    cv2.resizeWindow(title, width, height)
    cv2.waitKey()

    print("Image with " + noise + " noise")

    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities (or the 3 chanels that I currnelty have)
    flatImg = np.reshape(noisy_img, [-1, 3])

    # here run the meanshift approach
    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flatImg, quantile=0.919, n_samples=200)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Performing meanshift on flatImg
    print('. Using MeanShift algorithm, it takes time!')
    ms.fit(flatImg)
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels = ms.labels_

    # Finding and diplaying the number of clusters
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print(". number of estimated clusters : %d" % n_clusters_)

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])
    segmentedImg = label2rgb(segmentedImg) * 255  # need this to work with cv2.imshow

    title = "MeanShiftSegments to image with " + noise + " noise"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, segmentedImg.astype(np.uint8))
    cv2.resizeWindow(title, width, height)
    cv2.waitKey()

    # Calculate scores
    iou, f1 = calc_scores(segmentedImg, mask, title+" - important cluster mask")
    resultsDf = resultsDf.append({'Noise percentage': noise, 'Algorithm': 'MeanShift',
                                  'IOU score': iou, 'F1 score': f1}, ignore_index=True)

    # performing kmeans on flatImg
    print('. Using kmeans algorithm, it is faster!')
    km = MiniBatchKMeans(n_clusters=2)
    km.fit(flatImg)
    labels = km.labels_

    # Displaying segmented image
    segmentedImg = np.reshape(labels, originShape[:2])
    segmentedImg = label2rgb(segmentedImg) * 255  # need this to work with cv2.imshow

    title = "kmeansSegments to image with " + noise + " noise"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, segmentedImg.astype(np.uint8))
    cv2.resizeWindow(title, width, height)
    cv2.waitKey()

    iou, f1 = calc_scores(segmentedImg, mask, title + " - important cluster mask")
    resultsDf = resultsDf.append({'Noise percentage': noise, 'Algorithm': 'Kmeans',
                                  'IOU score': iou, 'F1 score': f1}, ignore_index=True)

    cv2.destroyAllWindows()

# export the results of all filters to excel file
writer = pd.ExcelWriter('OutputData/Results.xlsx')
resultsDf.to_excel(writer, 'Q3', index=False)
writer.save()
writer.close()
