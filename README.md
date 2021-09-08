# KmeanVSMeanShiftForImageSegmentation
Comparison of two clustering algorithms (K-means and Mean-Shift) for image segmentation after adding different noise levels.
## Requirements
- Python
- Numpy
- OpenCV
- Scikit-learn
- Scikit-image
- Pandas
## Code process and results
This code segments an image using two different clustering methods (K-means, Mean-Shift). This process is reapeated for several noise levels (0%, 5%, 10%, 15%, 20%) and the results are evaluated based on two defferent metrics (Intersection over union (IOU), F1 score). The goal is to locate an object-area of interest in the image as one of the clusters.
### Input Data
The image that was used is located in InputData folder (aeroplane.jpg). It is an image of a flying aeroplane. In the same folder there is also an annotated image (desired outcome), where the aeroplane is white and the rest image (sky) is black. The first image is the image which is processed by the clustering algorithms and the second one is the image which is used to compare the results of the algorithms whith the desired outcome.
### Metrics
The first metric is Intersection Over Union, which measures the degree to which the two areas defined by the two images coincide spatially with each other. So, in simple words it is the intersection of the results of the two images towards their union. The second metric is the F1 score, which is a combination of Recall and Precision score. Recall mesures how many of the pixels in the area of interest were detected as pixels of this area. Precision measures how many of the pixels classified as pixels of the area of interest are truly pixels of this area. F1 score is the harmonic mean of Precision and Recall.
### Results
![segmentationResults](https://user-images.githubusercontent.com/89779679/132486641-2c75dd1d-098e-41d9-9679-42ca3ca8f4fb.jpg)

From the results it is obvious that after 10% noise MeanShift fails, but Kmeans keep up with a high score in both metrics.
