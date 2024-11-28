import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def watershed(image):

    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(
        gray, thresh=0, maxval=255, type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations=3)

    dist_transform = cv.distanceTransform(
        opening, distanceType=cv.DIST_L2, maskSize=5)
    ret, sure_fg = cv.threshold(
        dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = np.int32(markers)
    markers = cv.watershed(imgRGB, markers)

    return markers


root = os.getcwd()
folderPath = os.path.join(r'infer (original pics)')

folderPathResults = 'Watershed_Results'
os.makedirs(folderPathResults, exist_ok=True)

figure = plt.figure()
rows, cols = 10, 2
i = 1

start = time.time()

for fileName in os.listdir(folderPath):
    if fileName.endswith(".jpg"):
        imgPath = os.path.join(folderPath, fileName)
        image = cv.imread(imgPath)
        imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        figure.add_subplot(rows, cols, i)
        if i == 1 or i == 11:
            plt.title('original picture')
        plt.axis("off")
        plt.imshow(imgRGB)

        labels = watershed(image)

        for label in np.unique(labels):

            # skip the background label (usually 0)
            if label == 0:
                continue

            # create a mask for the current object
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[labels == label] = 255

            # find the contours of the object
            contours, ret = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(imgRGB, (x, y), (x + w, y + h), (0, 255, 0), 2)

        figure.add_subplot(rows, cols, (i + 1))
        if (i + 1 == 2) or (i + 1 == 12):
            plt.title('watershed result')
        plt.axis("off")
        plt.imshow(imgRGB)

        imagePath = os.path.join(folderPathResults, fileName)
        imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)
        cv.imwrite(imagePath, imgBGR)

        i = i + 2

end = time.time()

duration = end - start
print(
    f'R.T. of watershed algorithm for 10 random pictures is {duration} seconds')

figure.suptitle('Watershed Algorithm', x=0.5, y=0.05)
plt.show()
