import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def contour(image):
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(
        gray, thresh=0, maxval=255, type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    contours, ret = cv.findContours(
        thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    return contours


root = os.getcwd()
folderPath = os.path.join(r'infer (original pics)')

folderPathResults = 'Contour_Results'
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

        contours = contour(image)

        # create a mask
        mask = np.zeros_like(imgRGB)
        cv.drawContours(mask, contours, -1, (255, 255, 255), cv.FILLED)
        foreground = cv.bitwise_and(imgRGB, mask)

        figure.add_subplot(rows, cols, (i + 1))
        if (i + 1 == 2) or (i + 1 == 12):
            plt.title('contour result')
        plt.axis("off")
        plt.imshow(foreground)

        imagePath = os.path.join(folderPathResults, fileName)
        imgBGR = cv.cvtColor(foreground, cv.COLOR_RGB2BGR)
        cv.imwrite(imagePath, imgBGR)

        i = i + 2

end = time.time()

duration = end - start
print(
    f'R.T. of contour for 10 random pictures is {duration} seconds')

figure.suptitle('Contour Detection', x=0.5, y=0.05)
plt.show()
