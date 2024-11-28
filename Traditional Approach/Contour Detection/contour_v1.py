import cv2 as cv
import matplotlib.pyplot as plt
import os

root = os.getcwd()
imgPath = os.path.join(r'sample_image_bird.jpg')
image = cv.imread(imgPath)

imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

figure = plt.figure()
rows, cols = 1, 3

figure.add_subplot(rows, cols, 1)
plt.title('gray version')
plt.axis("off")
plt.imshow(gray)

# find an approximate estimate of the image
ret, thresh = cv.threshold(gray, thresh=0, maxval=255,
                           type=cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
figure.add_subplot(rows, cols, 2)
plt.title('initial extraction')
plt.axis("off")
plt.imshow(thresh)

# find contours
# mode=cv.RETR_EXTERNAL, retrieves only the outer contours
# method=cv.CHAIN_APPROX_SIMPLE, applies a simpler contour approximation, which saves memory by storing only the endpoints of the contours
contours, ret = cv.findContours(
    thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

# draw contours
# draws all contours/contourIdx=-1 in red color with thickness 1
cv.drawContours(imgRGB, contours, contourIdx=-1,
                color=(255, 0, 0), thickness=1)

figure.add_subplot(rows, cols, 3)
plt.title('final result')
plt.axis("off")
plt.imshow(imgRGB)

figure.suptitle('Contour Detection', x=0.5, y=0.05)
plt.show()
