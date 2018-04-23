import numpy as np
from skimage.morphology import square
from skimage.morphology import disk
from skimage.filters.rank import median
from matplotlib import pyplot as plt
import cv2

corePath = "/home/piotr/Desktop/PycharmProjects/Vessels Recognition/images/all/"
imagesDir = "images/"
maskDir = "mask/"
imagesEnding = "_dr.JPG"
maskEnding = "_dr_mask.tif"

def prepareImage(fileName):
    image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(image)
    #cl = clahe.apply(cl)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    cl = clahe.apply(cl)
    cl = cv2.erode(cl, square(3), iterations=3)
    cl = cv2.erode(cl, square(2), iterations=3)

    return cl

def whiten(image, i, j, n):
    for k in range(i, i + n):
        for l in range(j, j + n):
            image[k][l] = 255
    return image

def drawVessels(fileName, preparedImage):
    image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    halfSquareEdge = 200 #polowa dlugosci kwadratu z ktorego liczymy percentyl
    step = 11

    #przesuwam sie po obrazie kwadratem o rozmiarze step x step,
    # jesli srednia wartosc pikseli z tego kwadratu jest mniejsza od 35 percentyla wartosci pikseli z otoczenia (kwadrat o boku dlugosci 2 * halfSquareEdge)
    # to wybielam caly kwadrat step x step
    for i in range(step, len(image) - step, step):
        for j in range(step, len(image[i]) - step, step):
            if np.mean(preparedImage[i - step : i + step, j - step : j + step]) < np.percentile(preparedImage[max(0, i - halfSquareEdge):min(len(image) - 1, i + halfSquareEdge), max(0, j - halfSquareEdge):min(len(image[i]) - 1, j + halfSquareEdge)], 35):
                image = whiten(image, i, j, step)

    return image

def drawSubplot(filePath, ax1, ax2):
    originalImage = cv2.imread(corePath + imagesDir + "01" + imagesEnding, cv2.IMREAD_COLOR)
    originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(corePath + maskDir + "01" + maskEnding, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV)

    preparedImage = prepareImage(filePath)
    image = drawVessels(filePath, preparedImage)
    image = mask + image

    ax1.imshow(originalImage)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(image, cmap='gray')
    ax2.set_xticks([])
    ax2.set_yticks([])

def main():
    figure, (ax1, ax2) = plt.subplots(1, 2)
    figure.set_size_inches(19, 13)

    drawSubplot(corePath + imagesDir + "01" + imagesEnding, ax1, ax2)

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()