from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pylab
import sys
import math
import numpy as np
from scipy import ndimage
import scipy.misc
import cv2
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray


class countFruit:
    fruitCount = 0

    def __init__(self, filename):
        self.filename = filename

    def numFruit(self):
        print("Number of fruit: %d" % self.fruitCount)

    def edges(self):
        img = cv2.imread(self.filename, 0)
        sigma = 0.33
        # compute the median of the single channel pixel intensities
        v = np.median(img)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        return edged

    def circles(self):
        # load the image, clone it for output, and then convert it to grayscale
        image = cv2.imread(self.filename)
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 1)
        # detect circles in the image
        #circles = cv2.cv.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)


        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # show the output image
            cv2.imshow("output", np.hstack([image, output]))
            cv2.imwrite("output.jpeg", np.hstack([image, output]))

    def circles2(self):
        image = cv2.imread(self.filename, 0)
        image_gray = rgb2gray(image)

        blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

        blobs_list = [blobs_log, blobs_doh]
        colors = ['yellow', 'red']
        titles = ['Laplacian of Gaussian',
                 'Determinant of Hessian']

        sequence = zip(blobs_list, colors, titles)

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        axes = axes.ravel()
        for blobs, color, title in sequence:
            ax = axes[0]
            axes = axes[1:]
            ax.set_title(title)
            ax.imshow(image, interpolation='nearest')
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)
        #plt.show()
        plt.savefig('output.png')
        #cv2.imwrite("output.jpeg", im)

    def AIM(self):
        image = cv2.imread(self.filename, 0)
        #blobs = cv2.imread(blobFile, 0)
        image = cv2.imread(self.filename, 0)
        image_gray = rgb2gray(image)

        blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        plt.imshow(image, interpolation='nearest')
        count = 0


        fig = plt.imshow(image, interpolation='nearest')

        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color= 'red', linewidth=2, fill=False)
            for sector in range(1,9):
                endx1 = x + (10 * r * math.cos(math.radians(45 * sector + 0)))
                endx2 = x + (10 * r * math.cos(math.radians(45 * sector + 15)))
                endx3 = x + (10 * r * math.cos(math.radians(45 * sector + 30)))

                endy1 = y + (10 * r * math.sin(math.radians(45 * sector + 0)))
                endy2 = y + (10 * r * math.sin(math.radians(45 * sector + 15)))
                endy3 = y + (10 * r * math.sin(math.radians(45 * sector + 30)))


                line1 = plt.plot([x, endx1], [y, endy1])
                line2 = plt.plot([x, endx2], [y, endy2])
                line3 = plt.plot([x, endx3], [y, endy3])

                plt.setp(line1, color='r', linewidth=1.0)
                plt.setp(line2, color='r', linewidth=1.0)
                plt.setp(line3, color='r', linewidth=1.0)

        plt.savefig('AIM.jpeg')


apples = countFruit("frame0001.jpeg")
#apples.circles2()
apples.AIM()