import pdb
import pylab
import time
import sys
import cv2
import math
from math import sqrt
from scipy import ndimage
import scipy.misc
import numpy as np
from itertools import cycle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh, hog
from skimage.color import rgb2gray
from skimage import data, color, exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import pickle
# -------------------------------------------------------------------------------------------------
# Comment out the following 2 lines before compiling. I'm using a virtual environment to run OpenCV
# and Matplotlib doesn't behave very well with it
#import matplotlib
#matplotlib.use('TkAgg')
# -------------------------------------------------------------------------------------------------

class countFruit:
    fruitCount = 0 #currently unused
    images = [] #all images ot train classifer
    filenames = [] #filenames containing all images
    all_blobs = [] #blob_doh for every image

    def __init__(self, files = None):
        if files == None:
            self.filenames = []
        else:
            self.filenames = files
            for i in range(0, len(self.filenames)):
                image = cv2.imread(self.filenames[i])
                self.images.append(image)
                image_gray = rgb2gray(image)
                #image_gray = scipy.misc.imresize(image_gray, 2.5)
                blobs = blob_doh(image_gray, max_sigma=39, threshold=.01)
                # Compute radii in the 3rd column.
                #blobs[:, 2] = blobs[:, 2] * sqrt(2)
                self.all_blobs.append(blobs)

    '''
    Returns the image at
    filenames[0] with edges detected
    '''
    def edges(self):
        img = cv2.imread(self.filenames[0], 0)
        sigma = 0.33
        # compute the median of the single channel pixel intensities
        v = np.median(img)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(img, lower, upper)
        return edged

    '''
    Retuns an array of blobs.
    Each blob is of the form [x,y,r]
    '''
    def circles(self, filename):
        image = cv2.imread(filename, 0)
        image_gray = rgb2gray(image)

        blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

        blobs_list = [blobs_log, blobs_doh]
        colors = ['yellow', 'red']
        titles = ['Laplacian of Gaussian', 'Determinant of Hessian']
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

        plt.savefig('output.png')
        plt.show()


    '''
    Returns an array of integers for each blob.

    This array of integers represents the distance
    from the center of the apple where the change in
    intensitiy is greater than or equal to .10. In other
    words, the each array represents the radius of a new
    intensity band
    '''
    def AIM(self, blobs, image):
        image_gray = rgb2gray(image)
        height, width = image_gray.shape[:2]

        num_blobs = 0;
        print("blobs_size", len(blobs))
        Y = []
        blob_AIMs = []

        for blob in blobs:
            IntDrops = [0,0,0,0,0,0,0,0,0]
            num_blobs = num_blobs + 1
            y, x, r = blob
            #print("x: ", x)
            #print("y: ", y)
            #print("r: ", r)
            #print("________")
            #c = plt.Circle((x, y), r, color= 'red', linewidth=2, fill=False)
            #currentAxis = plt.gca()
            #currentAxis.add_patch(Rectangle((x - r, y - r), 2*r, 2*r, fill=None, alpha=1))


            #plt.imshow(image_gray)
            curr_radius = 1
            x_init1 = x
            x_init2 = x
            x_init3 = x
            y_init1 = y
            y_init2 = y
            y_init3 = y

            x_end1 = x
            x_end2 = x
            x_end3 = x
            y_end1 = y
            y_end2 = y
            y_end3 = y

            intensities = [[],
                           [],
                           [],
                           [],
                           [],
                           [],
                           [],
                           [],
                           []]


            coords = [[[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]],
                      [[y_init1, x_init1], [y_init2, x_init2], [y_init3, x_init3]]]

            while curr_radius <= r:
                    for sector in range(0,8):
                        # Band calcaluations
                        y_init1 = coords[sector][0][0]
                        x_init1 = coords[sector][0][1]
                        y_init2 = coords[sector][1][0]
                        x_init2 = coords[sector][1][1]
                        y_init3 = coords[sector][2][0]
                        x_init3 = coords[sector][2][1]

                        # Expland scan lines
                        x_end1 = x_init1 + (1 * math.cos(math.radians(45 * sector + 0)))
                        x_end2 = x_init2 + (1 * math.cos(math.radians(45 * sector + 15)))
                        x_end3 = x_init3 + (1 * math.cos(math.radians(45 * sector + 30)))

                        y_end1 = y_init1 + (1 * math.sin(math.radians(45 * sector + 0)))
                        y_end2 = y_init2 + (1 * math.sin(math.radians(45 * sector + 15)))
                        y_end3 = y_init3 + (1 * math.sin(math.radians(45 * sector + 30)))

                        #Image bounds
                        if x_end1 >= width or x_end1 <= 0:
                            x_end1 = x_init1
                        if x_end2 >= width or x_end2 <= 0:
                            x_end2 = x_init2
                        if x_end3 >= width or x_end3 <= 0:
                            x_end3 = x_init3
                        if y_end1 >= height or y_end1 <= 0:
                            y_end1 = y_init1
                        if y_end2 >= height or y_end2 <= 0:
                            y_end2 = y_init2
                        if y_end3 >= height or y_end3 <= 0:
                            y_end3 = y_init3

                        i_init1 = image_gray[y_init1, x_init1]
                        i_init2 = image_gray[y_init2, x_init2]
                        i_init3 = image_gray[y_init3, x_init3]

                        i_end1 = image_gray[y_end1, x_end1]
                        i_end2 = image_gray[y_end2, x_end2]
                        i_end3 = image_gray[y_end3, x_end3]

                        i_diff1 = 0
                        i_diff2 = 0
                        i_diff3 = 0

                        if i_init1 != 0:
                            i_diff1 = (i_end1 - i_init1) / i_init1
                        if i_init2 != 0:
                            i_diff2 = (i_end2 - i_init2) / i_init2
                        if i_init3 != 0:
                            i_diff3 = (i_end3 - i_init3) / i_init3


                        # Is there a new band?
                        if i_diff1 <= -0.10 or i_diff2 <= -0.0 or i_diff3 <= -0.10:
                            #band_radii.append([x,y,curr_radius])
                            if i_diff1 <= -0.10:
                                intensities[sector].append(curr_radius)
                            elif i_diff2 <= -0.10:
                                intensities[sector].append(curr_radius)
                            elif i_diff3 <= -0.10:
                                intensities[sector].append(curr_radius)

                        #Store old coordinates
                        coords[sector][0] = x_end1
                        coords[sector][1] = x_end2
                        coords[sector][2] = x_end3

                        b1 = x_end1 > width
                        b2 = x_end2 > width
                        b3 = x_end3 > width
                        b4 = x_end1 < 0
                        b5 = x_end2 < 0
                        b6 = x_end3 < 0

                        b7 = y_end1 > height
                        b8 = y_end2 > height
                        b9 = y_end3 > height
                        b10 = y_end1 < 0
                        b11 = y_end2 < 0
                        b12 = y_end3 < 0
                        if(b1 or b2 or b3
                           or b4 or b5 or b6
                           or  b7 or b8 or b9
                           or b10 or b11 or b11):
                            continue

                        scan_init1 = image_gray[y_init1, x_init1]
                        scan_init2 = image_gray[y_init2, x_init2]
                        scan_init3 = image_gray[y_init3, x_init3]

                        scan_end1 = image_gray[y_end1, x_end1]
                        scan_end2 = image_gray[y_end2, x_end2]
                        scan_end3 = image_gray[y_end3, x_end3]

                        #decrease of intensity
                        if scan_init1 != 0:
                            diff1 = abs((scan_end1 - scan_init1)/scan_init1)
                        else:
                            diff1 = 0
                        if scan_init2 != 0:
                            diff2 = abs((scan_end2 - scan_init2)/scan_init2)
                        else:
                            diff2 = 0
                        if scan_init3 != 0:
                            diff3 = abs((scan_end3 - scan_init3)/scan_init3)
                        else:
                            diff3 = 0

                        #does intensity of points on scan line 1 decrease monotomicaly?
                        if diff1 <= .10: #and grow_scan1:
                            line1 = plt.plot([x_init1, x_end1], [y_init1, y_end1])
                            #plt.setp(line1, color='r', linewidth=1.0)
                        else:
                            grow_scan1 = False

                        #does intensity of points on scan line 2 decrease monotomicaly?
                        if diff2 <= .10: #and grow_scan2:
                            line2 = plt.plot([x_init2, x_end2], [y_init2, y_end2])
                            #plt.setp(line2, color='r', linewidth=1.0)
                        else:
                            grow_scan2 = False

                        #does intensity of points on scan line 3 decrease monotomicaly?
                        if diff3 <= .10: #and grow_scan3:
                            line3 = plt.plot([x_init3, x_end3], [y_init3, y_end3])
                            #plt.setp(line3, color='r', linewidth=1.0)
                        else:
                            grow_scan3 = False

                        coords[sector][0] = [y_end1, x_end1]
                        coords[sector][1] = [y_end2, x_end2]
                        coords[sector][2] = [y_end3, x_end3]


                    curr_radius = curr_radius + 1

            count = 0
            for i in intensities:
                if(len(i) > 5):
                    count = count + 1
            if count >= 4:
                Y.append(1)
            else:
                Y.append(0)
        return Y

    '''
    ###INCOMPLETE
    '''
    def testHOG(self, filename):
        image = cv2.imread(filename)
        image = rgb2gray(image)

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)


        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.savefig('RadHOG.png')
        plt.show()

        return hog_image_rescaled
        '''
    '''
    # Returns an array of arrays
    # Each array element is a blob in the image, and that
    # array element is an array of all the sums of the
    # difference between the keypoint and each pixel that
    # forms a cirlce at a certain radius

    Paramteters:
        -image: image
        -blobs: blobs from image
        -R: the number of circles to make from each keyboard
    '''

    def RadPIC(self, R, blobs, image):

        image_gray = rgb2gray(image)
        height, width = image_gray.shape[:2]
        arr = []

        for blob in blobs:
            y, x, r = blob
            sum_diff = []
            percentChange = []

            for r in range(1, R + 1):
                sum = 0
                for circum in range (0, 360):
                    x_end = x + (r * math.cos(math.radians(circum)))
                    y_end = y + (r * math.sin(math.radians(circum)))
                    if x_end > 0 and x_end < width and y_end > 0 and y_end < height:
                        sum = sum + (image_gray[y,x] - image_gray[y_end, x_end])
                sum_diff.append(sum)

            for i in range(0, len(sum_diff)-1):
                if(sum_diff[i] == 0):
                    percentChange.append(0)
                else:
                    percentChange.append((sum_diff[i+1]-sum_diff[i])/sum_diff[i])

            arr.append(percentChange)
        return arr

    '''
    Returns an array of the HOG feature descriptor for
    each fo the blobs

    Parameters:
        -image: image
        -blobs: blobs from image
    '''
    def HOG(self, hogSize, blobs, image):
        image_gray = rgb2gray(image)
        arr = []
        fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        for blob in blobs:
            y, x, r = blob
            tempImage = image_gray
            subImage = tempImage[(y-(hogSize/2)):(y+(hogSize/2)), (x-(hogSize/2)):(x+(hogSize/2))]
            fd= hog(image_gray, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualise=False)
            arr.append(fd)
        return arr

    def makeX(self, x1, x2):
        X = []
        for i in range(0, len(x1)):
            xTemp = x1[i]
            xTemp.extend(x2[i])
            X.append(xTemp)
        return X

    def trainClassifier(self):
        cols = 0

        for blobs in self.all_blobs:
            cols = cols + len(blobs)
        X = []
        Y = []
        lenRadPicX = 0
        imagesSize = len(self.images)

        for i in range(0, imagesSize):
            AIM = self.AIM(self.all_blobs[i], rgb2gray(self.images[i]))
            RadPic = self.RadPIC(5, self.all_blobs[i], rgb2gray(self.images[i]))
            inputHOG = self.HOG(5, self.all_blobs[i], rgb2gray(self.images[i]))
            print("AIM: ", AIM)
            X.extend(self.makeX(RadPic, inputHOG))
            Y = Y + AIM

        forest = RandomForestClassifier(n_estimators=100)

        print("x: ", len(X))
        print("y: ",len(Y))
        #print("x: ", X)
        print("y: ", Y)

        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(X, Y)
        with open('forest3.pkl', 'wb') as f:
            pickle.dump(forest, f)

    '''
    Uses the classifer trained in
    fruitDetection.trainClassifier in order to predict Y

    Parameters:
        -an image
    '''

    def useClassifier(self, filename):
        image = cv2.imread(filename)
        image_gray = rgb2gray(image)

        blobs = blob_doh(image_gray,
                         min_sigma=.5,
                         max_sigma=30,
                         num_sigma=15,
                         threshold=.005,
                         overlap=0.5)

        RadPic = self.RadPIC(5, blobs, image_gray)
        inputHOG = self.HOG(5, blobs, image_gray)
        X_final = self.makeX(RadPic, inputHOG)

        # Take the same decision trees and run it on the test data
        with open('forest3.pkl', 'rb') as f:
            forest = pickle.load(f)

        prediction = forest.predict(X_final)

        blobs_list = [[],blobs, []]

        titles = ['Original Image','Blob Detection', 'Random Forest Classifier']
        colors = ['black', 'orange', 'red' ]
        sequence = zip(blobs_list, colors, titles)
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        axes = axes.ravel()

        blobs2 = blobs

        for blobs, color, title in sequence:
            ax = axes[0]
            axes = axes[1:]
            ax.set_title(title)
            ax.imshow(image, interpolation='nearest')
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=.5, fill=False)
                ax.add_patch(c)
            if title == 'Random Forest Classifier':
                for i in range(1, len(blobs2)):
                    blob = blobs2[i]
                    y, x, r = blob
                    if prediction[i] == 1:
                        c = plt.Circle((x, y), r, color='red', linewidth=.5, fill=False)
                        ax.add_patch(c)

        plt.show()
        plt.savefig('output.png')

'''
-----TESTING SCRIPT BELOW-----
'''

#arr = ['frame0000.jpeg', 'frame0001.jpeg', 'frame0003.jpeg', 'frame0004.jpeg', 'frame0005.jpeg', 'frame0006.jpeg']
#apples = countFruit(arr)
#apples.trainClassifier()


applesTest = countFruit()
#applesTest.HOG(blobs, image)
applesTest.useClassifier("frame0009.jpeg")
