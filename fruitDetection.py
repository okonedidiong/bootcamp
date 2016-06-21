import pdb
import pylab
import sys
import cv2
import math
from math import sqrt
from scipy import ndimage
import scipy.misc
import numpy as np
from itertools import cycle
from PIL import Image, ImageDraw
# -------------------------------------------------------------------------------------------------
# Comment out the following 2 lines before compiling. I'm using a virtual environment to run OpenCV
# and Matplotlib doesn't behave very well with it
import matplotlib
matplotlib.use('TkAgg')
# -------------------------------------------------------------------------------------------------
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

class countFruit:
    fruitCount = 0 #currently unused
    images = [] #all images ot train classifer
    filenames = [] #filenames containing all images
    all_blobs = [] #blob_doh for every image

    def __init__(self, files = None):
        #print(cv2.imread("frame0001.jpeg"))
        if files == None:
            self.filenames = []
        else:
            self.filenames = files
            i = 0
            for f in self.filenames:
                self.images.append(cv2.imread(self.filenames[i], 0))
                image_gray = rgb2gray(self.images[i])
                blobs = blob_doh(image_gray, max_sigma=39, threshold=.01)

                #print("blobs_size: ", len(blobs))
                # Compute radii in the 3rd column.
                #blobs[:, 2] = blobs[:, 2] * sqrt(2)
                self.all_blobs.append(blobs)
                i = i + 1

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

    #returns and array of integers for each apple
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

        all_radii = []
        num_blobs = 0;

        for blob in blobs:
            num_blobs = num_blobs + 1;
            y, x, r = blob
            #c = plt.Circle((x, y), r, color= 'red', linewidth=2, fill=False)
            #currentAxis = plt.gca()
            #currentAxis.add_patch(Rectangle((x - r, y - r), 2*r, 2*r, fill=None, alpha=1))

            scan_init = image_gray[y,x]
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

            band_radii = []

            while curr_radius <= r:
                intens_init = image_gray[y,x]
                intens_curr = image_gray[y,x]
                intensities = [[intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init],
                               [intens_init, intens_init, intens_init]]

                for sector in range(0,8):

                    # Band calcaluations
                    i_init1 = image_gray[y_init1, x_init1]
                    i_init2 = image_gray[y_init2, x_init2]
                    i_init3 = image_gray[y_init3, x_init3]

                    i_end1 = image_gray[y_end1, x_end1]
                    i_end2 = image_gray[y_end2, x_end2]
                    i_end3 = image_gray[y_end3, x_end3]


                    i_diff1 = abs((int(i_end1) - int(intensities[sector][0])) / int(intensities[sector][0]))
                    i_diff2 = abs((int(i_end2) - int(intensities[sector][1])) / int(intensities[sector][1]))
                    i_diff3 = abs((int(i_end3) - int(intensities[sector][2])) / int(intensities[sector][2]))

                    # Is there a new band?
                    if i_diff1 >= .10:
                        band_radii.append((x_end1, y_end1, curr_radius))
                        intensities[sector][0] = image_gray[y_end1, x_end1]
                    elif i_diff2 >= .10:
                        band_radii.append((x_end2, y_end2, curr_radius))
                        intensities[sector][1] = image_gray[y_end2, x_end2]
                    elif i_diff3 >= .10:
                        band_radii.append((x_end3, y_end3, curr_radius))
                        intensities[sector][2] = image_gray[y_end3, x_end3]

                    #Expland scan lines
                    x_end1 = x_init1 + (1 * math.cos(math.radians(45 * sector + 0)))
                    x_end2 = x_init2 + (1 * math.cos(math.radians(45 * sector + 15)))
                    x_end3 = x_init3 + (1 * math.cos(math.radians(45 * sector + 30)))

                    y_end1 = y_init1 + (1 * math.sin(math.radians(45 * sector + 0)))
                    y_end2 = y_init2 + (1 * math.sin(math.radians(45 * sector + 15)))
                    y_end3 = y_init3 + (1 * math.sin(math.radians(45 * sector + 30)))

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


                    scan1 = image_gray[y_end1, x_end1]
                    scan2 = image_gray[y_end2, x_end2]
                    scan3 = image_gray[y_end3, x_end3]

                    intensities[sector] = (scan1, scan2, scan3)

                    #decrease of intensity
                    if scan1 != 0:
                        diff1 = abs((int(scan1) - int(scan_init))/int(scan1))
                    else:
                        diff1 = 0
                    if scan2 != 0:
                        diff2 = abs((int(scan2) - int(scan_init))/int(scan2))
                    else:
                        diff2 = 0
                    if scan3 != 0:
                        diff3 = abs((int(scan3) - int(scan_init))/int(scan3))
                    else:
                        diff3 = 0

                    #does intensity of points on scan line 1 decrease monotomicaly?
                    if diff1 <= .10: #and grow_scan1:
                        line1 = plt.plot([x_init1, x_end1], [y_init1, y_end1])
                        plt.setp(line1, color='r', linewidth=1.0)
                    else:
                        grow_scan1 = False

                    #does intensity of points on scan line 2 decrease monotomicaly?
                    if diff2 <= .10: #and grow_scan2:
                        line2 = plt.plot([x_init2, x_end2], [y_init2, y_end2])
                        plt.setp(line2, color='r', linewidth=1.0)
                    else:
                        grow_scan2 = False

                    #does intensity of points on scan line 3 decrease monotomicaly?
                    if diff3 <= .10: #and grow_scan3:
                        line3 = plt.plot([x_init3, x_end3], [y_init3, y_end3])
                        plt.setp(line3, color='r', linewidth=1.0)
                    else:
                        grow_scan3 = False

                    x_init1 = x_end1
                    x_init2 = x_end2
                    x_init3 = x_end3
                    y_init1 = y_end1
                    y_init2 = y_end2
                    y_init3 = y_end3

                curr_radius = curr_radius + 1
            all_radii.append(band_radii)
        return all_radii

    '''
    ###INCOMPLETE
    '''
    def testHOG(self, filename):
        image = cv2.imread(filename)
        image = rgb2gray(image)

        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        for i in hog_image:
            for j in i:
                if j != 0:
                    print("j: ", j)
                    print("YAY!")

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
            for r in range(1, R + 1):
                sum = 0
                for circum in range (0, 360):
                    x_end = x + (r * math.cos(math.radians(circum)))
                    y_end = y + (r * math.sin(math.radians(circum)))
                    if x_end > 0 and x_end < width and y_end > 0 and y_end < height:
                        sum = sum + (image_gray[y,x] - image_gray[y_end, x_end])
                sum_diff.append(sum)
            arr.append(sum_diff)
        return arr



    '''
    ###INCOMPLETE###

    Returns an array of the HOG feature descriptor for
    each fo the blobs

    Parameters:
        -image: image
        -blobs: blobs from image
    '''
    def HOG(self, blobs, image):
        #image = cv2.imread(filename)
        image_gray = rgb2gray(image)
        arr = []

        #currentAxis = plt.gca()

        for blob in blobs:
            y, x, r = blob

            tempImage = image_gray
            degree = math.atan2(y,x)

            y1 = y - r*math.sin(math.radians(135))
            y2 = y + r*math.sin(math.radians(135))
            x1 = x + r*math.cos(math.radians(135))
            x2 = x - r*math.cos(math.radians(135))

            subImage = image_gray[y1:(y1+2*r), x1:(x1+2*r)]
            #height, width = subImage.shape[:2]
            #h = hog.compute(image)
            #arr.append(h)

            linArr = hog(subImage, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualise=False)

            #print("linArr: ", linArr)

            temp = []
            #for i in blob_hog:
            #    for j in i:
            #        #print("len(j):", len(j))
            #        print("j:", j)
            #        print("_______________________________")
                    #print("j: " j)
                    #temp.append(j)

            #print(blob_hog)
            #print("------------------------")

            #currentAxis.add_patch(Rectangle((x1, y1), 2*r, 2*r, color='red', fill=False))
            arr.append(linArr)

        #plt.imshow(image)
        #plt.show()

        return arr

    def trainClassifier(self):
        cols = 0
        for blobs in self.all_blobs:
            cols = cols + len(blobs)
        X = []
        X_final = []
        Y_final = []
        lenRadPicX = 0
        maxHogSize = 0
        imagesSize = len(self.images)

        for i in range(0, imagesSize):
            #AIM = self.AIM(self.all_blobs[i], rgb2gray(self.images[i]))
            #print("AIM: ", AIM)
            RadPic = self.RadPIC(6, self.all_blobs[i], rgb2gray(self.images[i]))
            inputHOG = self.HOG(self.all_blobs[i], rgb2gray(self.images[i]))


            if len(inputHOG) > maxHogSize:
                maxHogSize = len(inputHOG)

            for j in range(0, len(RadPic)):
                #print("RadPic: ", RadPic[j])
                #print("inputHOG: ", inputHOG[j])
                #print("--------")
                lenRadPicX = len(RadPic[j])
                X.append(np.append(RadPic[j], inputHOG[j]))
                #X.append(RadPic[j])

            for k in range(0, len(RadPic)):
                count = 0
                for z in range(0, (len(RadPic[k])-1)):
                    if RadPic[k][z] < RadPic[k][z+1]:
                        count = count + 1
                if count > 2:
                    Y_final.append(1)
                else:
                    Y_final.append(0)

        # Make the RadPIC part of X contain the percent change
        # between different RadPIC sums
        for i in range(0, len(X)):
            temp1 = []
            X_final.append(temp1)
            for j in range(0, lenRadPicX-1):
                if (X[i][j] == 0):
                    temp1.append(0)
                else:
                    temp1.append((X[i][j+1] - X[i][j]) / X[i][j])
            X[i][lenRadPicX-1] = 0

        xLen = lenRadPicX + maxHogSize

        # Make length of all X's the same
        for i in range(0, len(X_final)):
            while len(X_final[i]) < xLen:
                X_final[i].append(0)

        #Append RadHOG to X_final
        for i in range(0, len(X)):
            for j in range(lenRadPicX, X[i]):
                X_final[i][j] = X[i][j]



        #print("X_final length: ", len(X_final))
        #print("Y_final length: ", len(Y_final))

        forest = RandomForestClassifier(n_estimators=100)

        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(X_final, Y_final)
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
        lenRadPicX = 0
        maxHogSize = 0

        #blobs = blob_doh(image_gray, max_sigma=30, threshold=.01)
        blobs = blob_doh(image_gray,
                         min_sigma=.5,
                         max_sigma=30,
                         num_sigma=15,
                         threshold=.005,
                         overlap=0.5)


        RadPic = self.RadPIC(6, blobs, image_gray)
        inputHOG = self.HOG(blobs, image_gray)

        X = []
        X_final = []


        if len(inputHOG) > maxHogSize:
            maxHogSize = len(inputHOG)

        for j in range(0, len(RadPic)):
            lenRadPicX = len(RadPic[j])
            X.append(np.append(RadPic[j], inputHOG[j]))

        # Make the RadPIC part of X contain the percent change
        # between different RadPIC sums
        for i in range(0, len(X)):
            temp1 = []
            X_final.append(temp1)
            for j in range(0, lenRadPicX - 1):
                if (X[i][j] == 0):
                    temp1.append(0)
                else:
                    temp1.append((X[i][j + 1] - X[i][j]) / X[i][j])
            X[i][lenRadPicX - 1] = 0

        xLen = lenRadPicX + maxHogSize

        # Make length of all X's the same
        for i in range(0, len(X_final)):
            while len(X_final[i]) < xLen:
                X_final[i].append(0)

        # Append RadHOG to X_final
        for i in range(0, len(X)):
            for j in range(lenRadPicX, X[i]):
                X_final[i][j] = X[i][j]

        # Take the same decision trees and run it on the test data
        with open('forest3.pkl', 'rb') as f:
            forest = pickle.load(f)

        print("X_final: ", X_final)


        prediction = forest.predict(X_final)

        #print("X_final: ", X_final)


        print(prediction)
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



arr = ['frame0000.jpeg', 'frame0001.jpeg', 'frame0002.jpeg', 'frame0003.jpeg', 'frame0004.jpeg']
#arr = ['frame0001.jpeg']
apples = countFruit(arr)
apples.trainClassifier()

'''
image = cv2.imread("frame0001.jpeg")

image_gray= rgb2gray(image)

blobs = blob_doh(image_gray,
                 min_sigma=.5,
                 max_sigma=30,
                 num_sigma=15,
                 threshold=.005,
                 overlap=0.5)
'''
#applesTest = countFruit()
#applesTest.HOG(blobs, image)
#applesTest.useClassifier("frame0009.jpeg")
