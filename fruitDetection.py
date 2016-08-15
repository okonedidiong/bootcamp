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
import PIL
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from scipy.misc import imresize
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh, hog
from skimage.color import rgb2gray
from skimage import data, color, exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn import svm, datasets
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import pickle
import glob
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
    croppedImages = []
    full_images = []
    full_image_labels = []
    labels = []


    def __init__(self, files = None):
        #self.croppedImages = glob.glob("cropped_images/frame*")
        #self.croppedImages = glob.glob("c")
        #self.croppedImages = sorted(self.croppedImages, key=self.sortKey)
        #self.labels = glob.glob("apple/*frame*")
        #self.labels = sorted(self.labels, key=self.sortKey)

        self.full_images = glob.glob("full_images/frame*")
        self.full_images = sorted(self.full_images, key=self.sortKey2)
        self.full_image_labels = glob.glob("full_image_labels/frame*")
        self.full_image_labels = sorted(self.full_image_labels, key=self.sortKey2)

        #print("full_images: ", self.full_images)
        #print("full_image_labels: ", self.full_image_labels)

        '''
        for x in range(0, len(testVal)):
            for y in range(0, len(testVal[x])):
                for z in range(0, len(testVal[x][y])):
                    if testVal[x][y][z] != 0:
                        print("z: ", testVal[x][y][z])
        '''

        #print("test image: ", cv2.imread(self.full_image_labels[1]))

        if files == None:
            self.filenames = []
            '''
            for i in range(0, 1000):
                image = cv2.imread((glob.glob("cropped_images/*" + self.filenames[i]))[0])
                self.images.append(image)
                image_gray = rgb2gray(image)
                file_processing = "Processing files: " + str(i + 1) + "/" + str(len(self.filenames))
                print(file_processing)
            '''
        else:
            self.filenames = files

            for i in range(0, len(self.filenames)):
                #image = cv2.imread((glob.glob("cropped_images/*" + self.filenames[i]))[0])
                image = cv2.imread((glob.glob("full_images/*" + self.filenames[i]))[0])
                self.images.append(image)
                image_gray = rgb2gray(image)
                #blobs = blob_doh(image_gray, min_sigma=1, max_sigma=25, num_sigma=15, threshold=.001)
                blobs = blob_dog(image_gray, min_sigma=.5, max_sigma=25, sigma_ratio=1.6, threshold=.25, overlap=0.5)
                if(len(blobs) != 0):
                    blobs[:, 2] = blobs[:, 2] * sqrt(2)

                self.all_blobs.append(blobs)
                file_processing = "Processing files: " + str(i+1) + "/" + str(len(self.filenames))
                #print("blobs size: ", len(blobs))
                print(file_processing)

    def sortKey(self, str):
        i = str.find("frame")
        j = str.find(".")
        return int(str[i+5:j])

    def sortKey2(self, str):
        i = str.find("frame")
        return int(str[i+5:i+9])

    def sortKey3(self, str):
        i = str.find("frame")
        return int(str[i + 5:i + 9])

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
        height, width = image_gray.shape[:2]

        blobs_log = blob_log(image_gray, min_sigma=3, max_sigma=30, num_sigma=10, threshold=.1)

        # Compute radii in the 3rd column.
        blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

        blobs_dog = blob_dog(image_gray, min_sigma=3, max_sigma=30, threshold=.1)
        blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

        blobs_doh = blob_doh(image_gray, min_sigma=3, max_sigma=30, threshold=.01)

        blobs_list = [[], blobs_log, blobs_dog, blobs_doh]
        colors = ['black', 'red', 'red', 'red']
        titles = ['Original Image', 'Laplacian of Gaussian','Difference of Gaussian', 'Determinant of Hessian']
        sequence = zip(blobs_list, colors, titles)

        fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        axes = axes.ravel()

        for blobs, color, title in sequence:
            ax = axes[0]
            axes = axes[1:]
            ax.set_title(title)
            #ax.imshow(image, interpolation='nearest')     #print("r: ", r)
            ax.imshow(cv2.imread(filename))
            #print("________")
            #c = plt.Circle((x, y), r, color= 'red', linewidth=2, fill=False)
            #currentAxis = plt.gca()
            #currentAxis.add_patch(Rectangle((x - r, y - r), 2*r, 2*r, fill=None, alpha=1))
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
                ax.add_patch(c)

        #plt.savefig('output.png')
        plt.show()

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
        height, width = image_gray.shape[:2]
        basewidth = 10
        arr = []
        RadHOG = []

        fd, hog_image = hog(image_gray, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(3, 3), visualise=True)

        # Example of HOG for paper

        for blob in blobs:
            #print("-----------")
            y, x, r = blob

            if x == float(0):
                x = x + 1
            if y == float(0):
                y = y + 1

            scaleR = r
            tempImage = image_gray

            #subImage = tempImage[(y-(hogSize/2)):(y+(hogSize/2)), (x-(hogSize/2)):(x+(hogSize/2))]
            y1 = y-r
            y2 = y+r
            x1 = x-r
            x2 = x+r

            if x1 <= 0:
                while x - scaleR <= 0:
                    scaleR = scaleR - 1
            if x2 <= 0:
                while x + scaleR <= 0:
                    scaleR = scaleR - 1
            if y1 <= 0:
                while y - scaleR <= 0:
                    scaleR = scaleR - 1
            if y2 <= 0:
                while y + scaleR <= 0:
                    scaleR = scaleR - 1
            if x1 >= width:
                while x - scaleR >= width:
                    scaleR = scaleR-1
            if x2 >= width:
                while x + scaleR >= width:
                    scaleR = scaleR-1
            if y1 >= height:
                while y - scaleR >= height:
                    scaleR = scaleR-1
            if y2 >= height:
                while y + scaleR >= height:
                    scaleR = scaleR-1

            subImage = tempImage[(y-(scaleR)):(y+(scaleR)), (x-(scaleR)):(x+(scaleR))]
            #print("scaleR: ", scaleR)
            #print("-----------")
            #plt.imshow(subImage)
            #plt.show()

            #print("subImage: ", subImage)


            testVal = subImage
            subImage = imresize(subImage, (30,30))

            fd, hog_image = hog(subImage, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(3, 3), visualise=True)

            angle = 0
            blob_feature = []

            while angle < 360:
                for radius in range(0, 15):
                    blob_feature.append(hog_image[15 + radius*math.sin(math.radians(angle))][15 + radius*math.cos(math.radians(angle))])
                angle = angle + 1

            arr.append(blob_feature)

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

        for i in range(0, 6):
            #AIM = self.AIM(self.all_blobs[i], rgb2gray(self.images[i]))
            RadPic = self.RadPIC(10, self.all_blobs[i], rgb2gray(self.images[i]))
            inputHOG = self.HOG(10, self.all_blobs[i], rgb2gray(self.images[i]))
            X.extend(self.makeX(RadPic, inputHOG))
            train_status = "Training Status: " + str(i+1) + "/" + str(len(self.images))
            print(train_status)

        zeroCount = 0
        oneCount = 0

        for i in range(0, 6):
            j = i
            tempStr = self.full_image_labels[j]

            '''
            while tempStr.find("frame" + str(i)) == -1:
                j = j + 1
                tempStr = self.full_image_labels[j]
            '''

            label_image = cv2.imread((glob.glob(tempStr))[0])
            blobs = self.all_blobs[i]

            for blob in blobs:
                y,x,r = blob
                if label_image[y,x][0] == 255 & label_image[y,x][1] == 255 & label_image[y,x][2] == 255:
                    Y.append(1)
                    oneCount = oneCount + 1
                else:
                    Y.append(0)
                    zeroCount = zeroCount + 1

        print("oneCount: ", oneCount)
        print("zeroCount: ", zeroCount)
        print("Y: ", Y)

        forest = RandomForestClassifier(n_estimators=100)

        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(X, Y)
        # with open('26may2016.pkl', 'wb') as f: #paramters for radpic and hog were 5 and 5
        #    pickle.dump(forest, f)
        with open('test.pkl', 'wb') as f:
            pickle.dump(forest, f)


    '''
    Uses the classifer trained in
    fruitDetection.trainClassifier in order to predict Y

    Parameters:
        -an image
    '''

    def useClassifier(self, filename):
        # this is a comment
        probabilities = []
        prediction = []
        image = cv2.imread(filename)
        image_gray = rgb2gray(image)

        #blobs = blob_doh(image_gray, min_sigma=1, max_sigma=25, num_sigma=15, threshold=.001)
        blobs = blob_dog(image_gray, min_sigma=1, max_sigma=25, sigma_ratio=1.6, threshold=.25, overlap=0.5)
        if (len(blobs) != 0):
            blobs[:, 2] = blobs[:, 2] * sqrt(2)

        RadPic = self.RadPIC(10, blobs, image_gray)
        inputHOG = self.HOG(10, blobs, image_gray)
        X_final = self.makeX(RadPic, inputHOG)

        # Take the same decision trees and run it on the test data
        #with open('croppedImagesV2.pkl', 'rb') as f:
        #    forest = pickle.load(f)
        with open('test.pkl', 'rb') as f:
            forest = pickle.load(f)

        probabilities = forest.predict_proba(X_final)
        print(probabilities)
        #print("Predict probability: ", )
        #prediction = forest.predict(X_final)
        for p in probabilities:
            if p[1] >= .3:
                prediction.append(1)
            else:
                prediction.append(0)

        #print("Prediction: ", prediction)

        blobs_list = [[], blobs, [], []]

        titles = ['Original Image','Blob Detection', 'Ground Truth Labels', 'Random Forest Classifier']
        colors = ['black', 'orange', 'blue', 'red']
        sequence = zip(blobs_list, colors, titles)
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
        fig.subplots_adjust(wspace=.05, hspace=0.001)
        plt.axis('off')
        axes = axes.ravel()

        blobs2 = blobs

        for blobs, color, title in sequence:
            ax = axes[0]
            axes = axes[1:]
            ax.set_title(title)
            if title != 'Ground Truth Labels':
                ax.imshow(image, interpolation='nearest')
            else:
                #frame = filename[filename.find("frame"): len(filename)]
                frame = filename[filename.find("frame"): filename.find("frame") + 9]
                print("frame: ", frame)
                ax.imshow(cv2.imread(glob.glob("full_image_labels/*" + frame + "_pos.png")[0]), interpolation='nearest')
                #ax.imshow(cv2.imread(glob.glob("full_images/*" + frame + ".png")[0]), interpolation = 'nearest')
                #print(glob.glob("apple/*" + frame)[0] + ".png")
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color=color, linewidth=1, fill=False)
                ax.add_patch(c)
            if title == 'Random Forest Classifier':
                for i in range(len(blobs2)):
                    blob = blobs2[i]
                    y, x, r = blob
                    if prediction[i] == 1: #IMPORTANT LINE#
                        c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
                        ax.add_patch(c)

        plt.show()
        plt.savefig('output.png')

    def precisionRecall(self):
        thresholds = []
        test_images = []
        label_images = []
        test_blobs = []
        X = []
        Y_ground = []
        Y_classifier = []
        Y_ground_pixel = []
        Y_classifier_pixel = []

        # Get blobs
        for i in range(6,11):
            #file = self.croppedImages[i]
            file = self.full_images[i]
            iter_image = rgb2gray(cv2.imread(file))

            iter_blobs = blob_dog(iter_image, min_sigma=1, max_sigma=25,
                                  sigma_ratio=1.6, threshold=.25, overlap=0.5)
            test_images.append(iter_image)
            test_blobs.append(iter_blobs)
            print("Get blobs (analysis): " + str(i) + "/1000")

        # Get X values for testing
        for i in range(0, len(test_images)):
            RadPic = self.RadPIC(10, test_blobs[i], rgb2gray(test_images[i]))
            inputHOG = self.HOG(10, test_blobs[i], rgb2gray(test_images[i]))
            X.extend(self.makeX(RadPic, inputHOG))
            print("X (analysis): " + str(i) + "/1000")


        #Get Y ground truth values
        for i in range(6,11):
            j = i
            #tempStr = self.labels[j]
            tempStr = self.full_image_labels[j]
            '''
            while tempStr.find("frame" + str(i)) == -1:
                j = j + 1
                tempStr = self.labels[j]
            '''

            label_image = cv2.imread((glob.glob(tempStr))[0])
            label_images.append(label_image)
            #cropped_image = cv2.imread((glob.glob("cropped_images/frame" + str(i) + ".jpg"))[0])
            #cropped_image = rgb2gray(cropped_image)

        pixel_count = []

        for i in range(0, len(test_blobs)):
            for blob in test_blobs[i]:
                y, x, r = blob
                if label_images[i][y, x][0] == 255 & label_images[i][y, x][1] == 255 & label_images[i][y, x][2] == 255:
                    Y_ground.append(1)
                else:
                    Y_ground.append(0)
                count = 0;
                for iter_r in range(0, int(r)):
                    for iter_c in range(0, 360):
                        iter_x = x + iter_r*math.cos(math.radians(iter_c))
                        iter_y = y + iter_r*math.sin(math.radians(iter_c))
                        height, width = label_images[i].shape[:2]
                        if iter_x > 0 and iter_x < width and iter_y > 0 and iter_y < height:
                            count = count + 1
                            if label_images[i][iter_y, iter_x][0] == 255 & label_images[i][iter_y, iter_x][1] == 255 & label_images[i][iter_y, iter_x][2] == 255:
                                Y_ground_pixel.append(1)
                            else:
                                Y_ground_pixel.append(0)
                pixel_count.append(count)


        # Open classifier
        with open('test.pkl', 'rb') as f:
            forest = pickle.load(f)

        # Precidicions
        Y_probabilities = forest.predict_proba(X)
        for i in range(0, len(Y_probabilities)):
            Y_classifier.append(Y_probabilities[i][1])

        for i in range(0, len(Y_probabilities)):
            for j in range(0, pixel_count[i]):
                Y_classifier_pixel.append(Y_classifier[i])


        # Precision-Recall (per keypoint)
        precision, recall, thresholds = precision_recall_curve(Y_ground, Y_classifier)

        # ROC (per keypoint)
        fpr, tpr, thresholds = roc_curve(Y_ground, Y_classifier, pos_label=1)

        # Precision-Recall (per pixel)
        precision_pixel, recall_pixel, thresholds_pixel = precision_recall_curve(Y_ground_pixel, Y_classifier_pixel)

        # ROC (per pixel)
        fpr_pixel, tpr_pixel, thresholds_pixel = roc_curve(Y_ground_pixel, Y_classifier_pixel, pos_label=1)

        #Precision-Recall Curve (per keypoint)
        plt.subplot(221)
        plt.plot(recall, precision, 'ro')
        plt.title('Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        #ROC Curve (per keypoint)
        plt.subplot(222)
        plt.plot(fpr, tpr, 'ro')
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        '''
        #Precision-Recall Curve (per pixel)
        plt.subplot(223)
        plt.plot(recall_pixel, precision_pixel, 'ro')
        plt.title('Precision-Recall Curve (Per Pixel)')
        plt.ylabel('Precision')
        plt.xlabel('Recall')

        #ROC Curve (per pixel)
        plt.subplot(224)
        plt.plot(fpr_pixel, tpr_pixel, 'ro')
        plt.title('ROC Curve (Per Pixel)')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        '''
        plt.show()

files_train = ["frame0000.jpg",
         "frame0001.jpg",
         "frame0002.jpg",
         "frame0003.jpg",
         "frame0004.jpg",
         "frame0005.jpg",]

files_test= ["frame0006.jpg",
         "frame0007.jpg",
         "frame0008.jpg",
         "frame0009.jpg",
         "frame0010.jpg"]


#for i in range(0, 6):
    #file = "frame" + str(i) + ".jpg"
    #file = "frame" + str(i)
    #files.append(file)



#apples = countFruit(arr, labels)

#test1 = countFruit(files_train)
#test1.trainClassifier()
#precall = countFruit(files_test)
#precall.precisionRecall()


#testUse = countFruit()
#testUse.useClassifier(testUse.full_images[8])

#getHOG = countFruit(['frame0009.jpg'])
#getHOG.circles('apple/P2020L90CappleIframe299.jpg.png')
#getHOG.circles('cropped_images/frame9.jpg')
#getHOG.useClassifier(getHOG.full_images[0])


fd2, hog_image2 = hog(rgb2gray(cv2.imread('cropped_images/frame184.jpg')), orientations=8, pixels_per_cell=(6, 6), cells_per_block=(3, 3), visualise=True)

plt.subplot(221)
plt.imshow(cv2.imread('cropped_images/frame184.jpg'))
plt.title('Original Image')

plt.subplot(222)
plt.imshow(hog_image2)
plt.title('Histogram of Oriented Gradients')

#plt.imshow()
plt.show()
