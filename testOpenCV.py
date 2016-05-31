from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pylab
import sys
import numpy
import cv2
import pdb

'''
#Task 1
img = cv2.imread('fix.jpg',0)
equ = cv2.equalizeHist(img)
res = numpy.hstack((img,equ)) #stacking images side-by-side
cv2.imwrite('res.png',res)
image = mpimg.imread("res.png")
plt.imshow(image)
plt.show()
'''


#Task 2
img2 = cv2.imread('edge.jpeg',cv2.IMREAD_GRAYSCALE)
sigma = 0.90
#pdb.set_trace()
# compute the median of the single channel pixel intensities
v = numpy.median(img2)

# apply automatic Canny edge detection using the computed median
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edged = cv2.Canny(img2, lower, upper)

res2 = numpy.hstack((img2,edged)) #stacking images side-by-side
cv2.imwrite('res2.png', res2)
display2 = mpimg.imread("res2.png")
display2 = cv2.applyColorMap(display2, cv2.COLORMAP_BONE)
plt.imshow(display2)
plt.show()

'''
You can try a couple of things in OpenCV (python bindings) to prepare for more interesting things ahead.

For example, can you do
1) exposure/contrast adjustments (image enhancement), and
2) edge detection (feature detection).

In each of the two tasks, display the original and changed image. You should be able to do all the plotting in matplotlib. Send me or point me to the results when you have it working. Also send me your first task results (rectangle on displayed image) if you are satisfied with it.

'''