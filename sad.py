import cv2
from skimage.metrics import structural_similarity as ssim

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops


"""Extract signatures from an image."""
# the parameters are used to remove small size connected pixels outliar 
constant_parameter_1 = 484
constant_parameter_2 = 250
constant_parameter_3 = 100

# the parameter is used to remove big size connected pixels outliar
constant_parameter_4 = 18

path1 = 'paul'
path2 = 'D:/Zayed-Work/OPUS-ML-TEAM/signature-checker/Try Own - SSIM/assets/paul/paul2.png'

# read the input image
file = 'assets/' + path1 + '/' + path1 + str(1) + '.png'
file2 = path2

img = cv2.imread(file, 0)
img2 = cv2.imread(file2, 0)

img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

# connected component analysis by scikit-learn framework
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)

blobs2 = img2 > img2.mean()
blobs_labels2 = measure.label(blobs2, background=1)
image_label_overlay2 = label2rgb(blobs_labels2, image=img)

# fig, ax = plt.subplots(figsize=(10, 6))


# # plot the connected components (for debugging)
# ax.imshow(image_label_overlay)
# ax.set_axis_off()
# plt.tight_layout()
# plt.show()


the_biggest_component = 0
the_biggest_component2 = 0
total_area = 0
total_area2 = 0
counter = 0
counter2 = 0
average = 0.0
average2 = 0.0

for region in regionprops(blobs_labels):
    if (region.area > 10):
        total_area = total_area + region.area
        counter = counter + 1
    # print region.area # (for debugging)
    # take regions with large enough areas
    if (region.area >= 250):
        if (region.area > the_biggest_component):
            the_biggest_component = region.area

average = (total_area/counter)
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))


for region2 in regionprops(blobs_labels2):
    if (region2.area > 10):
        total_area2 = total_area2 + region2.area
        counter2 = counter2 + 1
    # print region.area # (for debugging)
    # take regions with large enough areas
    if (region2.area >= 250):
        if (region2.area > the_biggest_component2):
            the_biggest_component2 = region2.area

average2 = (total_area2/counter2)
print("the_biggest_component 2: " + str(the_biggest_component2))
print("average 2: " + str(average2))


# experimental-based ratio calculation, modify it for your cases
# a4_small_size_outliar_constant is used as a threshold value to remove connected outliar connected pixels
# are smaller than a4_small_size_outliar_constant for A4 size scanned documents
a4_small_size_outliar_constant = ((average/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

a4_small_size_outliar_constant2 = ((average2/constant_parameter_1)*constant_parameter_2)+constant_parameter_3
print("a4_small_size_outliar_constant 2: " + str(a4_small_size_outliar_constant2))

# experimental-based ratio calculation, modify it for your cases
# a4_big_size_outliar_constant is used as a threshold value to remove outliar connected pixels
# are bigger than a4_big_size_outliar_constant for A4 size scanned documents
a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

a4_big_size_outliar_constant2 = a4_small_size_outliar_constant*constant_parameter_4
print("a4_big_size_outliar_constant 2: " + str(a4_big_size_outliar_constant2))

# remove the connected pixels are smaller than a4_small_size_outliar_constant
pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
pre_version2 = morphology.remove_small_objects(blobs_labels2, a4_small_size_outliar_constant2)
# remove the connected pixels are bigger than threshold a4_big_size_outliar_constant 
# to get rid of undesired connected pixels such as table headers and etc.
component_sizes = np.bincount(pre_version.ravel())
too_small = component_sizes > (a4_big_size_outliar_constant)
too_small_mask = too_small[pre_version]
pre_version[too_small_mask] = 0

component_sizes2 = np.bincount(pre_version2.ravel())
too_small2 = component_sizes2 > (a4_big_size_outliar_constant2)
too_small_mask2 = too_small2[pre_version2]
pre_version2[too_small_mask2] = 0

# save the the pre-version which is the image is labelled with colors
# as considering connected components
plt.imsave('pre_version.png', pre_version)
plt.imsave('pre_version2.png', pre_version2)


# read the pre-version
img = cv2.imread('pre_version.png', 0)
img2 = cv2.imread('pre_version2.png', 0)
# ensure binary
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# # save the the result
# cv2.imwrite("./outputs/output.jpg", img)


# find where the signature is and make a cropped region
points = np.argwhere(img==0) # find where the black pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
crop = img[y:y+h, x:x+w] # create a cropped region of the gray image
crop = cv2.resize(crop, (600, 160))
cv2.imwrite("cropped.png", crop)


# find where the signature is and make a cropped region
points2 = np.argwhere(img2==0) # find where the black pixels are
points2 = np.fliplr(points2) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points2) # create a rectangle around those points
x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
crop2 = img2[y:y+h, x:x+w] # create a cropped region of the gray image
crop2 = cv2.resize(crop2, (600, 160))
cv2.imwrite("cropped2.png", crop)