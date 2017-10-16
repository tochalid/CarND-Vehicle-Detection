import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import glob
import datetime
import pickle

from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from mySolution.VD_functions import *
from mySolution.VD_DetectorClass import Detector

# Just switches ot generate the pictures needed for the project documentation
GLOBAL_VIDEO = True
GLOBAL_SHOW = False
GLOBAL_SAVE = False
SCALE_SAVE = False
TEST_CAR = False

# Directory path where the pictures for the docu are stored for
PATH_TESTING = '../output_images/testing/'

# Load the classifier and scaler using pickle
with open('../model/scv_pickle.p', 'rb') as fh:
    svc, X_scaler, \
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat, \
    window_scale, x_start_stop, y_start_stop, xy_window, xy_overlap, color_values, \
    threshold, smoothing_limit = pickle.load(fh)

# Set cv2. config color-transformation
if color_space != 'RGB':
    if color_space == 'LUV':
        color_converter_code = cv2.COLOR_RGB2LUV
    elif color_space == 'HSV':
        color_converter_code = cv2.COLOR_RGB2HSV
    elif color_space == 'HLS':
        color_converter_code = cv2.COLOR_RGB2HLS
    elif color_space == 'YUV':
        color_converter_code = cv2.COLOR_RGB2YUV
    elif color_space == 'YCrCb':
        color_converter_code = cv2.COLOR_RGB2YCrCb
    print('Current color space', color_space, 'has been converted with cv2.COLOR_RGB2...=', color_converter_code)
else:
    print('Check color space conversion, current: ', color_space)

# Set timestamp for file saving
d = datetime.datetime.now().strftime('%Y-%m-%d_w%W_%H:%M:%S.%f')

test_images = []
if TEST_CAR:
    notcars = glob.glob('../data_set/non-vehicles/**/*.png', recursive=True)
    cars = glob.glob('../data_set/vehicles/**/*.png', recursive=True)

    test_image1 = cv2.imread(cars[999])  # select any file for the docu >999
    test_image1 = bgr2rgb(test_image1)
    test_image2 = cv2.imread(notcars[999])  # select any file for the docu >999
    test_image2 = bgr2rgb(test_image2)
    test_images.append(test_image1)
    test_images.append(test_image2)
    test_images = np.asarray(test_images)

else:
    ### Read in all the test images
    test_files = glob.glob('../test_images/*.jpg')
    # Use cv2.imread() to read files so that all files are scaled from 0-255
    for file in test_files:
        test_image = cv2.imread(file)
        test_image = bgr2rgb(test_image)
        test_images.append(test_image)

    test_images = np.asarray(test_images)
print("Test images shape is:", test_images.shape)

nrows, ncols = test_images.shape[0]//2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
for pos, image in enumerate(test_images):
    plt.subplot(nrows, ncols, pos + 1)
    plt.imshow(test_images[pos])
    plt.title("Test Image {:d} - Original".format(pos + 1))
fig.tight_layout()
if GLOBAL_SAVE: plt.savefig(PATH_TESTING + 'test_images_' + d + '.png')
if GLOBAL_SHOW: plt.show()
plt.close('all')

for test_image in test_images:
    img = cv2.cvtColor(test_image, color_converter_code)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(test_image)
    plt.title('Original')
    pos = 2
    for channel in range(img.shape[2]):
        hog_channel_image = img[:, :, channel]
        plt.subplot(1, 4, pos)
        plt.imshow(hog_channel_image, cmap='gray')
        plt.title("Color Channel {:d}".format(pos - 1))
        pos += 1
    fig.tight_layout()
    if GLOBAL_SAVE: plt.savefig(
        PATH_TESTING + 'color/color_cha_' + datetime.datetime.now().strftime('%Y-%m-%d_w%W_%H:%M:%S.%f') + '.png')
    if GLOBAL_SHOW: plt.show()
    plt.close('all')

window_scale = (0.73, 1.5) #0.73, 1.50
x_start_stop = [[640, 1280], [640, 1280]]
y_start_stop = [[405, 570], [405, 610]]
xy_window = (128, 128)
xy_overlap = (0.95, 0.95)
color_values = [(255, 255, 255), (130, 130, 130)]

if not TEST_CAR:  # detection not computed for data_set training images 64x64x3, only 720x1280x3 from video or test_images

    plt.figure(figsize=(10, 10))
    for idx, image in enumerate(test_images):
        n_windows = 0
        for i, scale in enumerate(window_scale):
            windows = slide_window(image, x_start_stop=x_start_stop[i], y_start_stop=y_start_stop[i],
                                   xy_window=[int(dim * window_scale[i]) for dim in xy_window], xy_overlap=xy_overlap)
            image = draw_boxes(image, windows, color_values[i])
            n_windows += len(windows)
        plt.subplot(3, 2, idx + 1)
        plt.imshow(image)
        plt.title("Test Image {:d}".format(idx + 1))
    fig.tight_layout()
    print("Total # of windows:", n_windows)
    if SCALE_SAVE: plt.savefig(PATH_TESTING + 'scaled_windows_' + d + '.png')
    if GLOBAL_SHOW: plt.show()
    plt.close('all')

    # Set window scales of HOG subsampling
    windim = 128
    window_scale_HOGsubsampling = (
        (window_scale[0] * windim / 64),
        (window_scale[1] * windim / 64))  # depends on number of detection layers, here=2

    detector = Detector(svc, X_scaler)
    detector.window_scale = window_scale_HOGsubsampling
    detector.x_start_stop = x_start_stop
    detector.y_start_stop = y_start_stop
    detector.orient = orient
    detector.pix_per_cell = pix_per_cell
    detector.cell_per_block = cell_per_block
    detector.spatial_size = spatial_size
    detector.hist_bins = hist_bins
    detector.threshold = threshold = 1  # 1
    detector.smoothing_limit = smoothing_limit = 15  # 15 # use 1 for project test images
    detector.color_space = color_space

    total_time = 0
    output_images = []
    for idx, image in enumerate(test_images):
        t = time.time()
        output_image = detector.vehicle_detection(image, DIR=PATH_TESTING)
        output_images.append(output_image)
        t2 = time.time()
        total_time += round(t2 - t, 2)

    plt.figure(figsize=(10, 10))
    for idx, image in enumerate(output_images):
        plt.subplot(3, 2, idx + 1)
        plt.imshow(image)
        plt.title("Test Image {:d}".format(idx + 1))
    fig.tight_layout()
    print("Average Image processing time: {:.3f} seconds".format(total_time / (idx + 1)))
    if GLOBAL_SAVE: plt.savefig(PATH_TESTING + 'detected_vehicles_' + d + '.png')
    if GLOBAL_SHOW: plt.show()
    plt.close('all')

for idx in range(0, len(test_images)):
    img = cv2.cvtColor(test_images[idx], color_converter_code)
    fig = plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)

    if TEST_CAR:
        plt.imshow(test_images[idx])
        plt.title('Original - Data Set')
    else:
        plt.imshow(output_images[idx])
        plt.title('Labeled Car')
    pos = 2
    for channel in range(img.shape[2]):
        hf, hog_channel_image = get_hog_features(img[:, :, channel], orient, pix_per_cell, cell_per_block, vis=True,
                                                 feature_vec=False)
        plt.subplot(1, 4, pos)
        plt.imshow(hog_channel_image, cmap='gray')
        plt.title("HOG Image - Ch{:d}".format(pos - 1))
        pos += 1
    fig.tight_layout()
    if GLOBAL_SAVE: plt.savefig(
        PATH_TESTING + 'hogcha/hog_cha_' + datetime.datetime.now().strftime('%Y-%m-%d_w%W_%H:%M:%S.%f') + '.png')
    if GLOBAL_SHOW: plt.show()
    plt.close('all')

# Run the detector on the project video and save the video output file
if GLOBAL_VIDEO:
    project_output = '../output_videos/project_video_' + d + '.mp4'
    # clip1 = VideoFileClip("../project_video.mp4").subclip(25, 26)
    # clip1 = VideoFileClip("../project_video.mp4").subclip(20, 40)
    clip1 = VideoFileClip("../project_video.mp4")

    clip = clip1.fl_image(detector.vehicle_detection)
    clip.write_videofile(project_output, audio=False)
    clip.reader.close()
