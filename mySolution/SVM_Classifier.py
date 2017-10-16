import glob
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mySolution.VD_functions import *

### Build and train the Classifier
# process the data

### Read in all data
notcars = glob.glob('../data_set/non-vehicles/**/*.png', recursive=True)
cars = glob.glob('../data_set/vehicles/**/*.png', recursive=True)
test_image = cv2.imread(cars[0])
test_image = bgr2rgb(test_image)

print("Image shape:", test_image.shape)
print("# of car images:", len(cars))
print("# of notcar images:", len(notcars))
print("# of images in total:", len(cars) + len(notcars))

### Define the variables
# Variables for feature extraction feeding SVM
color_space = 'LUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb # check function VD_functions.
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off

# Variables for sliding windows in vehicle detection
window_scale = None
x_start_stop = None
y_start_stop = None
xy_window = None
xy_overlap = None
color_values = None
threshold = None  # detection threshold, corresponds with hot-count required for a pixel to be labeled as car pixel
smoothing_limit = None  # Number of video frames adding heat before applying threshold

### Convert the images to feature vectors and normalize the feature vectors
t = time.time()
car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Segmentation and shuffling into training and test sets
# Randomize the sets
randomized = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=randomized)
# Balance class members in the splits
train_cars_perc = len(y_train[y_train == 1]) / len(y_train) * 100
train_notcars_perc = 100 - train_cars_perc
test_cars_perc = len(y_test[y_test == 1]) / len(y_test) * 100
test_notcars_perc = 100 - test_cars_perc

print("Length feature vector:", len(X_train[0]))
print("Training split: ", format(y_train.shape[0]))
print("Testing split: ", format(y_test.shape[0]))
print("% of cars/notcars in training split {:.2f}% / {:.2f}%:".format(train_cars_perc, train_notcars_perc))
print("% of cars/notcars in test split {:.2f}% / {:.2f}%:".format(test_cars_perc, test_notcars_perc))

### Initalize & train the classifier
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

model = '../model/scv_pickle.p'
with open(model, 'wb') as fh:
    pickle.dump([svc, X_scaler,
                 color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat,
                 hist_feat, hog_feat,
                 window_scale, x_start_stop, y_start_stop, xy_window, xy_overlap, color_values,
                 threshold, smoothing_limit], fh)
# Todo: change pickle input to dictionary with key-value pair for each parameter
print('Saved model in location ', model)
