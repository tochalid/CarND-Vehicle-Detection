## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**Vehicle Detection Project**

Steps of this project:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Additionally, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Normalize the features and randomize the image selection for training and testing of the classifier.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (project_video.mp4)
* Create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./examples/train_images_RGB.png
[image11]: ./examples/color_cha_1_LUV.png
[image12]: ./examples/color_cha_2_LUV.png
[image13]: ./examples/hog_cha_1_LUV.png
[image14]: ./examples/hog_cha_2_LUV.png
[image21]: ./examples/color_cha_1_HSV.png
[image22]: ./examples/color_cha_2_HSV.png
[image23]: ./examples/hog_cha_1_HSV.png
[image24]: ./examples/hog_cha_2_HSV.png
[image3]: ./examples/scaled_windows.png
[image4]: ./examples/detected_vehicles.png
[image51]: ./examples/llb_h_compare_1.png
[image52]: ./examples/llb_h_compare_2.png
[image53]: ./examples/llb_h_compare_3.png
[image54]: ./examples/llb_h_compare_4.png
[image55]: ./examples/llb_h_compare_5.png
[image56]: ./examples/llb_h_compare_6.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

In the following I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Provide a README that includes all the rubric points
This document, you're reading it.


##Histogram of Oriented Gradients (HOG)

###1. Extracting HOG features from the training images

The code for this step is contained in function `get_hog_features` in lines #239-258 of the file called `VD_functions.py`. The function is called in line #50,56 for car and notcars by the `extract_features` function in file `SVM_Classifier.py`. That function is defined also in `VD_functions.py`, see line #87-138.

I started by reading in all the `vehicle` and `non-vehicle` images, see #14-18 in `SVM_Classifier.py`. 

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `LUV` and `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a 'car' and a 'notcar' class:

####a. LUV Color Space
![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

####b. HSV Color Space

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]

###2. The final choice of HOG parameters

I tried many combinations of parameters and a guiding trade-off remained computing time vs. accuracy. For the color space i got better result for 'HSV'.
About half of the "Average Image Processing" time goes to hog feature extraction. Thus, number of channels (the less the shorter) and pixel-per-cell (the more the shorter) have the most impact. But less channels and more pixels both cause decrease in accuracy detecting hot pixels. Here I chose 3 channels and values of 32 for spacial and histogram parameters:

`color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb`

`orient = 9  # HOG orientations`

`pix_per_cell = 8  # HOG pixels per cell`

`cell_per_block = 2  # HOG cells per block`

`hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"`

`spatial_size = (32, 32)  # Spatial binning dimensions`

`hist_bins = 32  # Number of histogram bins`
 
Roughly the other half of the computation goes to spatial-, historgram-features extraction and sliding the window patch over the focus areas cutting hog features and adding up but smoothening the heatmap. The tweaking of spatial and histogram features remained less sensitive to the computation time, one reason is because of the higher total number of hog features. Nevertheless, they add a relevant signature (~2-3%)to the normalized feature-vector moving from ~96.x% to some 99.8% accuracy in predictions.

###3. Classifier training using selected HOG features and color features

For prediction I trained a linear SVM using `sklearn.svm.LinearSVC` class in file `SVM_Classifier.py`. First it reads all the data. In line #27-36 the extraction parameters are set used by the SVM. In line #50-61 the feature vector is build within the function `extract_features` of file `VD_functions`, therein line #87-137. 

`Feature Vector = Spatial + Histogram + HOG = 32*32*3 + 32*3 + 3*9*(8-1)*(8-1)*2*2 = 8460`

In line #65-73 the stacked features are normalized using `StandardScaler`. After the data is split into training and test. The split is balanced to contain equal volume of each class. In line #101 the classifier is trained.

##Sliding Window Search

###1. Implementation of the sliding window search.

I decided to search with a sliding window in 2 specific focus area, see following definitions in #115-120 of file `VehicleDetection.py`:
 
`window_scale = (0.73, 1.35)`

`x_start_stop = [[640, 1280], [640, 1280]]`

`y_start_stop = [[400, 570], [400, 610]]`

`xy_window = (128, 128)`

`xy_overlap = (0.90, 0.95)`
 
 Each area can be understood as vertical layer orthogonal to the lane scanning for objects in 2 different distances. I oriented the scale by intuition and "try-and-optimize" on the principle that objects appear smaller in distance and require higher granularity. Ultimately the patches are combined from each layer (adding in the heatmap) and are a calculus of the dimension of the layer, the dimension of patch and its overlap and the scale-factor chosen for each individual layer.
 
Total # of windows: 1299

![alt text][image3]

###2. Optimize the performance of the classier

Ultimately I searched using HOG features of all HSV-channels plus spatially binned color and histograms of color in the feature vector. 
In order to optimize the performance the HOG-features are calculated only once for each of the three layers. The sliding patch cuts the relevant information and combines it with the spacial and histogram filter using the same patch dimensions, see the code section in line #202-235 in function `find_cars` in file `VD_functions.py` .

The trade-off is a stable detection and bounding-box versus long computing times. In optimal case computing would be done in real-time. The model together with the sliding patch over pre-calculated HOG focus areas reduced the time from ca.2~3 sec to 0.08~0.15 sec per frame and depending on the parameter set, still too much for a real-world application.

Average Image processing time: 0.095 seconds

![alt text][image4]
---

##Video Implementation

###1. Link to final video output.  
The pipeline perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are there but it's identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_detected.mp4)


###2. Filter for false positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap in line #50-56 and then thresholded (defintion in #28-34) that map to identify vehicle positions, see file `VD_DetectorClass.py`.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

######All steps fo the pipeline using the project 6 "Test Images":
![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]
![alt text][image55]
![alt text][image56]
---

##Discussion

###4 Thoughts
i) The feature extraction has been optimized for the provided data set and the specific project video. Specifically the detection area has been minimized (only right and bottom half) to reduce the computation time for each frame. For a left curve, or hilly track the pipeline likely will fail. However, the detection area can be reconfigured to satisfy new requirements.

ii) Further, additional driving and lightning conditions may impact the prediction accuracy. This implementation uses one colorspace. The combination of selected channels from different colorspace could improve the accuracy and generalize better for a variaty of conditions. Another approach could be the use of a Neural Network for detection with augmenting data. In order to detect additional models more training data will be required for trucks, vans etc. The classifier would need to learn and predict additional classes.

iii) The performance of the frame processing remains critical for a real-time application. The more of the frame has to be scanned, the more different classes have to be predicted, the longer the pipeline will compute. One idea could be, instead of scanning each image, detecting the same object again and again in the entire scanning area, the object could be tracked once detected. E.g.the detection area in the next frame, could be the bounding-box area of the current frame. Scanning of the entire space could be reduced to each 5-10 frames in order to detect new objects. A challenge would be the handling of false positives if the number of frames is reduced. To allow frames unprocessed, may impacting the reaction time of the system with safety maneuvers, seems an option compared with the impact of the inertia of the mass of a relatively heavy car with velocity and speed.

iv) Further interesting readings can be found in following posts:

[Feature extraction for Vehicle Detection using HOG+](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10)

[Vehicle Detection & Lane Finding using OpenCV & LeNet-5 (2/2)](https://medium.com/@raza.shahzad/vehicle-detection-lane-finding-using-opencv-lenet-5-2-2-cfc4fea330b4)

-end.