##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features from the training images

The code for this step is contained in function `get_hog_features` in lines #239-258 of the file called `VD_functions.py`. The function is called in line #50,56 for car and notcars by the `extract_features` function in file `SVM_Classifier.py`. That function is defined also in `VD_functions.py`, see line #87-138.

I started by reading in all the `vehicle` and `non-vehicle` images, see #14-18 in `SVM_Classifier.py`. 

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `LUV` and `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for a 'car' and a 'notcar' class:

#####a. LUV Color Space
![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

#####b. HSV Color Space

![alt text][image21]

![alt text][image22]

![alt text][image23]

![alt text][image24]







####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

