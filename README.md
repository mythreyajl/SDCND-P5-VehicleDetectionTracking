## Poject Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/image0936.png
[image2]: ./writeup_images/image1.png
[image3]: ./writeup_images/HOG.jpg
[image4]: ./writeup_images/
[image5]: ./writeup_images/
[image6]: ./writeup_images/
[image7]: ./writeup_images/
[image8]: ./writeup_images/
[video1]: ./output_project_video_tracking.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading the writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extract HOG features from training features in lines 43-61 in the function`extract_hog` in the script `feature_extraction.py`. In order to extract features, I read all the images from the `vehicles` folder and added the extracted features into a list. Similarly, for negatives I extract features from images in the `non-vehicles` folder and put them in another dedicated list. I create 1 and 0 labels respectively for each of the images extracted

Below are example vehicle and non-vehicle images

![alt text][image1]

![alt text][image2]

After experimenting with several color spaces - which you could see from the options in the function `convert_format` in `feature_extraction.py` in lines 11-24 - I settled on using YCrCb as my color space of choice for extracting HOG features. I used 9 orientation bins and 8x8 pixels per cell and 2x2 cels per block. 

Here is an example of the HOG features using the `YCrCb` color space and parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. The top-left image corresponds to the YCrCb image, the top-right to the Y-channel HOG features, the bottom-left to the Cr-channel HOG features and the bottom-left to the Cb-channel HOG features.

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?


![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video_tracking.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

### Some of the problems I faced are as follows along with the steps I took to remedy them:
1. __Issue__: There weren't enough number of detected bounding boxes with cars in them.  
   __Fixes__: I increased the sampling frequency by decreasing the stride in order to gather a bigger region of support. I also had two different scales in which I looked for positive matches. These two techniques in conjunction increased the number of possible positive detections.
2. __Issue__: The detections were very wobbly from frame to frame in the first pass of the detection-only pipeline.
   __Fixes__: There are several fixes for this issue that I applied:
       * Firstly, I associated each new detection with previous frames detection of the same object. For this, I used an intersection-of-union (intersecting area divided by the area of the union) measure to associate vehicle bounding boxes with each other. 
       * Following this, I used the median of the bounding boxes (top, left, right, bottom) boundaries for each track of vehicles over 10 previous frames. The detections are tracked over frames and at each frame, the overlaid bounding boxes are a result of median filtering of the bounding boxes. Median filter is especially good at 1-D outlier rejection (like salt-pepper noise). 
3. __Issue__: False positives.
   __Fixes__: These were the techniques applied for this particular issue:
       * I considered a bounding box invalid if it had no prior history. This eliminated single-frame false positives.
       * I created a heat map for potential bouning boxes for cars. Following this, I found true vehicles by using an adaptive threshold based on the maximum value of the heatmap response. I rejected any box that was lesser than a percentage of the maximum which was capped at the lower end at a value of 10.
       
### Potential areas where my pipeline would fail are as follows:
1. I detect vehicles only between 400-656 pixels in the y-axis. This is an assumption made based on the pitch of the vehicle. If the road is bouncy, there are possible cases in which my pipeline would fail.
2. I detect vehicles only on the right half of the image. My pipeline will fail if the vehicle isn't driven on the leftmost lane. 
3. My pipeline involves an exhaustive search of an 8000+ dimensional feature window. It can never run real time unless a few fixes are made.

### A few fixes for robustness:
1. One could improve the runtime efficency of this submission by tailoring a search in the following manner:
    * For every frame, detect potential new vehicles in the following four regions: in a slice along the bottom, left and right and in a slice right below the horizon.
    * Track earlier discovered vehicles by tailoring search for those vehicles very close to the previous frames detections. 
2. The runtime efficiency can be improved further by downsizing the image.
3. It can also be improved by reducing dimensionality of the feature descriptor. One could use PCA, Fisher-LDA, SVD or Decision trees to do so. Reducing the dimensionality decreases the complexity of feature extraction and matching.
4. The quality of vehicle detection can be improved by predicting the size and location of the bounding box of a new frame by observing the past few frames. Further, 2D splines could be fit to the bounding box vertices and the centroid of the vehicle and its new location can be predicted by evaluating the spline at the next time step.
       

