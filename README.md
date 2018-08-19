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

After experimenting with several color spaces - which you could see from the options in the function `convert_format` in `feature_extraction.py` in lines 11-24 - I settled on using `YCrCb` as my color space of choice for extracting HOG features. I used `9` orientation bins and `8x8` pixels per cell and `2x2` cels per block. 

Below is an example of HOG features for a vehicle.

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.
After trying several HOG parameters, I realized that the best discriminating power that I obtained - `~98.90% accuracy in validation set` - was using a feature descriptor with all channels of a YCrCb image with parameters described in the above section. Also, I used a 32-bin color histogram of the `YCrCb` channels and a spatial histogram of the same color space. Overall, this was an `8000+D` feature descriptor.   

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
In the `feature_extraction.py` function, I gathered functions to extract features, these are used in the main method in `feature_extraction.py` to extract features from all car and non-car images in the training/vaidation set. The main function in this script extracts features from images in every subfolder within the `vehicles` and `non-vehicles` folder. The `classify.py` uses pre-extracted features from the previous step to build a `LinearSVC` classifier and estimate the validation accuracy. This is then stored as a pickle file - `train.p` - which is used in the vehicle detection and tracking project.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The `find_cars` function in `single_image_detection.py` contains the code to implement a sliding window based search for potential cars between `lines 61-129`. Initially the vehicle detections were very inaccurate. To fix this, I increased the overlap ratio by a factor of two following which I had better detections. The detections were then improved tremendously when I started searching for a scaling factor of 1.5 in addition to searching in the base scale. For improving the time complexity, the search was conducted only for the right half of the screen between the y-coordinates of 400 and 656 pixels.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video_tracking.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
In the video pipeline, I have two modules to take care of false positives: using the heatmap to eliminate false positives and using temporal consistensy to eliminate stray false positive bounding boxes.

In the functions `heat_map` between lines `158 and 166` in `single_image_detection.py` and `draw_stable_bboxes` between lines `124 and 149` in `video_detection_and_tracking.py`, I extract the heatmap of the vehicle detections by accumulating the votes for every pixel position in the image. Then, I find the highest values and find an optimum threshold which is the greater of the value 10 or 20% of the maximum votes for the image. Then, the `draw_stable_bboxes` function extracts the exact bounds of the bounding boxes for blobs that have votes more than the threshold.

Following this, I use the bounding boxes to either add to trackings of previously detected bounding boxes or generate new trackings. This pipeline is in between lines `46 and 121` in `video_detection_and_tracking.py` where I create and manage tracks. I perform data association by using the intersection-of-union measure and find tracks to associate with. If a bounding box appears only in one image with no successor or predecesor, it is discarded. If a bounding box is plausible and it has the bounds for the previous frame. That vehicles bounds are represented as the median values for each one of the 4 2-D bounds. This brought about a great deal of stability in the video pipeline and got rid of false positives in single frames.  


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
       

