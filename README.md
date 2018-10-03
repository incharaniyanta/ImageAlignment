# ImageAlignment
Feature based Image Alignment

##### about the files:
###### executable file - encapsulation.py
###### graph of high level model - graph.jpg
###### input image pairs - scanned-form and form / scanned-form1 and form1


## Steps for Feature Based Image Alignment

### Read Images : 
We first read the reference image (or the template image) and the image we want to align to this template in Lines 70-80 in C++ and Lines 56-65 in the Python code.

### Detect Features:
We then detect ORB features in the two images. Although we need only 4 features to compute the homography, typically hundreds of features are detected in the two images. We control the number of features using the parameter MAX_FEATURES in the Python and C++ code. Lines 26-29 in the C++ code and Lines 16-19 in the Python code detect features and compute the descriptors using detectAndCompute.

### Match Features:
In Lines 31-47 in C++ and in Lines 21-34 in Python we find the matching features in the two images, sort them by goodness of match and keep only a small percentage of original matches. We finally display the good matches on the images and write the file to disk for visual inspection. We use the hamming distance as a measure of similarity between two feature descriptors. The matched features are shown in the figure below by drawing a line connecting them. Notice, we have many incorrect matches and thefore we will need to use a robust method to calculate homography in the next step.
Matching Keypoints in OpenCV
Figure 4. Matching keypoints are shown by drawing a line between them. Click to enlarge image. The matches are not perfect and therefore we need a robust method for calculating the homography in the next step.

### Calculate Homography:
A homography can be computed when we have 4 or more corresponding points in two images. Automatic feature matching explained in the previous section does not always produce 100% accurate matches. It is not uncommon for 20-30% of the matches to be incorrect. Fortunately, the findHomography method utilizes a robust estimation technique called Random Sample Consensus (RANSAC) which produces the right result even in the presence of large number of bad matches.Lines 50-60 in C++ and Lines 36-45 in Python accomplish this in code.

### Warping image:
Once an accurate homography has been calculated, the transformation can be applied to all pixels in one image to map it to the other image. This is done using the warpPerspective function in OpenCV. This is accomplished in Line 63 in C++ and Line 49 in Python


### Reference:
https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
