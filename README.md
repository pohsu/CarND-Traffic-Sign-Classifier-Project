# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/augmentation.png "augmentation"
[image3]: ./examples/augmentation_hist.png "augmentation_hist"
[image4]: ./examples/pre_processing.png "pre_processing"
[image5]: ./examples/pre_processing_batch.png "pre_processing_batch"
[image6]: ./examples/precision.png "precision"
[image7]: ./examples/recall.png "recall"
[image8]: ./examples/web_images.png "web_images"
[image9]: ./examples/prob.png "prob"
[image10]: ./examples/fmap0.png "fmap0"
[image11]: ./examples/fmap1.png "fmap1"
[image12]: ./examples/fmap2.png "fmap2"
[image13]: ./examples/fmap3.png "fmap3"
[image14]: ./examples/fmap4.png "fmap4"
[image15]: ./examples/fmap5.png "fmap5"
[image16]: ./examples/fmap6.png "fmap6"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is my project notebook: [project code](https://github.com/pohsu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the histogram of the training set labels. It can be seen that the distribution of the samples is far from a uniform distribution, which could be an issue.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Before the pre-processing step, I noticed that the distribution of the training samples is not spreading evenly across classes. Thus, I decided to perform adjustment of sample distribution. The easiest way is to manipulate the samples of the classes that are below the average number of images per class, ~809 samples, with duplicated copies of themselves. However, this may introduce redundancy of information; thus a better way is to apply image augmentation.

I went online and search for different methods and finally I ended up using only three methods, which are rotation, perspective adjustment, Gaussian Noises as I found that they could forge random imperfectness of the samples. Other methods such as shear, translation, or brightness adjustment are either covered by the efficient combination of the above methods applied or not effective due to later pre-processing techniques.

Here are some example images after augmentation:

![alt text][image2]

The post-augmented histogram becomes:

![alt text][image3]

`Pre-augmentation training size: 34799, Post-augmentation training size: 352678`

The distribution more or less becomes closer to a uniform distribution with the number of the samples increased to 352678.

For the pre-processing, I then applied the global normalization and keep only the Y channel so the processed image look like in grayscale. Grayscale has been proven immune to color noises as the color images do not bring much additional information for the traffic sign case. Some samples after pre-processing are shown:

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		     |     Description	        		        		   |
|:----------------------:|:-------------------------------------------:|
| Input         		     | 32x32x1 Y image   							             |
| Convolution 3x3     	 | 1x1 stride, valid padding, outputs 30x30x6  |
| Batch Normalization    |                                             |
| RELU					         |												                     |
| Max pooling	      	   | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 3x3	       | 1x1 stride, valid padding, outputs 12x12x10 |
| Batch Normalization    |                                             |
| RELU					         |												                     |
| Max pooling	      	   | 2x2 stride, valid padding, outputs 6x6x10   |
| Convolution 3x3	       | 1x1 stride, valid padding, outputs 4x4x16   |
| Batch Normalization    |                                             |
| RELU					         |												                     |
| Fully connected	+ RELU (2)  | FC1 outputs 120, FC2 outputs 84    		 |
| Fully connected        |     		                                   |
| Softmax 		          |         	 								||




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I pretty much followed the general guideline of training a model from coarse to fine tunings. The followings are the final values of hyperparamters:

* beta (L2 regularization) = 4e-4
* learning rate = 5e-4
* learning decay rate after loss plateau = 0.5
* loss plateau threshold = 0.01
* batch size = 240
* number of epochs = 30
* Optimizer = Adam

First, I trained the model by automatically sweeping through all these hyperparameters with a small dataset in order to find some good approximated values. The values are then obtained based on the best validation accuracy. Then I start using the full augmented dataset and tried a little bit in order to fine tune these values.
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

After pinning down a good set of hyperparameters, I ran the models many times with 30~40 epochs per each training time and preserve the best weights based on the highest validation accuracy. In between, I also tried different hyperparameters, different model architecture, and various augmentation methods to boost the robustness of the model. Finally I was able to improve the validation accuracy from 0.95, 0.96, 0.967, 0.977, then 0.981.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.981
* test set accuracy of 0.96

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I tried the default LeNet architecture to get a starting point.   
* What were some problems with the initial architecture?
  * The accuracy seems to plateau at around 0.96 and I think the reason is that the low capacity of the model.
* How was the architecture adjusted and why was it adjusted?
  * Batch normalization: I use batch normalization as it helps to improve the training speed and serve partially for preventing over-fitting. Based on the literature, Batch norm and dropout are normally exclusive so I did not attempt to use dropout.
  * Adding an additional convolution layer: I reduced the filter sizes from 5 to 3 but added an additional layer so the receptive zone is similar while smaller filters can help to preserve finer details of the images.
* Which parameters were tuned? How were they adjusted and why?
  * Regularization constant: I tuned this parameters to prevent over-fitting as in the beginning its very easy to get 1.00 training set accuracy while having the validation accuracy below 0.95.
  * Learning rate and decay rate: I noticed that to attain high validation accuracy is to apply learning rate decay (the learning rate itself becomes less sensitive!). Whenever the loss function enters plateau with the difference between epochs lower than 0.01, the decay rate of 0.5 is applied to the learning rate so that it gets better to enter a deeper valley to get better accuracy. This method is extremely effective as can be seen from the loss function versus iterations. Right after the decayed learning rate, the loss function drops immediately along with the validation and training accuracy.
* What are some of the important design choices and why were they chosen?
  * I think the successfulness of the model could be contributed to the reduced sizes of the filter with a deeper layer of convolution networks. However, most things are pretty heuristic and it is still far from enough for me to fully understand the details. More experimental tests should be conducted to be able to understand why the models work better than the original LeNet. Some papers suggests to use a two-stage design with a feedforward layer that allows the fully-connected layers to obtain higher resolutions of feature maps in early layers. I tried the models and found it effective; however, I did not report this model as I tend to make a good model based on a simpler structure then maybe further conduct tests with the model with feed-forward structures.

<!-- If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? -->


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Before picking the images, I compute the precision/recall based on the test set. Here are the bar charts for computing precision/recall per class:

![alt text][image6]
![alt text][image7]

I then picked two images with low recall and precision (I actually tested much more than these five images) and three easy classes. Here are five German traffic signs from the web that I used:

![alt text][image8]

* Ice/Snow: Some parts of the sign are covered by snow so its a real snowy sign not just what it means to show. Also, the class 30 has particularly low precision and recall, so I am very curious to see what prediction I am getting.
* Bumpy Road: I intentionally made it blurry to test the model and the image has lots of irrelevant details.
* Double curve: This one has the lowest recall among classes.
* 30 kph limit: '30' could be easily predicted as '80' sometimes.
* Right-of-way : The class 23 has no good precision and I found that the class 11 can be easily predicted as 23. This is to show such a case.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Ice/Snow      		| Right-of-way   									|
| Bumpy Road     			| Bumpy Road 										|
| Double curve			| Double curve										|
| 30 km/h	      		|  30 km/h					 				|
| Right-of-way			| Slippery Road      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares to the accuracy on the test set of 96%. But it is a bit unfair as I selected hard classes to test my model.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in `Step 3.4`.

Here I present the top 5 probabilities using the bar chart for each image:

![alt text][image9]

It seems like the model is pretty "stubborn" in choosing only a single class without any other possibility, which might be a sign of over-fitting that minimizes the cross-entropy losses too much so that the distributions are very spiky. But it is too hard to make any certain conclusion based on these 5 samples from the web.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here I choose to plot the RELU activation before the max-pooling of the first CONV layer which has six filters so I can have easy-to-visualize images:

`<tf.Tensor 'Relu:0' shape=(?, 30, 30, 6) dtype=float32>`

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

Now it's easier to see what the model sees in this activation layer:
* Fmap0: I think it is looking at left edges with the gradient pointing +45 degs.
* Fmap1: Right edges with the gradient pointing +135 degs thanks to lots of images with triangular frames.
* Fmap2: Left edges with the gradient pointing -45 degs.
* Fmap3: Edges without particular directions.
* Fmap4: Finding chunks of blocks in the image.
* Fmap5: Not very sure.

#### 2. Some plots of other layers' feature maps
* RELU activation maps of the 2nd CONV layer:
` <tf.Tensor 'Relu_1:0' shape=(?, 13, 13, 10) dtype=float32>`
![alt text][image15]
* RELU activation maps of the 3rd CONV layer:`<tf.Tensor 'Relu_2:0' shape=(?, 4, 4, 16) dtype=float32>`
![alt text][image16]

When it goes deeper, the size of the maps reduces with increased filter sizes so that it becomes very detailed and difficult to obtain good understanding.

### Possible improvements
Since I wrote this report at the end, here is what I found could be useful to improve the accuracy of the mdoel.

* Improvement of augmentation methods and obtain more augmented data can really help to increase robustness and thus the test/validation accuracy.
* With precision/recall charts, we can perform further just-enough augmentation for those classes. Too much handcrafting may lead to over-fitting with influences of the test-set.
* After visualization, it is possible to re-design the filter sizes of each conv layer to be able to capture meaningful characteristics of the images.
* Trying popular models, it is always good to start from using the models that have been proven effective as they have been well studied with lots of efforts. So we can then improve the model performance progressively without wasting too much time. However, for me this project is to enhance the tanning techniques with a not-too-good model (If a well-done model gives you 99.9%  test accuracy, where are tears and funs??).  
