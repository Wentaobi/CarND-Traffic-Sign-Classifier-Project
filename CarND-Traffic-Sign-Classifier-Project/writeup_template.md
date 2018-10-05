# **Traffic Sign Recognition** 

## Based on Paper: Traffic Sign Recognition with Multi-Scale Convolutional Networks

### In this Traffic sign recognition project, I firstly used LeNet to try, then find paper as mentioned in class, they used ConvNets, I will explain in detail later.

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

[image1]: ./visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic-signs/5x.png "Traffic Sign 1"
[image5]: ./traffic-signs/6x.png "Traffic Sign 2"
[image6]: ./traffic-signs/3x.png "Traffic Sign 3"
[image7]: ./traffic-signs/9x.png "Traffic Sign 4"
[image8]: ./traffic-signs/2x.png "Traffic Sign 5"
[image9]: ./7top5.jpg "top 5 plot"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/7ee8d0d4-561e-4101-8615-66e0ab8ea8c8/concepts/a96fb396-2997-46e5-bed8-543a16e4f72e)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pickle, numpy calculate summary statistics of the traffic
signs data set:

* The size of training set is (39209, 32, 32, 3)
* The size of the validation set is (7842, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is (43)

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because:

* We can reduce input pixels, training time,  and improve network capacity
* The paper told us using grayscale img can get high sccuracy

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this step convert 0 - 255 pixel value to  -1 - 1, which make the data range more narrow than before, I think that's why normally we set smaller learning rate or step to back propagation optimizer.

I decided to generate additional data because can make your network architecture more stable from traning to validation or testing.

### Generate data additional data (OPTIONAL!)

But I do not have time to do it, Life and Work is so busy, sorry here.

To add more data to the the data set, I can rotate/enlarge/adding noise based on original images, and save/generate to new images.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Model Architecture
Implement LeNet-5
Implement the LeNet-5 neural network architecture.

This is the only cell you need to edit.

Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

Architecture
Layer 1: Convolutional. The output shape should be 28x28x6.

Activation. Your choice of activation function.

Pooling. The output shape should be 14x14x6.

Layer 2: Convolutional. The output shape should be 10x10x16.

Activation. Your choice of activation function.

Pooling. The output shape should be 5x5x16.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Layer 3: Fully Connected. This should have 120 outputs.

Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.

Concat. Add 2 flatten layers together.

Output (Layer 5): Fully Connected (Logits). This should have 43 outputs.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
|1 Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6  	|
|1 Activation RELU		|												|
|1 Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
|2 Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16  	|
|2 Activation RELU		|												|
|2 Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
|2 Flatten	        	| outputs 400                   				|
|3 Convolution 5x5     	| 1x1 stride, same padding, outputs 1x1x400  	|
|3 Activation RELU		|												|
|3 Flatten	        	| outputs 400                   				|
|Concat         	    | outputs 800  									|
| Fully connected		| outputs 43   									|
| Liinear output		| logits       									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the Adam optimizer (already implemented in the LeNet lab). The final settings used were:

batch size: 100
epochs: 60
learning rate: 0.0009
mu: 0
sigma: 0.1
dropout keep probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were: Multi-Scale Convolutional Networks paper
* training set accuracy of 99.5% in EPOCH 60
* validation set accuracy of 100%
* test set accuracy of 95.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    The first architecture that I try is LeNet, because this Network model was already shown in class video, which is easy for me to start.
    
* What were some problems with the initial architecture?
    The LeNet architecture has good performance in 1998, about 90+% test accuracy. In traditional ConvNets, the output of the lsat stage is fed to be a classifier, In the present work the output of all the stages are fed to the classifier. This allows calssifier to use, not just high- level features, which tend to be global, invariant, but with little precise details, but also pooled low-level features, which tend to be more local, less invariant, and more accurately encode local motifs.

* How was the architecture adjusted and why was it adjusted?
    Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    Multi-scale CNN features, usual ConvNets are organized in strict feed-forward layered architectures in which the output layer of one layer is fed only to the layer above, Instead, the output of the first satge is branched out and fed to the classifier, in adddition to the output of the first stage after pooling/ subsampling rather than before. Additionly, applying a second subsampling stage onthe branched output yielded higher accuracies than with just one. Therefore, the branched stage 1 outputs are more subsampled rhan in tradituinal ConvNets but  overall undergoes the same amount of subsampling 4x4 here, than the satge 2 outputs. The motivatin for combining representation from multi stages in the classifier is to provide different scales of received fields to the classifier.

* Which parameters were tuned? How were they adjusted and why?
    I run this neural network many times, this is final result I got.
keep_prob: 0.5
EPOCHS: 60
BATCH_SIZE: 100


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Convolution Neural Network is designed to extract features inside the gray img, after 3 times of CNN, use flatten to get previous 2 CNN max pooling output, then use drop out and fully connected layer to get final label calsseds. This is still kind of confusing to me for why we should do/design like this. 


If a well known architecture was chosen:
* What architecture was chosen? 
    LeNet, VGG, or AlexNet, GoogLeNet
    
* Why did you believe it would be relevant to the traffic sign application?
    After I studied the transfer learning class in the later cheapters, I know a good Netowrk model can help you get a good start, because bias, weights, was trained already. And also, it was proved good in the world. This's transfer learning works.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
     The model was working very well. I got highest accuracy in my training history: 99.5%
     EPOCH 56 ...
    Validation Accuracy = 0.993

    EPOCH 57 ...
    Validation Accuracy = 0.994

    EPOCH 58 ...
    Validation Accuracy = 0.993

    EPOCH 59 ...
    Validation Accuracy = 0.994

    EPOCH 60 ...
    Validation Accuracy = 0.995

    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver2 = tf.train.import_meta_graph('./lenet.meta')
            saver2.restore(sess, "./lenet")
            test_accuracy = evaluate(X_test_normalized, y_test)
            print("Test Set Accuracy = {:.3f}".format(test_accuracy))

    And I got Test Set Accuracy = 0.952, 100% accuracy in images set that I pick out from data set.
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is just a direction pointing to the SouthEast, It can be confused when another sign points to NorthWest.
The second image might be difficult to classify because it is left curve combined with straight line and left turn. It can be classified to 
left turn or u turn as well.
The third image might be difficult to classify because it has two square inside with yellow color, it is hard to detect in gray level to know what's this sign does.
The 4th image might be difficult to classify because it has a shape inside, the shape is like mountain, which is hard to detect.
The 5th image might be difficult to classify because this speed 30 can be confused as 80 or something else.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	   					| 
|:---------------------:|:-------------------------------------:| 
| 38,Keep right    		| 100% 									| 
| 34,Turn left ahead	| 100%									|
| 12,Priority road		| 100%									|
| 25,Road work     		| 100%					 				|
| 1,Speed limit (30km/h)| 100%      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 100% in most of cases. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.
![alt text][image9]

TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))
TopKV2(values=array([[  1.00000000e+00,   2.52059441e-13,   4.71558658e-14,
          1.20217285e-17,   7.49231310e-23],
       [  9.57205594e-01,   4.27940823e-02,   3.63944622e-07,
          3.02277279e-08,   2.66300209e-08],
       [  1.00000000e+00,   9.05392397e-21,   2.38077808e-31,
          1.26110749e-31,   1.40517539e-34],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   8.44409433e-25,   2.60013422e-29,
          4.23133304e-36,   1.31100014e-37],
       [  1.00000000e+00,   5.38205209e-14,   2.52341062e-19,
          1.18754960e-19,   5.28451281e-21],
       [  9.99934435e-01,   4.15059912e-05,   1.32883715e-05,
          9.98891119e-06,   4.23746599e-07],
       [  1.00000000e+00,   3.36135984e-29,   8.41569760e-33,
          1.33324856e-34,   8.45739774e-35]], dtype=float32), indices=array([[ 1,  6,  2,  5,  0],
       [ 3,  6, 25,  5, 35],
       [11, 30, 27, 26, 21],
       [38,  0,  1,  2,  3],
       [18, 27, 26, 11, 28],
       [34, 38, 11, 12, 26],
       [25, 39, 37, 33, 20],
       [12,  9, 40,  3, 35]], dtype=int32))


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


