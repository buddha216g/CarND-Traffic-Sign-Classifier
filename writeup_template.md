#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://https://github.com/buddha216g/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

I Plotted a random image and its label and checked it against signnames.csv to make sure the labels and images are matching
I plotted a bar chart showing the number of samples in each class

Both image and bar chart can be seen in the ipynb note book

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I decided to generate additional data because not all classes were represented equally in the given training set

To add more data to the the data set, I used the translation and rotation

I plotted the bar chart after adding additonal data. The distribution seems more reasonable now.

I then decided to convert the images to grayscale because, it is easier to train the model with fewer weights.

I also normalized the data to improve accuracy. (when i ran the model without normalizing my accuracy was below 90%)


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| outputs 400        									|
| Fully connected		| 400,120        									|
| RELU				|        									|
| Dropout		| 0.7        									|
| Fully connected		| 120,84        									|
| RELU				|        									|
| Dropout		| 0.7        									|
| Fully connected						|	84,43											|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
    adam optimizer (which is a part of LeNet architechture code i used in MNIST project)
    batch size = 128
    Epochs = 50 (with 50 i was getting 94.5% validation accuracy)
    Epochs = 100 ( with 100 accuracy didnt increase by much; i am submitting the 100 epoch model but i could have easily stopped at 50
    learnrate = 0.001
    drop out keep probability = 0.7 (i tried with 0.5, 0.6,0.8 and 0.9, but i found 0.7 to be most optimum for my model)

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 94.5%
* test set accuracy of 93%

If an iterative approach was chosen:
* I first used the default architecture i got from the MNIST data project from the lesson before the project
* I found that the validation accuracy with 10 epochs was not improving beyond 90%
* So i first greyscaled and normalized the images. The accuracy didnt improve much.It was peaking at 91% even after i increased my epochs to 50
* I then plotted the number of samples in each class for the training data. Since all classes were not represented well, i tried creating fake data. I ended up using translation and rotation techniques to generate more data to make sure most classes were represented properly.
* I still could not get my validation accuracy to go beyond 92% even after i increased the number of Epochs to 100.
* I noticed that my training accuaracy is starting pretty high and my model is learing very fast but my validation accuracy is not keeping up.
* After i used drop out (finally settled at 0.7 after making some iterations), my validation accuracy crossed 93%
* I achieved 94.5% accuarcy with 50 epochs. Even after i increased the number of epochs to 100 accuarcy didnt change much.
* I stayed with LeNET architecture
* My test accuracy of 93% shows that my model learnt well and is able to classify newer images

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five images from the web are uploaded to the github folder:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h     		| 60 km/h   									| 
| Right of way at next intersection    			| Right of way at next intersection 										|
| 30 km/h					| 30 km/h											|
| Priority Road	      		| Priority Road					 				|
| Keep Right		| Keep Right      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the Each image, the model is relatively sure probability of 1.0 for the first probability), and the image does contain a correct sign. The top soft max probabilities (next four were zeros) were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| 60 km/h    									| 
| .20     				| Right of way at next intersection 										|
| .05					| 30 km/h											|
| .04	      			| Priority Road					 				|
| .01				    | Keep Right      							|




