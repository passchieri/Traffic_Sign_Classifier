# **Build a Traffic Sign Recognition Neural Network** 

This document describes the results of the Traffic Sign Recognition Project of the Udacity CarND.

*Author*: Igor Passchier

## Goals of the projects
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[datastats]: ./images/datastatistics.png "Dataset statistics"
[datasample]: ./images/datasample.png "Samples of traffic sign images"
[augment]: ./images/augment.png "Image Augmenting"
[augmentsample]: ./images/augmentsample.png "Sample of augmented images"
[augmentpreprocessed]: ./images/augmentpreprocessed.png "Sample of preprocessed augmented images"
[internet]: ./images/internet.png "Original and preprocessed images from internet"
[quality]: ./images/quality.png "Overview of quality of the results of the sign recognition"
[image1]: ./images/1.jpg "Internet image 1"
[image2]: ./images/2.jpg "Internet image 2"
[image3]: ./images/3.jpg "Internet image 3"
[image4]: ./images/4.jpg "Internet image 4"
[image5]: ./images/5.jpg "Internet image 5"

## Rubric Points

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
2. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
3. Include an exploratory visualization of the dataset.
4. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
5. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
6. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
7. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
8. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications? *Optional*

---
### Writeup

This is the writeup of my project. The project code can be found on  [github](https://github.com/passchieri/Traffic_Sign_Classifier)

### Data Set Summary & Exploration

I used the plain python and numpy to calculate a summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

So we have a training set of almost 35000 RGB 32x32 images.

### Visualization of the dataset
The image below gives an overview of all the signs in the dataset.

![Statistics of the data set][datastats]

It can be observed that some signs occur much more frequent than others. For every sign type, I have plotted 10 random images to investigate the image quality. Below, these are shown for 4 signs. 

![Overview of random images per sign type. All others can be found in the html page][datasample]
In the [HTML](https://github.com/passchieri/Traffic_Sign_Classifier/blob/master/report.html) page of the jupyter notebook, I have also shown 10 random images per sign class. These images show that the quality of the images fluctuates a lot. Especially, many images are rather dark. This means it is crucial to do a proper preprocessing to normalize the image content.

### Design and Test the Model Architecture
I focussed my optimization on the preprocessing steps, starting with the Lenet model used previously. This gave a stable basis to investigate the effects of the preprocessing. 

#### Preprocessing
I tested several preprocessing steps
1. Grayscale. I experimented with using either color images or grayscale images. In the end, I used grayscale images, as that gave the best results. This came a bit as a surprise, as the color information already provides a lot of information on the traffic sign. Probably with a better normalization algoritm for the color images, it would have been possible to get better results with color images. However, I did not manage to find a good (combined) solution for color image usage and normalization. Filtering out some colors was difficult due to the large variation in intensitiy, and the sharp lines that result from color filtering

2. Recropping. I tried recropping the images based on the provided bounding boxes. This improved the training, but was difficult to use in combination with the augmentation. Therefore, I did not use the recropping in the end. The code is still available in the HTML page.

3. Normalizing. The first normalization I tried was just scaling by subtracting 128 and dividing by 128. I obtained better results when using the min and max intensity in the grayscale images to really go to -1 - + 1 scale. The formula used is: pix=(pix-max/2-min/2)/(max/2-min/2)

4. Balancing. Many signs are over/under represented in the data set. This will result in a better classification of images that are over represented. I tried to fix this my keeping the same number of images per sign type. However, then far too little images are left to do a proper training (traning accuracy >> validation accuracy). Therefore, I did not use the balancing in the end.

5. Image augmentation. The increase the number of images for training, I used to augmentation techniques: rotating the images by +-30 degrees, and by warping with a projective transform. I used the skimage.transform library for the actual transformations. In this way, I generated 5 times more images than the original training set. Of course, I did not use the augmentation for the validation and test data sets. The image below shows the 2 transformations for a sample image
![Example of the two augmentation techiques used][augment]

Below, 16 samples of the augemented images are provided.
![16 samples of augmented images][augmentsample]


The whole preprocessing resulted in **173995** training samples. 16 samples are shown below.
![16 samples of fully preprocessed images][augmentpreprocessed]



### Model architecture

I use the Lenet model with only little adaptations. I have added 2 dropout functions, after the max pooling layers,

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout            | p=0.8 during training |
| Convolution 5x5	    | x1 stride, valid padding, outputs 10x10x16    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout            | p=0.8 during training |
|Flatten | Output 400 |
| Fully connected		| Output 120        									|
| RELU					|												|
| Fully connected		| Output 84        									|
| RELU					|												|
| Fully connected		| Output 43 (number of classes)        									|
| Softmax				| Output 43 Probabilities        									|
 

### Model training

To train the model, I started with the Lenet model from the previous excercise. I played around with the number of epochs, rate, and batch size. In the end, with a rate of 0.001 and batch size of 128, the validation results stop improving after 40-50 epochs. Therefore, I choose these values, and an 60 epochs. I added the Dropout to the model to prevent overfitting, and improvide the validation accuracy. I also tried with RGB images (only changing the input depth), but that did not give better results. Maybe with additional training images, better normalization and/or modified convolutions it would be possible to get better results with RGB images, as those contain more information then the RGB images (going to Grayscale really throws away information from the image)

My final model results were:
* Training Accuracy = 0.979
* Validation Accuracy = 0.945
* test Accuracy of 0.930

### Test the Model on New Images
Below are the images used from the internet to test the performance of the final model. 

![New images used from internet][internet]

All images are nice and clean, which of course is a little bit cheating. However, these where the first images popping up in my serach without watermarks on it. I have cut out the images myself, and reduced the size to 32x32. I tried to cut out in a similar fashion as the provided data set.

The model predicted all signs correctly. The probabilities are provided in the image below

![results of the top 5 probabilities for the 5 internet images. Note, that the horizontal axis is logaritmic][quality]


The code to create these images is given below:
```
def plot_quality(img,y,values,indices):
    plt.figure(figsize=(12,8))

    ax1 = plt.subplot2grid((4,8), (0,7), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((4,8), (0,1), rowspan=1, colspan=6)

    names=label_names[indices]
    ax1.imshow(img)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    
    
    ax2.barh(range(len(indices)),values)
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels(names)
    ax2.set_xscale('log') 

    plt.show()
    
with tf.Session() as sess:
    # Load the weights and bias
    saver.restore(sess, tf.train.latest_checkpoint('./') )
    prediction=tf.nn.softmax(logits=logits)
    top5=tf.nn.top_k(prediction,k=5)
    out=sess.run(top5, feed_dict={x: X_internet, y: y_internet, pars.conv1_p:1.0,pars.conv2_p:1.0})
    values=out.values
    indices=out.indices
    for i in range(len(values)):
        plot_quality(X_internet_img[i],y_internet[i],values[i],indices[i])
```
The certainty for most images is extremely high (>99%). The 30 km sign is sligtly lower, but still >90%. FOr the 30 km sign, the other candidates are the other max speed limits, wich is logical, as these resemble the image the most. Something similar can also be observed in the fifth image, where the other candidates also for humans resemble the "turn right" sign the most. It is also clear from the results that color is not taken into account, otherwise a red stopsign would not show up in the list of top 5 probabilities for the "turn right ahead" sign.

### (Optional) Visualizing the Neural Network
I did not do this optional task.



