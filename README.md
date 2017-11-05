# PDC
J- component
Traffic Light Detection
===

### 
Authors

* Vinish Makkar 

###
Installation 

* Install Python3
Run 

###
Install Microsoft Visual Studio 
Upload the files in the drectory of visual studio
Run the files making a collaboration
-output will be shown.
		
pip3 install -r requirements.txt

* 
Install MATLAB >= 2014a

### 

overview

The perception subsystem dynamically classifies the color of traffic lights in front of the vehicle. In the given simulator and test site environment, the car faces a single traffic light or a set of 3 traffic lights in the same state (green, yellow, red). We assume it is not possible to have multiple traffic lights in the different states at the same time.

We have considered different approaches to solve the traffic light classification task:

classification of the entire image using CNN;
object (traffic light in state) detection;
object (traffic light) detection and classification using separate model.
Considering the fact that traffic lights are always in the same state, and focusing on the creation of a lightweight and fast model, we've chosen the direction of classifying the entire image. This approach uses a Convolutional Neural Network, which takes a whole image from the front camera as an input and predicts the traffic light state (we've decided to use Red / None prediction classes) as an output. We used the transfer learning technique on the MobileNet architecture with the Tensorflow Image Retraining Example (tutorial: https://goo.gl/HgmiVo, code: https://goo.gl/KdVcMi).

Dataset

There are multiple datasets, available for model training:

images from the Udacity Simulator (images as well as the ground truth from the frontal camera are available as a ROS topic);
https://drive.google.com/open?id=0Bw5abyXVejvMci03bFRueWVXX1U
rosbag, captured on the Udacity's test site;
https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view
Bosch Small Traffic Lights Dataset.
We've trained our model on a mixture of the datasets above.
Image pre-processing

On the image pre-processing step we've applied multiple visual transformations:

random cropping of the image;
rotation on the random angle (+/- 5 degrees);
random flipping of the up to 20% images;
random color jittering;
applying shadows (reference: https://goo.gl/VzoxcY).
In order to slightly balance dataset, some images (manually chosen) were augmented.
Neural Network Model

"Simple transfer learning with MobileNet model" example from TensorFlow was used to re-train our model. We started with a MobileNet model pre-trained on the ImageNet images, and trained a new set of fully-connected layers with dropout, which can recognize our traffic light classes of images. The model works with the image dimensions 224x224x3. The top fc layer receives as input a 1001-dimension feature vector for each image.



MobileNets are neural networks constructed for the purpose of running very efficiently (high FPS, low memory footprint) on mobile and embedded devices. MobileNets achieve this with 3 techniques:

Perform a depthwise convolution followed by a 1x1 convolution rather than a standard convolution. The 1x1 convolution is called a pointwise convolution if it's following a depthwise convolution. The combination of a depthwise convolution followed by a pointwise convolution is sometimes called a separable depthwise convolution.
Use a "width multiplier" - reduces the size of the input/output channels, set to a value between 0 and 1.
Use a "resolution multiplier" - reduces the size of the original input, set to a value between 0 and 1.
These 3 techiniques reduce the size of cummulative parameters and therefore the computation required. Of course, generally models with more paramters achieve a higher accuracy. MobileNets are no silver bullet, while they perform very well larger models will outperform them. MobileNets are designed for mobile devices, NOT cloud GPUs. The reason we're using them in this lab is automotive hardware is closer to mobile or embedded devices than beefy cloud GPUs.[1]
Accuracy on the simulator data: 

Accuracy on the Udacity's test track data: 

Usage
Prepare Data

This step is the same as my previous repo kitti_ssd, train kitti dataset on ssd, this repo using dataset is LISA, a traffic light detection dataset. As the same of previous, we get Images and Label folder which contains all images and labels, one image reflect on same named label file, in ./data folder we have a script gen_all_labels.sh to generate same named label files contains nothing, just in case some images has no label bound box, this may cause unable to train SSD.

Generate lmdb data

To train caffe, we'd better use lmdb database to feed data into network, since we got images and labels, oh, another thing I just forgot, label file format as follow:

class_index x_min y_min x_max y_max
Continue, to generate lmdb data you only need to type:

bash ./data/create_list.sh
This will generate trainval.txt , test.txt and test_name_size.txt. In create_list.sh you have to change your dataset name, and we suppose your data placed in ~/data. So, if you got a dataset named FACE, then your trainval.txt and test.txt will seems like this:

bash ./data/create_data.sh
In the sh file, we have a data_dir viraible, please make sure data_dir + trainval.txt line can reach your real image, otherwise you cannot generate lmdb file, or just get zero lenght. After this step, you are going have trainval.lmdb and test.lmdb 

Predict the block negativity of traffic system

In this part of the system we will the block negativity by counting the number of cars 
that has been passed through that traffic light, so that we can actually prevent conjustion 

Counting the frequency of the system cojustion is done by blobs.cpp




Predict Using SSD

ok, before train SSD, you can test this, you can download 2 model, VGG16.v2.caffemodel and VGG_LISA_SSD_414x125_iter_120000.caffemodel`, the first one is SSD pretrain model, you gonna need it when you train SSD, and second is I trained model using for traffic light detection. Place first model into ./models/VGGNet/LISA/SSD_414x125 Place second model into ./models/VGGNet And then type:

Train SSD
To train SSD, you just need:

python train_ssd_lisa.py
But you need to change some directory and datset names.

Predict Many Images

We provide a script to predict all images under an dir, using ssd_detection_all.py instead.




