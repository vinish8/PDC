# PDC
J- component

Implementation at:
https://drive.google.com/file/d/12uvb7jggms4iqc-8sfZ96aS5Q1SxZIt5/view?usp=sharing


Car Detection at Traffic Light 
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




This below set aims to provide tools and information on training your own OpenCV Haar classifier.

Instructions

Install OpenCV & get OpenCV source

 brew tap homebrew/science
 brew install --with-tbb opencv
 wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.9/opencv-2.4.9.zip
 unzip opencv-2.4.9.zip
Clone this repository

 git clone https://github.com/mrnugget/opencv-haar-classifier-training
Put your positive images in the ./positive_images folder and create a list of them:

 find ./positive_images -iname "*.jpg" > positives.txt
Put the negative images in the ./negative_images folder and create a list of them:

 find ./negative_images -iname "*.jpg" > negatives.txt
Create positive samples with the bin/createsamples.pl script and save them to the ./samples folder:

 perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
   "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
   -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 80 -h 40"
Use tools/mergevec.py to merge the samples in ./samples into one file:

 python ./tools/mergevec.py -v samples/ -o samples.vec
Note: If you get the error struct.error: unpack requires a string argument of length 12 then go into your samples directory and delete all files of length 0.

Start training the classifier with opencv_traincascade, which comes with OpenCV, and save the results to ./classifier:

 opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
   -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
   -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024
If you want to train it faster, configure feature type option with LBP:

  opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt\
   -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
   -numNeg 600 -w 80 -h 40 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024 -featureType LBP
After starting the training program it will print back its parameters and then start training. Each stage will print out some analysis as it is trained:

===== TRAINING 0-stage =====
<BEGIN
POS count : consumed   1000 : 1000
NEG count : acceptanceRatio    600 : 1
Precalculation time: 11
+----+---------+---------+
|  N |    HR   |    FA   |
+----+---------+---------+
|   1|        1|        1|
+----+---------+---------+
|   2|        1|        1|
+----+---------+---------+
|   3|        1|        1|
+----+---------+---------+
|   4|        1|        1|
+----+---------+---------+
|   5|        1|        1|
+----+---------+---------+
|   6|        1|        1|
+----+---------+---------+
|   7|        1| 0.711667|
+----+---------+---------+
|   8|        1|     0.54|
+----+---------+---------+
|   9|        1|    0.305|
+----+---------+---------+
END>
Training until now has taken 0 days 3 hours 19 minutes 16 seconds.
Each row represents a feature that is being trained and contains some output about its HitRatio and FalseAlarm ratio. If a training stage only selects a few features (e.g. N = 2) then its possible something is wrong with your training data.

At the end of each stage the classifier is saved to a file and the process can be stopped and restarted. This is useful if you are tweaking a machine/settings to optimize training speed.

Wait until the process is finished (which takes a long time — a couple of days probably, depending on the computer you have and how big your images are).

Use your finished classifier!

 cd ~/opencv-2.4.9/samples/c
 chmod +x build_all.sh
 ./build_all.sh
 ./facedetect --cascade="~/finished_classifier.xml"

