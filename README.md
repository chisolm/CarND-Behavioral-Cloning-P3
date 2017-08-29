# Behaviorial Cloning Project for Self Driving Car Class

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./writeup_images/flip_lrc_images.png "2 sets of input images"
[image2]: ./writeup_images/brightness_aug.png "Brightness Augmentation"
[image3]: ./writeup_images/steering_angle_hist.png "Histogram of steering angles"

## Overview

My car is successfully able to navigate the 1st track easily and repeatably at the lower
default speed of 9mph specified in the drive.py script.  At 15-20mph setting it will 
navigate the track succesfully but has some instability in the 2 turns after the bridge.
It succeeds in most cases, however if the initial conditions of the turn are out of its
expected range it will drive off track.

## Installation & Resources

1. Anaconda 
2. Udacity [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
3. Udacity car [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
4. Udacity supplied [data](hyperlink)


## Files and Usage

### Files
* `model.py` : The script used to create and train the model.

* `drive.py` : original file, only modified one line for speed

* `data.nvidia.model.h5` : The saved model weidghts. 

* `README.md` : This file.  Description of the project.
* `run1.mp4` : a recording of a successful run

### Usage of model.py


```
usage: model.py [-h] [--epochs EPOCHS] [--prefix PREFIX] [--display]
                [--model MODEL] [--generator] [--batch BATCH] [--augmentbase]

Process flags.

optional arguments:
  -h, --help       show this help message and exit
  --epochs EPOCHS  Number of epochs to run
  --prefix PREFIX  prefix name data dir and model.
  --display        Display pretty graphics
  --model MODEL    Model to use. default: lenet
  --generator      User a generator for data and validation. default: False
  --batch BATCH    Batch size. default: 128
```

Model.py will run with no options where it selects defaults values.  The primary option of interest is `--model` which will select a model network to use.  Current options are 'lenet', 'nvidia' and 'nvidia_1'.

## Rubric Points

### Files Submitted and Code Quality

#### 1. Submitted files.  See files above for list.

#### 2. Collecting data

I collected 2 complete loops of the easy track and recovery data for several
corners that I had trouble with in my original model.  The trouble corners 
were the first corner with the dirt section entry and the following tight
corner.  My data was collected with the keyboard and suffers from quite high
steering angles.  This data was nearly unusable.

I ended up using only the udacity data when I moved to the nvidia model and
a varient on it.  I also stayed with the udacity data to be a more consistent
comparison of models.  I did collect a limited ammount of recovery data.  The
recovery data had to be edited to remove some of the start/stopping lines as
the did not describe desired behavior.

High quality data collection appears to be the key for this project as most
model tests I made had no perceptable performance change.

I did do data collection for the second track.  A single lap, it did show a good 
variety and balanced set of steering angles.  I would need 2-3x the data at minimum
for a success run based on the numbers in the supplied data set, even that assumes the
complexity level of the tracks are similar which seems unlikely.

Here is a histgram of steering angles for the data sets I have, note the differing 
y-axis values:
![alt text][image3]


#### 3. Image processing and augmentation

The initial image load is combined with basic augmentation.  It adds the left and right images with a .2 steering angle adjustment for those images.  For the center image it flips the image for a balanced set of turns.  See below for 2 sets of input images:

![alt text][image1]

My initial version did not account for cv2.imread using BGR representation causing some significant delay.

The steering angle adjustment value of .2/-.2 for the left and right images was found with trial and error.  It seems to be about correct, but without a method to *score* a lap around the track it is somewhat subjective.

If the `--augmentbase` option is selected, 25% of the images have a brightness adjustment made.  The adjustment is from .3 to 1.2 brightness.  Assuming the data is mostly "sunny" this biases towards darkening the image.  A set of 5 images with brightness(darness) augmentation below each:

![alt text][image2]

#### 4. Models

I used 3 models and multiple varients of each one.  The three were:

* lenet - it performed well and can drive around the track, steering is somewhat jerky.
* nvidia - This is the basic nvidia model covered in class.  It is the best overall performer.
* nvidia_1 - This is the nvidia model with L2 regularization on the fully connected layers.  It performs well, seems to smooth driving somewhat but has larger excursions.
* nvidia_2 - This is the nvidia model with significantly reduced fully connected layers.  I had expected improved training times with the reduced parameters, but the times were unchanged.  The model was slower to converge to a similar loss value.  It is more unstable in certain turns.

##### Model details and layers

* lenet

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 61, 316, 6)    456         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 30, 158, 6)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 26, 154, 6)    906         maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 13, 77, 6)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 6006)          0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 120)           720840      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 120)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 84)            10164       dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 84)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 20)            1700        dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             21          dense_3[0][0]                    
====================================================================================================
Total params: 734,087
Trainable params: 734,087
```

* nvidia
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

* nvidia_1 - I settled with the same model as above with a parameter of `W_regularizer = l2(0.001)` on all four Dense layers.

```
nvidia_1 has the same structure as the nvidia model above.
```

#### 5. Learning rate

The adam optimizer was chosen with the default parameters.  The last project of classifying german traffic
signs strongly suggested a decaying/adaptive learning rate would be useful.

## Conclusion

My car is able to reliably complete the circuit of the track at the default speed.  My key difficulty
does appear to be collecting good data that will guide the training of the model in specific situations.

The model changes I made did not improve the performance perceptibly over the nvidia model described
in the class lectures.  To be able to compare models, I feel it requires a metric to more reliably
say the model A is better than model B.  Human observation of a single lap or few laps does not make 
for a robust comparison.


