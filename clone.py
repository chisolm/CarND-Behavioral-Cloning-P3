import csv
import numpy as np
import cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Cropping2D
from keras.regularizers import l2

#prefixlist = ['data']
prefixlist = []

parser = argparse.ArgumentParser(description='Process flags.')
parser.add_argument('--epochs', type=int, default=7, help='Number of epochs to run')
parser.add_argument('--prefix', action='append', default=prefixlist, help='prefix name of a data dir and model.')
parser.add_argument('--display', action="store_true", help='Display pretty graphics')
parser.add_argument('--model', default='nvidia', help='Model to use.  default: nvidia')
parser.add_argument('--generator', action='store_true', default=False, help='User a generator for data and validation. default: False')
parser.add_argument('--batch', type=int, default=128, help='Batch size. default: 128')
parser.add_argument('--augmentbase', action='store_true', default=False, help='Augment the data')

args = parser.parse_args()

# Print out basic config params for logging
print("prefix ", args.prefix)
print("prefix type ", type(args.prefix))
print("batch ", args.batch)
print("augmentbase ", args.augmentbase)
print("generator: ", str(args.generator))
print("model: ", args.model)
print("epochs: ", args.epochs)

if len(args.prefix) == 0:
    print("Must supply a location for training data")
    exit()

images = []
measurements = []

def read_driving_log(lines, prefix):
    with open(prefix + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def add_img(image, measurement):
    images.append(image)
    measurements.append(measurement)

def load_images_adjust_lrc_flip(lines, prefix):
    for line in lines:
        if line[3] == 'steering':
            continue
        for i, steering_adjustment in zip([0, 1, 2], [0, 0.2, -0.2]):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = prefix + '/IMG/' + filename
            image = cv2.imread(current_path)
            if image is None: 
                print("Something broke")
                print(line)
                print(source_path)
                print(type(image))
                print("len images", len(images))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            measurement = float(line[3]) + steering_adjustment
            add_img(image, measurement)
            if i == 0:
                image_flipped = np.fliplr(image)
                measurement_flipped = -measurement
                add_img(image_flipped, measurement_flipped)
    return lines


def brightness_adjust(image):
    # Change to HSV colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Assumming sunny input generate brightness change, slight brighter and 
    # significantly darker
    rand = random.uniform(0.3,1.2)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

def augment_data(image):
    # 50% of the time augment the image
    if random.choice([0, 1]) == 1:
        return image 
    else:
        new_img = brightness_adjust(image)
        return new_img 


def augment_base_data(images, measurements):
    for i, image in enumerate(images):
        if i % 4 != 0:
            continue
        new_img = brightness_adjust(image)
        add_img(new_img, measurements[i])
    return images, measurements

# Time to make the donuts, er the models.

# Common model elements
def create_preprocessing_layers(model):
    model.add(Cropping2D(cropping=((75,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255 - 0.5))
    return model

# Lenet
def create_lenet_model(model):
    model = create_preprocessing_layers(model)
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.summary()

# Nvidia as described in the lecture
def create_nvidia_model(model):
    model = create_preprocessing_layers(model)
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

# Nvidia with l2 regularization
def create_nvidia_model_alt1(model):
    model = create_preprocessing_layers(model)
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', W_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', W_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu', W_regularizer = l2(0.001)))
    model.add(Dense(1, W_regularizer = l2(0.001)))
    model.summary()

# Significantly reduced Nvidia - modification from nvidia as described in the lecture
def create_nvidia_model_alt2(model):
    model = create_preprocessing_layers(model)
    model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()

# Can I use variable scope to combine the 2 generators?

def data_generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    s = features.shape
    fshape = (batch_size, s[1], s[2], s[3])
    print("train batch shape", fshape)
    batch_features = np.zeros(fshape)
    batch_labels = np.zeros((batch_size,1))
    while True:
        #features, labels = shuffle(features, labels)
        for offset in range(0, batch_size, batch_size):
            batch_features = features[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            for i in range(batch_size):
                batch_features[i] = augment_data(batch_features[i])
        yield batch_features, batch_labels

def valid_generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    s = features.shape
    fshape = (batch_size, s[1], s[2], s[3])
    print("valid batch shape", fshape)
    valid_features = np.zeros(fshape)
    valid_labels = np.zeros((batch_size,1))
    while True:
        #features, labels = shuffle(features, labels)
        for offset in range(0, batch_size, batch_size):
            valid_features = features[offset:offset+batch_size]
            valid_labels = labels[offset:offset+batch_size]
            for i in range(batch_size):
                valid_features[i] = augment_data(valid_features[i])
        yield valid_features, valid_labels

# Load the data from multiple directories assumed to be in the current directory
prefixconcat = ""
for prefix in args.prefix:
    print("processing data set: ", prefix)
    lines = []
    lines = read_driving_log(lines, prefix) 
    lines = load_images_adjust_lrc_flip(lines, prefix)
    prefixconcat = prefixconcat + "." + prefix

import re
prefixconcat = re.sub(r'^\.' , "", prefixconcat)


print("prefix label ", prefixconcat)
# If spcified on the command line augment the data, currently brightness augmentation
if args.augmentbase:
    images, measurements = augment_base_data(images, measurements)

# Conver data from lists for use in tensorflow 
X_train = np.array(images)
y_train = np.array(measurements)

# If using the generator split the data, if not using the generator the code will
# use the built in split in model.fit().
if args.generator:
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print ("shape X", X_train.shape, "shape y", y_train.shape)

# Display of data only performed on a local machine to validate some data during development
if args.display:
    display_n = 5
    plt.figure(figsize=(5, 5))

    for i in range(display_n):
        index = random.randint(0, len(X_train))
        image = X_train[index].squeeze()
        ax = plt.subplot(1, 5, i + 1)
        plt.imshow(image)
        ax.set_title(str(y_train[index]))
        
    plt.show()

model = Sequential()

# Choose model to use for this run 
if args.model == 'lenet':
    create_lenet_model(model)
elif args.model == 'nvidia':
    create_nvidia_model(model)
elif args.model == 'nvidia_1':
    create_nvidia_model_alt1(model)
elif args.model == 'nvidia_2':
    create_nvidia_model_alt2(model)
else:
    print("unknown model: ", args.model)
    exit()

model.compile(loss='mse', optimizer='adam')

if args.generator:
    data_generator = data_generator(X_train, y_train, args.batch)
    valid_generator = valid_generator(X_valid, y_valid, args.batch)

    model.fit_generator(data_generator, samples_per_epoch = len(X_train), nb_epoch=args.epochs,
                        validation_data = valid_generator, nb_val_samples = len(X_valid))
else:
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=args.epochs)

# Save the model, naming it with data used and model type

model.save(prefixconcat + '.' + args.model + '.model.h5')
