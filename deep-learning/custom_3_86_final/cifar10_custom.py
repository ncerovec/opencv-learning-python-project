import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.utils.visualize_util import plot
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('th')

# randomize just to be sure
seed = 7
numpy.random.seed(seed)

# load CIFAR-10 dataset and it lables
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert pixels to floats and normalize to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# one-of-K scheme for label encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# get number of classed
num_classes = y_test.shape[1]

# create a new image generator with necessary transformations
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
    channel_shift_range=0.2,
    horizontal_flip=True)

# create a flow from the data generator for training
train_generator = train_datagen.flow(X_train, y_train, batch_size=64)

# instantiate a model
model = Sequential()
# first build unit - 32
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# second build unit - 64
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# third build unit - 128
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# fourth build unit - 256
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# full connected layer
model.add(Flatten())
model.add(Dropout(0.2))
# weight normalization with max norm
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
# once again weight normalization
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
# output to classes with softmax
model.add(Dense(num_classes, activation='softmax'))

# store a picture for us to see
print("Model visualisation saved to custom.png")
plot(model, to_file='custom.png', show_shapes=True, show_layer_names=True)

# compile the model with loss function for categories, use Adadelta and show us accuracy for each epoch
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# print us a model summary
print(model.summary())

# just to be sure
numpy.random.seed(seed)

# train the model for 100 epochs using the train generator flow and training set size
# validate everything on untouched data and show us only final progress
epochs = 100
model.fit_generator(
    train_generator,
    samples_per_epoch=X_train.shape[0],
    nb_epoch=epochs,
    validation_data=(X_test, y_test),
    verbose=2)

# not needed - but nice to see
# check for the final score and print it
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
