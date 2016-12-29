import h5py
import numpy
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.datasets import cifar10
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


# path to the VGG16 weights files.
weights_path = '../vgg16_weights.h5'

# dimensions of our images.
img_width, img_height = 32, 32

# build the VGG16 network since we can't import it
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# build a classifier model (a fully connected layer) to put on top of the VGG16 model
# use weight normalization with aggressive dropout 40%
top_model = Sequential()
# use input shape of last convolution block from VGG16
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
top_model.add(Dropout(0.4))
top_model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
top_model.add(Dropout(0.4))
top_model.add(Dense(num_classes, activation='softmax'))
# stick it to the bottom of VGG16
model.add(top_model)


# save us pictures of both
print("FC layers visualisation saved to fc.png")
plot(top_model, to_file='fc.png', show_shapes=True, show_layer_names=True)

print("Model visualisation saved to vgg16.png")
plot(model, to_file='vgg16.png', show_shapes=True, show_layer_names=True)


# freeze all layers except last convolution block and our fully connected layer
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with loss function for categories, use Adadelta and show us accuracy for each epoch
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# print us a summary of model
print(model.summary())


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