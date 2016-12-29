import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

train_datagen = ImageDataGenerator(
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=30,
    channel_shift_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_datagen.fit(X_train)
test_datagen.fit(X_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

train_generator = train_datagen.flow(X_train, y_train, batch_size=64)

validation_generator = test_datagen.flow(X_test, y_test, batch_size=64)

# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
#
# num_classes = y_test.shape[1]

model = Sequential()
model.add(
    Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(Dropout(0.2))
model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

epochs = 50
learning_rate = 0.01
decay = learning_rate/(epochs*2)
sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

numpy.random.seed(seed)

model.fit_generator(
    train_generator,
    samples_per_epoch=50000,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=300000,
    verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
