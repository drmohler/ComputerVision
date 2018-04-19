# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D 
from keras.layers.core import Activation
from keras.layers.core import Flatten,Dropout
from keras.layers.core import Dense
from keras import regularizers
from keras import backend as K

class MohlerNet1: #AKA ShallowNet 
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        
        
        return model

#ADD CONVOLUTION LAYER, POOLING LAYER, AND DROPOUT
class MohlerNet2:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", activation='relu',
            input_shape=inputShape))
        #model.add(Activation("relu"))
        
        #ADD CONVOLUTION LAYER, POOLING LAYER, AND DROPOUT
        model.add(Conv2D(64,(3,3),padding="same",activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #model.summary()
        # return the constructed network architecture
        return model

class MohlerNet3:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(64, (3, 3), padding="same", activation='relu',
            input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        #ADD CONVOLUTION LAYER, POOLING LAYER, AND DROPOUT
        model.add(Conv2D(128,(3,3),padding="same",activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3),padding="same",activation='relu'))
        model.add(Conv2D(32,(3,3),padding="same",activation='relu'))
        model.add(Conv2D(32,(3,3),padding="same",activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Dropout(0.5))
        model.add(Activation("softmax"))
        #model.summary()
        # return the constructed network architecture
        return model


class MohlerNet4:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", activation='relu',
            input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Activation("relu"))
        
        #ADD CONVOLUTION LAYER, POOLING LAYER, AND DROPOUT
        model.add(Conv2D(32,(3,3),padding="same",activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(64,(3,3),padding="same",activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("tanh"))
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #model.summary()
        # return the constructed network architecture
        return model
