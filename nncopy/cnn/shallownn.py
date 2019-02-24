from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K 
import keras


class ShallowNN:
    @staticmethod
    def build(width, height, depth, classes):
        model = keras.Sequential()
        inputShape= (height, width, depth)

        if K.image_data_format()=="channels_sfirst":
            inputShape=(depth, height, width)
    
        model.add(Conv2D(32, (3,3),padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model



