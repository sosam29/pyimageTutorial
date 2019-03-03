from nncopy.conv.lenet import LenNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from keras import backend as K 
import numpy as np 
import pandas as pd 
from mnist import MNIST

mndata = MNIST('/Users/samuelsonawane/Downloads/ML_TensorFlow/mnist_data_full/samples')

X1, y1 = mndata.load_training()
X2, y2 = mndata.load_testing()

# y1 = np.array(y1, dtype=np.int)
# y2 = np.array(y2, dtype=np.int)

X = X1 + X2
# y = np.concatenate(y1, y2)
y = y1 + y2

# X = pd.concat(X1, X2)
# y = pd.concat(y1, y2)

# print(X1.shape)
# print(y1.shape)
# print(X2.shape)
# print(y2.shape)
# X1= [[float(x/255.0) for  x in i ] for i in X1[:10]]
# X2= [[float(x/255.0) for  x in i ] for i in X2[:10]]
X = np.array(X)
y = np.array(y)
if K.image_data_format() =='channels_first':
    X= X.reshape(X.shape[0], 1, 28, 28)
else:
    X= X.reshape(X.shape[0],28, 28, 1)
    
(trainX, testX, trainY, testY) = train_test_split(X/255.0, y.astype(int), test_size=0.25, random_state=42)

# X1= [[float(x/255.0) for  x in i ] for i in X1]
# X2= [[float(x/255.0) for  x in i ] for i in X2]

lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

opt = SGD(lr=0.01)
# error expected here as we need to reshape data
model = LenNet.build(28, 28, 1, 10)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)
model.fit(trainX, trainY, validation_data=(testX,testY), batch_size=128, epochs=40, verbose=1)


pred = model.predict(testY, batch_size=128)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))