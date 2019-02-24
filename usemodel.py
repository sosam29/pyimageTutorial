from nncopy.preprocessing.image2arrpreprocessor import ImageToArrayPreProcessor
from nncopy.preprocessing.simplepreprocessor import SimplePreprocessor
from nncopy.datasets import SimpleDatasetLoader
from imutils import paths
import numpy as np
from keras.models import load_model
import cv2

labelname = ['cat', 'dog','panda']
imagePaths = np.array(list(paths.list_images(r'animals/'))) # path to load images

# print(imagePaths)
indx = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[indx]


sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreProcessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)

# print(data)
data = data.astype("float")/255.0

model = load_model('animal_weights.hdf5')


pred = model.predict(data, batch_size= 32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(labelname[pred[i]]), (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)



