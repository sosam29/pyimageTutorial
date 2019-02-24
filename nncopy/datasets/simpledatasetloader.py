import numpy as np 
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors= None):
        self.preprocessors= preprocessors

        if self.preprocessors is None:
            self.preprocessors=[]

    def load(self, imagepaths, verbose =-1):
        data=[]
        labels=[]

        for(i, imagepath) in enumerate(imagepaths):
            image = cv2.imread(imagepath)
# labels are nothing but folers in the datasets (last but one part)
            label= imagepath.split(os.path.sep)[-2]
# Process image based on arrays of preprocessor so as to make it processeable for next steps
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
# addung image to data and label list            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] Processed {}/{}".format(i+1, len(imagepath)))
            
        return (np.array(data), np.array(labels))
            
