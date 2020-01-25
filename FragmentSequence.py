import keras
import numpy as np

import skimage.io
import skimage.transform

class FragmentSequence(keras.utils.Sequence):
    """
    This class is used as a sequence that gives mini-batches of preprocessed images. 
    """
    
    def __init__(self, setInput, setOutput, batchSize, widthImage, heightImage, prefixPath, probabilityCrop, probabilityHorizontalFlip, probabilityVerticalFlip):
        """
        This is the initialization method.
        
        parameters:
        -----------
        - setInput: The input set.
        - setOutput: The output set.
        - batchSize: The size of the batch.
        - widthImage: The width of the image.
        - heightImage: The height of the image.
        - prefixPath: The path to give as prefix for every path stored in setInput.
        - probabilityCrop: The probability to make a crop as reduction operation. If a crop is not done, then a resize operation is done.
        - probabilityHorizontalFlip: The probability to flip horizontally each image.
        - probabilityVerticalFlip: The probability to flip vertically each image.
        """
        
        self.setInput = setInput
        self.setOutput = setOutput
        self.batchSize = batchSize
        self.widthImage = widthImage
        self.heightImage = heightImage
        self.prefixPath = prefixPath
        self.probabilityCrop = probabilityCrop
        self.probabilityHorizontalFlip = probabilityHorizontalFlip
        self.probabilityVerticalFlip = probabilityVerticalFlip
    
    
    def __len__(self):
        """
        This method returns the total number of mini-batches in the sequence.
        
        returns:
        --------
        - The total number of mini-batches in the sequence.
        """
        
        length = int(np.ceil(len(self.setInput) / float(self.batchSize)))
        
        return length
        
    
    def __getitem__(self, index):
        """
        This method returns a mini-batch given the index provided.
        
        parameter:
        ----------
        - index: The index of the mini-batch.
        
        returns:
        --------
        - A mini-batch given the index provided.
        """
        
        batchInput = self.setInput[(index * self.batchSize):(index * self.batchSize + self.batchSize)]
        batchOutput = self.setOutput[(index * self.batchSize):(index * self.batchSize + self.batchSize)]
        

        batchInputReturned = [[skimage.io.imread(self.prefixPath + filePath[0]), skimage.io.imread(self.prefixPath + filePath[1])] for filePath in batchInput]
        
        imageGenerator = keras.preprocessing.image.ImageDataGenerator()
        
        batchInputTransformed1 = []
        batchInputTransformed2 = []

        i = 0
        while i < len(batchInputReturned):
            picture1 = batchInputReturned[i][0]
            picture2 = batchInputReturned[i][1]
            randomNumber = np.random.random()
            
            if randomNumber < self.probabilityHorizontalFlip:
                picture1 = imageGenerator.apply_transform(picture1, {"flip_horizontal" : True})
                picture2 = imageGenerator.apply_transform(picture2, {"flip_horizontal" : True})
            
            elif randomNumber < (self.probabilityHorizontalFlip + self.probabilityVerticalFlip):
                picture1 = imageGenerator.apply_transform(picture1, {"flip_vertical" : True})
                picture2 = imageGenerator.apply_transform(picture2, {"flip_vertical" : True})
            
            
            
            randomNumberCrop = np.random.random()
            
            #Picture 1
            if randomNumberCrop < self.probabilityCrop:
                
                # Sometimes, images are too small and must be resized in order to take a crop of the wanted dimension.
                if ((picture1.shape[0] - self.heightImage) < 0) and ((picture1.shape[1] - self.widthImage) < 0):
                    picture1 = skimage.transform.resize(picture1, (self.heightImage, self.widthImage))
                
                elif (picture1.shape[0] - self.heightImage) < 0:
                    picture1 = skimage.transform.resize(picture1, (self.heightImage, picture1.shape[1]))
                
                elif (picture1.shape[1] - self.widthImage) < 0:
                    picture1 = skimage.transform.resize(picture1, (picture1.shape[0], self.widthImage))
            
            
            else:
                picture1 = skimage.transform.resize(picture1, (self.heightImage, self.widthImage))
            
            #Picture 2
            if randomNumberCrop < self.probabilityCrop:
                
                # Sometimes, images are too small and must be resized in order to take a crop of the wanted dimension.
                if ((picture2.shape[0] - self.heightImage) < 0) and ((picture2.shape[1] - self.widthImage) < 0):
                    picture2 = skimage.transform.resize(picture2, (self.heightImage, self.widthImage))
                
                elif (picture2.shape[0] - self.heightImage) < 0:
                    picture2 = skimage.transform.resize(picture2, (self.heightImage, picture2.shape[1]))
                
                elif (picture2.shape[1] - self.widthImage) < 0:
                    picture2 = skimage.transform.resize(picture2, (picture2.shape[0], self.widthImage))
            
            
            else:
                picture2 = skimage.transform.resize(picture2, (self.heightImage, self.widthImage))
            
            #Picture 1
            picture1 = np.expand_dims(picture1, axis=0)
            #Should be replaced by the appropriate network-specific preprocessing function
            picture1 = keras.applications.resnet50.preprocess_input(picture1)
            picture1 = picture1[0]
            
            if randomNumberCrop < self.probabilityCrop:
                randomY = np.random.randint(0, picture1.shape[0] - self.heightImage + 1)
                randomX = np.random.randint(0, picture1.shape[1] - self.widthImage + 1)
                
                picture1 = picture1[randomY:(randomY + self.heightImage), randomX:(randomX + self.widthImage), :]

            #Picture 2
            picture2 = np.expand_dims(picture2, axis=0)
            #Should be replaced by the appropriate network-specific preprocessing function
            picture2 = keras.applications.resnet50.preprocess_input(picture2)
            picture2 = picture2[0]
            
            if randomNumberCrop < self.probabilityCrop:
                randomY = np.random.randint(0, picture2.shape[0] - self.heightImage + 1)
                randomX = np.random.randint(0, picture2.shape[1] - self.widthImage + 1)
                
                picture2 = picture2[randomY:(randomY + self.heightImage), randomX:(randomX + self.widthImage), :]
            
            
            batchInputTransformed1.append(picture1)
            batchInputTransformed2.append(picture2)
            
            i += 1
        
        
        batchInputTransformed1 = np.array(batchInputTransformed1)
        batchInputTransformed2 = np.array(batchInputTransformed2)
        
        
        batchOutputReturned = np.array(batchOutput)
        
        
        return [batchInputTransformed1, batchInputTransformed2], batchOutputReturned