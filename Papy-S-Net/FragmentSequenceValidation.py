import keras
import numpy as np

import skimage.io
import skimage.transform

class FragmentSequenceValidation(keras.utils.Sequence):
    """
    This class is used as a sequence that gives mini-batches of preprocessed images. When taking a crop, the crop is always taken on the center of the image. 
    """
    
    def __init__(self, setInput, setOutput, batchSize, widthImage, heightImage, prefixPath):
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
        """
        
        self.setInput = setInput
        self.setOutput = setOutput
        self.batchSize = batchSize
        self.widthImage = widthImage
        self.heightImage = heightImage
        self.prefixPath = prefixPath
    
    
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
               
        batchInputTransformed1 = []
        batchInputTransformed2 = []

        i = 0
        while i < len(batchInputReturned):

            picture1 = batchInputReturned[i][0]
            picture2 = batchInputReturned[i][1]
                
            ############
            # Resize images if needed
            ############

            
            #Picture 1                
            if ((picture1.shape[0] - self.heightImage) < 0) and ((picture1.shape[1] - self.widthImage) < 0):
                picture1 = skimage.transform.resize(picture1, (self.heightImage, self.widthImage))
            
            elif (picture1.shape[0] - self.heightImage) < 0:
                picture1 = skimage.transform.resize(picture1, (self.heightImage, picture1.shape[1]))
            
            elif (picture1.shape[1] - self.widthImage) < 0:
                picture1 = skimage.transform.resize(picture1, (picture1.shape[0], self.widthImage))
                       
            #Picture 2                
            if ((picture2.shape[0] - self.heightImage) < 0) and ((picture2.shape[1] - self.widthImage) < 0):
                picture2 = skimage.transform.resize(picture2, (self.heightImage, self.widthImage))
            
            elif (picture2.shape[0] - self.heightImage) < 0:
                picture2 = skimage.transform.resize(picture2, (self.heightImage, picture2.shape[1]))
            
            elif (picture2.shape[1] - self.widthImage) < 0:
                picture2 = skimage.transform.resize(picture2, (picture2.shape[0], self.widthImage))
            

            ############
            # Preprocess and center crop
            ############

            
            #Picture 1
            picture1 = np.expand_dims(picture1, axis=0)
            #Should be replaced by the appropriate network-specific preprocessing function
            #picture1 = keras.applications.resnet50.preprocess_input(picture1)
            picture1 = picture1[0]
                       
            indexY = (picture1.shape[0] - self.heightImage) // 2
            indexX = (picture1.shape[1] - self.widthImage) // 2
            
            picture1 = picture1[indexY:(indexY + self.heightImage), indexX:(indexX + self.widthImage), :]

            #Picture 2
            picture2 = np.expand_dims(picture2, axis=0)
            #Should be replaced by the appropriate network-specific preprocessing function
            #picture2 = keras.applications.resnet50.preprocess_input(picture2)
            picture2 = picture2[0]
                        
            indexY = (picture2.shape[0] - self.heightImage) // 2
            indexX = (picture2.shape[1] - self.widthImage) // 2
            
            picture2 = picture2[indexY:(indexY + self.heightImage), indexX:(indexX + self.widthImage), :]
            
            
            batchInputTransformed1.append(picture1)
            batchInputTransformed2.append(picture2)
            
            i += 1
        
        
        batchInputTransformed1 = np.array(batchInputTransformed1)
        batchInputTransformed2 = np.array(batchInputTransformed2)
        
        
        batchOutputReturned = np.array(batchOutput)
        
        
        return [batchInputTransformed1, batchInputTransformed2], batchOutputReturned