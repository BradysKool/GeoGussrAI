import torch.nn as nn
from transformers import CLIPModel
import torch.nn.functional as F

# This is to add final layer to the Model so that it can be used for my task
class CLIPGeoguessrModel(nn.Module):
    def __init__(self, modelName, numCountries):
        super().__init__()
        
        # Load the pre-trained CLIP model
        self.clipModel = CLIPModel.from_pretrained(modelName)
        
        #Freeze the weights so they dont get ruined during traing
        for param in self.clipModel.parameters():
            param.requires_grad = False
            
        
        #2 extra layers that I will be traing in the RL
        self.dense1 = nn.Linear(768, 512) # The CLIP outputs dim is 768
        self.dense2 = nn.Linear(512, numCountries) 
        

    def forward(self, x):
        x = self.clipModel.get_image_features(pixel_values=x)#CLIP has both a image and text models so need to specifi which one im using
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
    
        return x