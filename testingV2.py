import torch
import torch.nn.functional as F
import countrysListV2
import functions
import cv2
import keyboard
import clipModel
from transformers import CLIPProcessor
from PIL import Image

clipModelName = "openai/clip-vit-large-patch14"
model = clipModel.CLIPGeoguessrModel(modelName=clipModelName, numCountries=222)
processor = CLIPProcessor.from_pretrained(clipModelName)
device = torch.device("cuda")
model.to(device)

model.load_state_dict(torch.load("datasetModelWeights2.pth")) # This loads the weights from the training

for i in range(int(input('How many countrys do you wanna guess: '))):
    print("press tab to continue")  
    keyboard.wait('tab')
    
    functions.fullScreenShot()
    image = cv2.imread('screenshot.png')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #form to be able to convert to pytorch

    pilImage = Image.fromarray(image)

    inputs = processor(images=pilImage,return_tensors = 'pt')
    pyTensorImage = inputs['pixel_values'].to(device)

    model.eval()# We are just evlauting in this script not traing
    with torch.no_grad():
        logits = model(pyTensorImage)
        probibiltys = F.softmax(logits,dim=1)

    guessdCountryIndex = torch.argmax(probibiltys,dim=1).item() 

    guessdCountryISO = countrysListV2.countries[guessdCountryIndex]
    gussedContryName = countrysListV2.country_codes_to_names_updated[guessdCountryISO]
    functions.makeguess(gussedContryName)

    