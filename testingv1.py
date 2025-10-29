import torch
import torch.nn.functional as F
import countryLists
import functions
import cv2
import time
import torchModel

model = torchModel.Countryguessr(numCountrys=len(countryLists.countries))
print(model)
device = torch.device("cuda")
model.to(device)

model.load_state_dict(torch.load("modelWeights.pth")) # This loads the weights from the training

print('running in 2 seconds')
time.sleep(2)
functions.fullScreenShot()
image = cv2.imread('screenshot.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #form to be able to convert to pytorch

pyTensorImage = torch.from_numpy(image).permute(2,0,1)
pyTensorImage = pyTensorImage.float()/255.0
pyTensorImage = pyTensorImage.unsqueeze(0)# converted to torchTensor as welll as normalized the data
pyTensorImage = pyTensorImage.to('cuda')

model.eval()# We are just evlauting in this script not traing
with torch.no_grad():
    logits = model(pyTensorImage)
    probibiltys = F.softmax(logits,dim=1)

guessdCountryIndex = torch.argmax(probibiltys,dim=1).item() 

guessdCountry = countryLists.countries[guessdCountryIndex]

print(guessdCountry)