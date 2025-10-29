import clipModel
import torch
import torch.nn.functional as F
import countryLists
import functions
import torchvision.models as models
import time
import pyautogui
from PIL import Image
from transformers import CLIPProcessor


iterations = 1000#The amount of iterations that  will be used to train

clipModelName = "openai/clip-vit-large-patch14" #This is just the specific model I am using
processor = CLIPProcessor.from_pretrained(clipModelName)
model = clipModel.CLIPGeoguessrModel(modelName=clipModelName,numCountries=len(functions.countries))
optimizer = torch.optim.Adam(model.parameters(),lr=.001)
alpha = .99 #used for reward function
runningRewardMean = 0

initalTemp = 2.0 #exploration(initial temp)
finalTemp = 1.0 # no exploration
annealingSteps = 800
T = initalTemp

device = torch.device("cuda")
model.to(device)
print('starting in 5')
for i in range(iterations):
    time.sleep(5)
    functions.fullScreenShot()
    image = Image.open('screenshot.png').convert("RGB") #Original image of size 2560x1440
    inputs = processor(images=image,return_tensors="pt", padding=True) # Process the image so that it is able to be used by the model
    inputs['pixel_values'] = inputs['pixel_values'].to(device) #add it to my gpu to increase speed

    model.eval()
    with torch.no_grad():
        logitsEval = model(inputs['pixel_values'])
        tempLogits = logitsEval / T
        probibiltys = F.softmax(tempLogits,dim=1).squeeze(0)
    guessdCountryIndex = torch.multinomial(probibiltys,num_samples=1).item() 
    
    guessdCountry = countryLists.countries[guessdCountryIndex]

    reward = 0
    numCorrectguesss = 0 #Place holder to be able to see the accuracy of the modle

    functions.makeguess(guessdCountry)
    correctCountry = functions.checkCounty()
    if correctCountry != None:#function to calculate reward
        reward = 1 - (functions.distanceBetweenCountrys(guessdCountry,correctCountry)/853.3147)#853.3147 is the max distance
        
        if i == 0:
            runningRewardMean = reward
        else:
            runningRewardMean = alpha * runningRewardMean + (1-alpha) * reward
        advantage =  reward - runningRewardMean

        model.train()
        optimizer.zero_grad()

        logitsTrain = model(inputs['pixel_values'])
        probTrain = F.softmax(logitsTrain,dim=1)
        logProb = torch.log(probTrain.squeeze(0)[guessdCountryIndex])
        loss = -logProb * advantage

        loss.backward()
        optimizer.step()# this is the part where the model actully gets trained

        if i < annealingSteps:
            T = initalTemp -(i/annealingSteps) * (initalTemp-finalTemp)
        if T < finalTemp:
            T = finalTemp


    else:# If something gose wrong reset it to a new round and go again
        print("no country found")
        time.sleep(4)
        pyautogui.moveTo(1298,730,.1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.hotkey('ctrl','r')
        time.sleep(1)
        pyautogui.moveTo(1203,1384,duration=.1)#press the play agian button
        pyautogui.click()
        i -= 1
      
    if i % 10 == 0:
        print(str((i*100)/iterations)+ "% Completed")
        print(str(numCorrectguesss)+' correct over the last 10')
        numCorrectguesss = 0
    
print("saved")
torch.save(model.state_dict(),'modelWeights.pth')