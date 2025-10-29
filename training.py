import torchModel
import torch
import torch.nn.functional as F
import countryLists
import functions
import cv2
import torchvision.transforms as transforms
import time
import pyautogui

model = torchModel.Countryguessr(numCountrys=len(countryLists.countries))
optimizer =  torch.optim.Adam(model.parameters(),lr = .001)

device = torch.device("cuda")
model.to(device)

for i in range(10):
    time.sleep(2)
    functions.fullScreenShot()
    image = cv2.imread('screenshot.png')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #form to be able to convert to pytorch

    pyTensorImage = torch.from_numpy(image).permute(2,0,1)
    pyTensorImage = pyTensorImage.float()/255.0
    pyTensorImage = pyTensorImage.unsqueeze(0)# converted to torchTensor as welll as normalized the data
    pyTensorImage = pyTensorImage.to('cuda')

    model.eval()
    with torch.no_grad():
        logits = model(pyTensorImage)
        probibiltys = F.softmax(logits,dim=1)

    guessdCountryIndex = torch.argmax(probibiltys,dim=1).item()
    guessdCountry = countryLists.countries[guessdCountryIndex]
    print('\n')
    print(guessdCountry)
    reward = 0
    numCorrectguesss = 0 #Place holder to be able to see the accuracy of the modle

    functions.makeguess(guessdCountry)
    correctCountry = functions.checkCounty()
    print(correctCountry)

    if correctCountry != None:#function to calculate reward
        if guessdCountry == correctCountry:
            numCorrectguesss +=1
            reward += 10
        if functions.checkNeighbor(guessdCountry,correctCountry) == True:
            reward += 5
        if functions.checkContinent(guessdCountry,correctCountry) == True:
            reward +=1

        print(reward)

    else:# If something gose wrong reset it to a new round and go again
        print("no country found")
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'r')
        time.sleep(4)

    pyautogui.moveTo(1196,1388,duration=.1)
    pyautogui.click()    

    model.train()

    optimizer.zero_grad()
    logitsTrain = model(pyTensorImage)
    probTrain = F.softmax(logitsTrain,dim=1)

    logProb = torch.log(probTrain.squeeze(0)[guessdCountryIndex])

    loss = -logProb * reward
    loss.backward()

    optimizer.step()# this is the part where the model actully gets traine
    if i // 10:
        print(str(i/10)+ "%")
        print(str(numCorrectguesss*10)+'% Correct over the last 10')
        numCorrectguesss = 0
print('saved')
torch.save(model.state_dict(),'modelWeights.pth')