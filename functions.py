import pyautogui
import countryLists
import pytesseract
import time
import math

pyautogui.FAILSAFE = False

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

countries = countryLists.countries
continents = countryLists.continents
countryBorders = countryLists.countryBorders
countryLocation = countryLists.countryLocation
countryWithMultiWordUniqeWords = countryLists.countryWithMultiWordUniqeWords

def fullScreenShot():
    screenshot = pyautogui.screenshot()
    screenshot.save("screenshot.png")

def checkCounty():
    
    screenshot = pyautogui.screenshot(region=(689,1131,1262,226))
    screenshot.save("countyNameScreenShot.png")
    text = pytesseract.image_to_string(screenshot) #take words from image
    textList = text.split()
    countriesInText = []#This is a list of all of the county names that it sees in the image.
    
    for t in textList: 
        if t in countries or t in countryWithMultiWordUniqeWords:
            countriesInText.append(t)

    pyautogui.moveTo(1203,1384,duration=.1)#press the play agian button
    pyautogui.click()
    
    #The last country in the list will be the correct one so its checking if the last county is not multi-word
    try:
        if countriesInText[-1] in countries:
            return(countriesInText[-1])
        elif " ".join([countriesInText[-2],countriesInText[-1]]) in countries:
            return " ".join([countriesInText[-2],countriesInText[-1]])
        elif " ".join([countriesInText[-3],countriesInText[-2],countriesInText[-1]]) == "United Arab Emirates": # Only 3 word country in list in UAE 
            return "United Arab Emirates"
    except:
        return None
        
def checkContinent(guessCountry, correctCountry): # check if the guess was in the same continent to help give reward to model 
    for continet, countries_list in continents.items():
        if guessCountry in countries_list:
            guessContinet = continet
        if correctCountry in countries_list:
            correctContinet = continet
    if guessContinet == correctContinet:
        return True
    else: 
        return False

def checkNeighbor(guessCountry, correctCountry):
    if correctCountry in countryBorders[guessCountry]:
        return True
    else:
        return False
    
def makeguess(country):
    list = countryLocation[country]
    pyautogui.moveTo(2261,1208,duration=.2)#Just to open up the map
    time.sleep(.1)
    if len(list) == 1: # IF I dont need to zoom in
        pyautogui.moveTo(list[0][0],list[0][1],duration=.1)#Have to call a list in a list becaues of the double wrapping shanagians I have going on in the list
        pyautogui.click()
    else: # If I need to zoom in 
        pyautogui.moveTo(list[0][0],list[0][1],duration=.1)
        for _ in range (list[0][2]):
            pyautogui.doubleClick()
            time.sleep(.5)
        pyautogui.moveTo(list[1][0],list[1][1],duration=.1)
        pyautogui.click()
    time.sleep(.2)
    pyautogui.click(1958,1398)#This is the Guess button
    time.sleep(1.5)

def distanceBetweenCountrys(guessCountry,correctCountry):
    return math.sqrt((countryLists.countryLocation[guessCountry][0][0]-countryLists.countryLocation[correctCountry][0][0])**2 + (countryLists.countryLocation[guessCountry][0][1]- countryLists.countryLocation[correctCountry][0][1])**2)
