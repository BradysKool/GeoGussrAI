Goal: Build a bot that can beat me a GeoGuessr
Outcome: The bot beat me and can get around 45% of its guesses correct

To play the game, the model will be fed one screenshot of some location on google street view, and it has to guess the country in which it was taken.  Since I’m in the top 1% of GeoGuessr players, the fact that it is beating me consistently now is quite impressive.

Check timeline.txt for the work that was done each day. 

Through the course of the project, I built 3 versions of the model:
V1- This version had torchModel.py and traning.py as the model and training algorithm respectively.

Limitations: With this model, it was building an entire image recognition model from scratch, which would have taken quite a long time. Also, it was getting it’s training data as it was playing on GeoGuessr’s website which is often pretty glitchy

V2- This version had clipModel.py and clipModelTraing.py as the model and training algorithm respectively.

Updates: In this version I was fine tuning the OpenAI CLIP model as the main model and
I also added a different reward function which was based off distance, and O added a temperature annealing function to stabilize it.

Limitations: It was still running through GeoGuests website to make the training data and I ended up getting banned for 2 days because of the volume and speed of gameplay. 


V3- This version had clipModel.py and datasetTraing.py as the model and training algorithm respectively.

Updates: In this model I am now using a data set to greatly increase the speed and stability of training. I also updated the reward function to take full advantage of the millions of images that it had.

Limitations: This data set did not include any of the new countries that where added to GeoGuessr, like Namibia and Costa Rica so it got these wrong every time.


Future Goal: I want to update the model and add some of my own data so that I can get the accuracy up to 75%.


