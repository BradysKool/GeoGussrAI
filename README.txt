Goal: Build a bot that can beat me a geogussr
Outcome: The bot beat me and can get around 45% of its gusses correct

for clarficication this model will be feed one screenshot of some locaiton on google streatview and it has to gusse the contry that it was taken
I am also in top 1% of geogussr players so beating me is quite impressive

There where 3 verisons to the model:
V1- This version had torchModel.py and traning.py as the model and traing algoritham respectivly.

Limitations: with this model was that it was building a whole image reconition model from scratch which would have taken quite a long time.
Also it was getting its traing data as it was playing on geogussrs website which is oftain pretty glitchy


V2- This version had clipModel.py and clipModelTraing.py as the model and traing algoritham respectivly

Updates: In this verision I was fintuning the open ai CLIP model as the main model
I also added a diffrent reward function to go based off distance and added a tempature annealing function to stablize it

Limitations: It was still runnig through geogussrs website to make the traing data and I ened up getting banned for 2 days.


V3- This verison had clipModel.py and datasetTraing.py as the model and traing algoratham respectivly.

Updates: In this model I am now using a data set to greatly increase the speed and stabilty of traing.
I also updated the reward function to take full advatage of the millions of images that it had.

Limitations: This data set did not include any of the new countrys that where added to geogussr like Namiba and Costa Rica so it got these wrong everytime


future goals: I want to tweak the model and add some of my own data so that I can get the accurcy up to 75%