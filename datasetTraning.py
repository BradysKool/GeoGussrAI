import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from transformers import CLIPProcessor
import clipModel
from datasets import load_dataset
import time

batchSize = 256 
learingRate = 1e-5 
epochs = 1 
weightDecay = 0.01 
clipModelName = "openai/clip-vit-large-patch14"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Step 1: loading dataset and preparing labels")


rawData = load_dataset("osv5m/osv5m", split="train") 
countryList = sorted(list(set(rawData['country'])))
contrysToIndex = {}
for i in range(len(countryList)):
    contrysToIndex[countryList[i]] = i
numCountrys = len(countryList)


class OpenStreetViewDataset(Dataset):
    #class to handle the image loading and processing
    def __init__(self, raw_data, processor, country_map):
        self.raw_data = raw_data
        self.processor = processor
        self.country_map = country_map

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        example = self.raw_data[idx]
        
        try:
            image = example['image'].convert("RGB")
        except:
            #if we cant upload/convert image to rgb then just retrn blank vector in its place
            print(f"Couldntload image at index {idx}")
            return torch.zeros(3, self.processor.image_processor.size['height'], self.processor.image_processor.size['width']), torch.tensor(0, dtype=torch.long)
        
        country_label = example['country'] #extracts correct country
        label = self.country_map[country_label]
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pixel_values, label_tensor


processor = CLIPProcessor.from_pretrained(clipModelName)
model = clipModel.CLIPGeoguessrModel(modelName=clipModelName, numCountries=numCountrys)
model.to(device)
model.load_state_dict(torch.load("datasetModelWeights.pth"))#for the second batch of traning will use weights from first round

dataset = OpenStreetViewDataset(rawData, processor, contrysToIndex)
dataloader = DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True, # to avoid convergence based on order
    num_workers=0, # for no multiprocessing(tends to cause erros on my pc)
    pin_memory=True #  for CUDA acceleration
)

optimizer = optim.AdamW(model.parameters(), lr=learingRate, weight_decay=weightDecay)
criterion = nn.CrossEntropyLoss()

totalSameples = len(dataset)
stepsPer = len(dataloader) # The DataLoader calculates this
print(f"Total samples: {totalSameples}. Steps per epoch: {stepsPer}")

for epoch in range(epochs):
    model.train()
    totalLoss = 0
    correctPredictions = 0
    start_time = time.time()
    
    for step, (pixel_values, labels) in enumerate(dataloader):
        if step>= stepsPer // 2:
            print("reached a half of an epoch ")
            break #It takes 20 hours to reach 1/2 th of an epoch so I will just cut it there
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(pixel_values) 

        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        totalLoss += loss.item()
        
        # Calculate Accuracy
        _, predicted = torch.max(logits.data, 1)
        correctPredictions += (predicted == labels).sum().item() #add 1 to corrcet each time the perdeciton was correct
        #Had Gemini make this to calculate time left
        if (step + 1) % 50 == 0:
            avg_batch_loss = totalLoss / (step + 1)
            # The accuracy calculation needs to use the total number of samples processed so far
            total_processed_samples = batchSize * (step + 1) 
            accuracy = correctPredictions / total_processed_samples
            
            # Estimate remaining time for this epoch
            elapsed_time = time.time() - start_time
            time_per_step = elapsed_time / (step + 1)
            remaining_steps = stepsPer - (step + 1)
            eta_seconds = remaining_steps * time_per_step
            eta_minutes = eta_seconds / 60
            
            print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{stepsPer} | Loss: {avg_batch_loss:.4f} | Acc: {accuracy:.4f} | ETA: {eta_minutes:.1f} mins")
    avg_epoch_loss = totalLoss / stepsPer
    epoch_accuracy = correctPredictions / totalSameples
    print(f"\n--- Epoch {epoch+1} Complete ---")
    print(f"Average Loss: {avg_epoch_loss:.4f}, Total Training Accuracy: {epoch_accuracy:.4f}")
    
print("\nTraining complete. Saving weights.")
torch.save(model.state_dict(), 'datasetModelWeights2.pth')
