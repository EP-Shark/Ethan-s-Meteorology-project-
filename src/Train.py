''' File that is used to actually train the machine to learn the features in non-tornadic versus tornadic cells. '''

#Section 1 - Importing packages
import sys
sys.path.append('/Users/epalmer/Ethan-s-Meteorology-project-')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.dataset import TorNETDatabase
from src.models.cnn import TornadoCNN
import numpy as np

#Section 2 - Configs
DATA_DIR = '/Users/epalmer/Ethan-s-Meteorology-project-/data/raw/train/2021' #Directory of the training data
CHECKPOINT_DIR = '/Users/epalmer/Ethan-s-Meteorology-project-/checkpoints' #Directory for checkpoint files
Batch_size = 32 #number of radar images fed into the system
EPOCHS = 3 # 3 (for inital macbook test) #number of times the dataset is passed through the machine 
Learning_rate = 0.001 #Rate of learning
Val_split = 0.2 # save 20% of the data to be used for validation
Device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') # Torch will use MPS for mac if available, otherwise uses the device CPU
print(f'Training on {Device}') # Prints what device the model is running on

#Section 3 - loading in the dataset
print('Loading dataset...') #prints a string to show the dataset is being loaded
full_dataset = TorNETDatabase(DATA_DIR) #Access the data to be loaded

#Train and validation
val_size = int(len(full_dataset) * Val_split) 
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset,[train_size, val_size])

print(f"Training samples:   {train_size}")
print(f"Validation samples: {val_size}")

# Creating the loader
train_loader = DataLoader(
    train_dataset, 
    batch_size = Batch_size,
    shuffle = True, #Shuffles data so the model doesnt just "learn" the order and actually need to learn the properties
    num_workers = 0 # Loads data into GPU in parallel so it doesnt sit idle
)

val_loader = DataLoader(
    val_dataset,
    batch_size = Batch_size,
    shuffle = False, #doesnt need to be shuffled
    num_workers = 0
)

# Section 4 - class weights
# there is about 13x more non-tornadic cell examples compared to tornadic cells in the 2021 torNET dataset, so in order to make the model
#actually learn we want to add a class weight to make identifing tornadic cells more satisfying to the machine learning algorithm. 

#not-perfect at the moment, will change to be a dynamic calculation of class weight, however hardcoded for simplicity atm
pos_weight = torch.tensor([5.0]).to(Device) # 5x weight 

#Section 5 - Model, loss, optimizer
#initialize model
model = TornadoCNN(in_channels=13).to(Device)

#loss function
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#Optimzer
Optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

#Scheduler
Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    Optimizer,
    mode = 'min',
    patience = 3,
    factor = 0.5
)

# Section 6 - Training loop

def train_epoch(model, loader, criterion, optimizer, device):
    model.train() #model being set into training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x,y) in enumerate(loader):
        #Move data to the GPU
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        #Forward Pass
        optimizer.zero_grad() #Clear old gradients
        outputs = model(x)  #get predictions
        loss = criterion(outputs,y) # calculate the loss

        #Backward Pass
        loss.backward() #Calc gradients
        optimizer.step() #Update weights

        #Track metrics
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == y).sum().item()
        total += y.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} "
                  f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss/len(loader) #Calculate the average loss
        accuracy = correct/total #How tors did the model get correct
        return avg_loss, accuracy 
    
#Section 7 - Validation loop 

def validate(model, loader, criterion, device):
    model.eval() #Set the model to evaluation mode
    total_loss = 0 #Inital loss, calculated later

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            predicted = (outputs > 0.5).float() 
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # See the max_output 
    print(f"  Output stats - Min: {outputs.min().item():.3f} | Max: {outputs.max().item():.3f} | Mean: {outputs.mean().item():.3f}")
    avg_loss = total_loss / len(loader)

    #Calculate MET characteristics
    all_preds = torch.tensor(np.array(all_preds))
    all_labels = torch.tensor(np.array(all_labels))

    TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
    FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
    FN = ((all_preds == 0) & (all_labels == 1)).sum().item()

    POD = TP / (TP + FN + 1e-8)   # Probability of Detection, what fraction did we catch? - The higher this number the better
    FAR = FP / (TP + FP + 1e-8)   # False Alarm Rate, OF all the detected tornados, how many were false alarms? Lower this number the better
    CSI = TP / (TP + FP + FN + 1e-8)  # Critical Success Index, combines both POD and FAR into one metric, higher is better

    return avg_loss, POD, FAR, CSI

#Section 8 - Main training loop

def main():
    best_csi = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        #Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, Optimizer, Device)

        #Validation 
        val_loss, POD, FAR, CSI = validate(model, val_loader, criterion, Device)

        #Update learning rate
        Scheduler.step(val_loss)

        #Epoch Summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"POD: {POD:.4f} | FAR: {FAR:.4f} | CSI: {CSI:.4f}")

        #Save the best model
        if CSI > best_csi:
            best_csi = CSI
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': Optimizer.state_dict(),
                'CSI': CSI,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            print(f"  ✓ New best model saved (CSI: {CSI:.4f})")

if __name__ == '__main__':
    main()
