import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cvdcm import cvdcm_model
import numpy as np
import pandas as pd
import os
import platform
from train import train
from pathlib import Path
from datetime import datetime
from ImageChoicedata_preprocessing import ImageChoiceDataset

# Clears the terminal
# os.system('clear')
if __name__ == "__main__":
    
    # Get timestamp
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%H%M_%d_%m_%Y")

    # Initialise paths
    working_folder = Path(os.path.dirname(os.path.realpath(__file__)))

    # Initialise paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent # Adjust as necessary
    data_dir = repo_root / "data"
    img_path = data_dir / "images"
    img_path = '/Users/sandervancranenburgh/Documents/Repos_and_data/Data/window_project_data/images'
    choice_data_file = data_dir / "data_CV_DCM.csv"

    # Now use img_path and choice_data_file in your code
    print("Image path:", img_path)
    print("Choice data file:", choice_data_file)

    # Keep diary
    keep_diary = 1
    output_filen = f'{dateStr}_output.txt'
    if keep_diary == 1:
        def printLog(*args, **kwargs):
            print(*args, **kwargs)
            with open(working_folder/output_filen,'a') as file:
                print(*args, **kwargs, file=file)
    else:
        def printLog(*args, **kwargs):
                print(*args, **kwargs)
    
    # Display code started
    printLog("CODE STARTED at:",dateStr)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    printLog(f'Using device: {device}')

    # Hyperparameters
    learning_rate = 1e-5
    batch_size = 10
    wd = 0.1
    num_epochs = 100
    patience = 5
    workers = 2
    printLog(f'Hyperparameters: learning_rate = {learning_rate}, batch_size = {batch_size}, wd = {wd}')
    
    # Load data
    dataset_train = ImageChoiceDataset(data_file = choice_data_file, img_path = img_path, set_type = 'train', transform = True)
    dataset_test  = ImageChoiceDataset(data_file = choice_data_file, img_path = img_path, set_type = 'test' , transform = True)

    # Create dataloaders
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(dataset=dataset_test,  batch_size=batch_size, shuffle=False,num_workers=workers, pin_memory=True)

    # Load the model
    working_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    path_pretrained_model = working_folder / 'pretrained_models' / 'Model3_CVDCM_CE0585_retrained.pt' # Pretrained model path
    model = cvdcm_model(path_pretrained_model, device=device).to(device=device,non_blocking=False)

    # Loss function
    criterion = nn.BCELoss(reduction='sum')
    
    # Train model
    best_model, train_loss, test_loss = train(model, train_loader, test_loader, criterion, wd, learning_rate, patience, num_epochs, device,printLog)

    # Display the behavioural weights of the model
    printLog(f'\nBehavioural weights of the model after scaling back:')
    beta_hhc = best_model.dcm_p.weight[0][0].cpu().item()
    beta_tti = best_model.dcm_p.weight[0][1].cpu().item()
    printLog(f'beta_hhc = {beta_hhc * (1/225):10.3f}')
    printLog(f'beta_tti = {beta_tti * (1/15):10.3f}')
    
    # Find the index where test_loss is minimum
    min_loss_index = test_loss.index(min(test_loss))
    
    # Use the index to get the corresponding minimum losses for CE and MSE
    min_loss_str_CE  = f'{test_loss[min_loss_index]:0.3f}'.replace('.', '')

    # Construct the model name
    model_name = f'{dateStr}_CVDCM_CE{min_loss_str_CE}.pt'

    # Save the model
    torch.save(best_model.state_dict(), working_folder/model_name)
    print(f'\nSaved model as {model_name}')