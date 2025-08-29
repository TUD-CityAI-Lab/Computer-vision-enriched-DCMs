import torch
import numpy as np
from ImageChoicedata_preprocessing import data_to_cuda
import torch.optim as optim
import sys
from datetime import datetime
from copy import deepcopy


def train(model, train_loader, test_loader, criterion, wd, learning_rate, patience, n_epochs, device,printLog):
    
    # Report progress
    printLog("Starting model training")

    # Initialize objects + variables
    optimizer = optim.SGD([
        {'params': model.cvmodel.parameters(), 'lr': learning_rate, 'weight_decay' : wd},
        {'params': model.dcm_f.parameters(),   'lr': learning_rate, 'weight_decay' : wd},
        {'params': model.dcm_p.parameters(),   'lr': learning_rate, 'weight_decay' : 0}], lr=learning_rate, momentum=0.98)

    # Initialize lists
    train_loss_all, test_loss_all = [], []
    best_test_loss = np.inf

    # Train model
    for epoch in range(1,n_epochs+1):
        
        # Model training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_loss_all.append(train_loss)

        # Model evaluation
        test_loss = eval_epoch(model, test_loader, criterion, device, epoch)
        test_loss_all.append(test_loss)
        
        # Model selection
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            # Save model
            best_model = deepcopy(model)
            counter = 0
        else:
            counter += 1
        
        # Print progress
        if epoch % 1 == 0:
            
            # Clear line
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.flush()
            
            # Get timestamp
            dateTimeObj = datetime.now()
            dateStr = dateTimeObj.strftime("%H%M_%d_%m_%Y")

            # Print progress
            printLog(f"{dateStr}\tEpoch {epoch:03d}  CEtrain | CEtest | CEtest_best\t {train_loss:0.3f} | {test_loss:0.3f} | {min(test_loss_all):0.3f} {'+' * counter}") 
            beta_hhc = best_model.dcm_p.weight[0][0].cpu().item()
            beta_tti = best_model.dcm_p.weight[0][1].cpu().item()
            printLog(f'beta_hhc = {beta_hhc:10.3f}, beta_tti = {beta_tti:10.3f}')

        if counter >= patience:
            break
        
    # Return best_model
    return best_model, train_loss_all, test_loss_all

def train_epoch(model, data_loader, optimizer, criterion, device, epoch):
    
    # Initialize variables
    model.train()
    total_loss = 0
    num_instances = 0

    for j, (image1, image2, y_label, c1, c2, tt1, tt2, img_name1, img_name2, ID) in enumerate(data_loader):

        if j % (len(data_loader) // np.min((50,len(data_loader)))) == 0:
            sys.stdout.write(f"\rEpoch {epoch:03d} -- Training | {(j / len(data_loader)) * 100:3.0f}% of the batches completed")
            sys.stdout.flush() 

        # Push data to cuda
        image1, image2, c1, c2, tt1, tt2, y_label = data_to_cuda(image1, image2, c1, c2, tt1, tt2, y_label, device) 

        # Forward pass
        prob, V1, V2 = model(image1, image2, c1, tt1, c2, tt2)

        # Compute loss
        loss = criterion(prob, y_label)

        # Backward pass + update weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store loss and num_batch
        total_loss += loss.item()
        num_instances += len(prob)
        
    # Return avg. negative log-likelihood loss over Epoch
    sys.stdout.write(f"\rTrain loss epoch {total_loss/num_instances:0.3f} {' ' * 100} ")
    sys.stdout.flush()
    return total_loss/num_instances

def eval_epoch(model, data_loader, criterion, device, epoch):
    
    # Initialize variables
    model.eval()
    total_loss = 0
    num_instances = 0

    for j, (image1, image2, y_label, c1, c2, tt1, tt2, img_name1, img_name2, ID) in enumerate(data_loader):

        if j % (len(data_loader) // np.min((50,len(data_loader)))) == 0:
            sys.stdout.write(f"\rEpoch {epoch:03d} -- Evaluation | {(j / len(data_loader)) * 100:3.0f}% of the batches completed")
            sys.stdout.flush() 
        
        with torch.no_grad():
            
            # Push data to cuda
            image1, image2, c1, c2, tt1, tt2, y_label = data_to_cuda(image1, image2, c1, c2, tt1, tt2, y_label, device)

            # Forward pass
            prob, V1, V2 = model(image1, image2, c1, tt1, c2, tt2)

            # Compute loss
            loss = criterion(prob, y_label)

            # Store loss and num_batch
            total_loss += loss.item()
            num_instances += len(prob)
        
    # Return avg. negative log-likelihood loss over Epoch
    return total_loss/num_instances