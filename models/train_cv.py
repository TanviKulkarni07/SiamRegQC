import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.losses import CosineContrastiveLoss
from config.config import is_multimodal
ce_loss = nn.CrossEntropyLoss()
cc_loss = CosineContrastiveLoss()

def train_epoch(model, train_loader, optimizer, device, add_cc_loss):
    average_train_loss = 0.
    print('-----------Training Epoch----------------')
    for input1, input2, labels in train_loader:            
            
            input1 = input1.type(torch.FloatTensor).to(device)
            input2 = input2.type(torch.FloatTensor).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs_clf = model(input1, input2)

            loss = 0.75*ce_loss(outputs_clf, labels)
            
            # Compute loss
            if add_cc_loss:
                output1 = model.forward_one(input1)
                output2 = model.forward_one(input2)
                loss += 0.25*cc_loss(output1, output2, labels)
            
            average_train_loss += loss
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return average_train_loss.item()/len(train_loader)


def test_epoch(model, test_loader, device, add_cc_loss):
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input1, input2, labels in test_loader:
            
            input1 = input1.type(torch.FloatTensor).to(device)
            input2 = input2.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            outputs_clf = model(input1, input2)

            loss = 0.75*ce_loss(outputs_clf, labels)

            # Compute loss
            if add_cc_loss:
                output1 = model.forward_one(input1)
                output2 = model.forward_one(input2)
                loss += 0.25*cc_loss(output1, output2, labels)

            test_loss += loss.item()
            
            predicted = torch.argmax(outputs_clf.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    return average_test_loss, accuracy

def train_cv(model, dataset, optimizer, device, add_cc_loss, num_folds, num_epochs, batch_size, model_filename):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    best_acc=0.0
    # Iterate over folds

    for fold, (train_indices, val_indices) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"Fold {fold + 1}/{num_folds}")

        # Create DataLoader for training and validation
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            if is_multimodal:
                print('-----Running multimodal epoch-----')
                average_train_loss = train_epoch_multimodal(model, train_loader, optimizer, device, add_cc_loss)
                average_test_loss, average_test_accuracy = test_epoch_multimodal(model, val_loader, device, add_cc_loss)
            else:
                average_train_loss = train_epoch(model, train_loader, optimizer, device, add_cc_loss)
                average_test_loss, average_test_accuracy = test_epoch(model, val_loader, device, add_cc_loss)
            
            if average_test_accuracy >= best_acc:
                best_acc = average_test_accuracy
                print(f'Best Acc achieved! {best_acc * 100:.2f}%')
                torch.save(model.state_dict(), model_filename)            
        
            print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {average_test_loss:.4f} Accuracy: {average_test_accuracy * 100:.2f}%")


def test_epoch_multimodal(model, test_loader, device, add_cc_loss):
    
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for input1, input2, labels in test_loader:
            
            input1 = input1.type(torch.FloatTensor).to(device)
            input2 = input2.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            outputs_clf = model(input1, input2)

            loss = 0.75*ce_loss(outputs_clf, labels)

            # Compute loss
            if add_cc_loss:
                # output1 = model.forward_one(model.vgg_encoder, input1)
                # output2 = model.forward_one(model.vgg_encoder, input2)
                output1 = model.forward_one(model.encoder_T1, input1)
                output2 = model.forward_one(model.encoder_T2, input2)
                loss += 0.25*cc_loss(output1, output2, labels)

            test_loss += loss.item()
            
            predicted = torch.argmax(outputs_clf.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    return average_test_loss, accuracy

def train_epoch_multimodal(model, train_loader, optimizer, device, add_cc_loss):
    average_train_loss = 0.
    print('-----------Training Epoch----------------')
    for input1, input2, labels in train_loader:            
            
            input1 = input1.type(torch.FloatTensor).to(device)
            input2 = input2.type(torch.FloatTensor).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs_clf = model(input1, input2)

            loss = 0.75*ce_loss(outputs_clf, labels)
            
            # Compute loss
            if add_cc_loss:
                # output1 = model.forward_one(model.vgg_encoder, input1)
                # output2 = model.forward_one(model.vgg_encoder, input2)
                output1 = model.forward_one(model.encoder_T1, input1)
                output2 = model.forward_one(model.encoder_T2, input2)
                loss += 0.25*cc_loss(output1, output2, labels)
            
            average_train_loss += loss
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return average_train_loss.item()/len(train_loader)
