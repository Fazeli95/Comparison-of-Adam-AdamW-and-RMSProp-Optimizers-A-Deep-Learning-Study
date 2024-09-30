import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import SimpleNN
from utils import set_seed
from data_loader import load_kmnist

def train_model(trial, device, criterion, num_epochs=9):
    dataset = load_kmnist()

    lr = trial.suggest_loguniform('lr', 5e-4, 1e-2)
    betas = (trial.suggest_uniform('beta1', 0.85, 0.95), trial.suggest_uniform('beta2', 0.98, 0.999))
    eps = trial.suggest_loguniform('eps', 1e-9, 1e-7)
    weight_decay = trial.suggest_loguniform('weight_decay', 5e-3, 5e-2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    kfold = KFold(n_splits=4, shuffle=True, random_state=43)
    writer = SummaryWriter()
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = SimpleNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * data.size(0)
                _, pred = torch.max(output, 1)
                train_correct += torch.sum(pred == target.data)

            train_accuracy = train_correct.double() / len(train_loader.dataset)
            val_accuracy = validate_model(model, val_loader, criterion, device)
            accuracies.append(val_accuracy)
            
            writer.add_scalar(f'Fold {fold + 1}/Accuracy', val_accuracy, epoch)
        print(f'Fold {fold+1}: Accuracy: {val_accuracy}')
    writer.close()
    return np.mean(accuracies)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            val_correct += torch.sum(pred == target.data)
    return val_correct.double() / len(val_loader.dataset)
