import os
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MEL_PATH = os.path.join(PROJECT_ROOT, "data/processed/mel_spectrograms")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "data/splits")


# fixed input length
def fix_length(mel, target_length=1300):

    if mel.shape[1] < target_length:
        pad_width = target_length - mel.shape[1]
        mel = np.pad(mel, ((0,0),(0,pad_width)), mode='constant')

    else:
        mel = mel[:, :target_length]

    return mel 

class StemDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        self.le = LabelEncoder()
        self.df["label"] = self.le.fit_transform(self.df["label"])

        # group by track
        self.grouped = self.df[self.df["type"] == "stem"].groupby("file")

        self.files = self.df["file"].values

    def __len__(self):
        return len(self.files)

    def load_mel(self, file):

        path = os.path.join(MEL_PATH, file)
        mel = np.load(path)

        mel = fix_length(mel, target_length=1300)

        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.from_numpy(mel).float()

        return mel.unsqueeze(0)

    def __getitem__(self, idx):

        file = self.files[idx]
        label = self.df.iloc[idx]["label"]

        mel = self.load_mel(file)

        return mel, label
    

class CNNModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x
    
# training loop
def train(model, loader, optimizer, criterion, device):

    model.train()
    total_loss = 0

    for x, y in tqdm(loader):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# val loop
def validate(model, loader, criterion, device):

    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():

        for x, y in loader:

            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            total_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()

    acc = correct / len(loader.dataset)

    return total_loss / len(loader), acc

EPOCHS = 30

# main training function
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = os.path.join(SPLIT_PATH, "train.csv")
    val_df = os.path.join(SPLIT_PATH, "val.csv")

    train_dataset = StemDataset(train_df)
    val_dataset = StemDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    num_classes = len(train_dataset.le.classes_)

    model = CNNModel(num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience = 3
    counter = 0

    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(EPOCHS):

        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()