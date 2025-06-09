import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tqdm.rich import tqdm
import logging as log
import log_basic as log_basic

import math
import numpy as np
import matplotlib.pyplot as plt

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.BatchNorm1d(57),

            nn.Linear(57, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
def init():
    log_basic.__init__(log.INFO)
    log.info("torch import complete")
    
    if (device := torch.device("cuda" if torch.cuda.is_available() else "cpu")) == "cuda":
        log.info(f"{torch.cuda.get_device_name(0)}")
        log.info(f"{torch.cuda.device_count()}")
    log.info(f"Using device: {device}")
    return device

class Dataset(Dataset):
    def __init__(self, path):
        x_path, z_path, y_path = path
        dta = np.load(x_path)
        # dta = np.hstack((x,np.load(z_path)))
        self.data = torch.tensor(dta).float()
        self.labels = torch.tensor(np.load(y_path)).float()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'label': self.labels[idx]
        }

def plot(acc, loss, num_epochs=20):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    acc = np.array(acc)*100
    
    # accuracy plt
    ax1.set_xlim((0,num_epochs))
    ax1.set_ylim((0,100))
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy rate (%)")

    ax1.plot(acc[0], "r-", acc[1], "b-", lw=1)
    ax1.axhline(y=60, c="m", ls="--", lw=0.5)
    ax1.axhline(y=80, c="m", ls="--", lw=0.5)

    # loss plt
    ax2.set_xlim((0,num_epochs))
    ax2.set_ylim((0,-math.log(1/10)))
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Loss")

    ax2.plot(loss[0], "r-", loss[1], "b-", lw=1)
    ax2.axhline(0.5, c="m", ls="--", lw=0.5)
    ax2.axhline(1.0, c="m", ls="--", lw=0.5)
    ax2.legend(('training','valid'),loc=1)

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.savefig(r"training_plot.png")
    plt.show()

def init():
    log_basic.__init__(log.INFO)
    log.info("torch import complete")
    
    if (device := torch.device("cuda" if torch.cuda.is_available() else "cpu")) == "cuda":
        log.info(f"{torch.cuda.get_device_name(0)}")
        log.info(f"{torch.cuda.device_count()}")
    log.info(f"Using device: {device}")
    return device

def main():
    
    device = init()
    dataset_path = (r"MLP_x_data.npy", r"MLP_z_data.npy", r"MLP_y_data.npy")

    num_epochs = 50
    lr = 1e-3
    batch_size = 2000

    model = MLPModel()
    model.to(device)

    Loss, Acc = [[],[]], [[],[]]
    loss, acc = 0, 0

    dataset = Dataset(path=dataset_path)
    train_dataset, val_dataset = train_test_split(dataset, test_size = 0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(batch_size), shuffle=True)

    num_classes = 10

    log.info(f"class weight : {np.argmax(np.load(r"MLP_y_data.npy"), axis=1)}")
    weights = compute_class_weight(
        class_weight='balanced', classes=np.arange(num_classes),
        y = np.argmax(np.load(r"MLP_y_data.npy"), axis=1)
        )
    
    class_weights = torch.tensor(weights, dtype=torch.float32)
    

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=3e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for epoch in range(num_epochs):
        
        # ===============
        #    training
        # ===============

        total, total_loss, correct = 0, 0, 0

        model.train()
        for batch in tqdm(train_loader, leave=True):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            total += labels.size(0)
            preds = logits.argmax(dim=-1)
            label = labels.argmax(dim=-1)
            correct += (preds == label).sum().item()

        scheduler.step()
        acc = correct / total
        total_loss = total_loss / len(train_loader)

        if (epoch+1)%10 == 0:
            log.info(f"\033[32mEpoch: {epoch+1: <3} / {num_epochs: <3}\033[0m")
            Loss[0].append(total_loss)
            Acc[0].append(acc)
            log.info(f"Training   | Loss: {total_loss:.4f} | Accuracy: {100*acc:4.2f}%")

        # ===============
        #      vaild
        # ===============

        total, total_loss, correct = 0, 0, 0
        all_preds, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                
                total += labels.size(0)
                preds = logits.argmax(dim=-1)
                label = labels.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                correct += (preds == label).sum().item()

        acc = correct / total
        total_loss = total_loss / len(val_loader)

        if (epoch+1)%10 == 0:
            Loss[1].append(total_loss)
            Acc[1].append(acc)
            log.info(f"Validation | Loss: {total_loss:.4f} | Accuracy: {100*acc:4.2f}%")

    confusion = confusion_matrix(all_preds, all_labels)
    precision = np.around(100*np.diagonal(confusion) / np.sum(confusion, axis=1), 2)
    recall = np.around(100*np.diagonal(confusion) / np.sum(confusion, axis=0), 2)
    print("\033[32mconfusion matrix:\033[0m")
    genres = ["blues", "classical", "country", "disco", "hiphop", 
            "jazz", "metal", "pop", "reggae", "rock", "precision(%)"]
    print(" "*11,end="")
    for genre in genres: print(f"{genre:<10}",end="")
    i = 0
    for correct_row in confusion:
        print("\n"+f"{genres[i]:<10}",end="")
        for col in correct_row:
            print(f"{col:< 10}",end="")
        print(f"{precision[i]:< 10}",end="")
        i += 1
    print("\n"+"recall(%)  ",end="")
    for genre in np.append(recall, np.around(100*acc, 2)): print(f"{genre:<10}",end="")
    print()
            
    torch.save(model.state_dict(),r"mlp_parms.pth")
    log.info("model save complete !")
                
    return Acc, Loss

if __name__ == "__main__": 
    Acc, Loss = main()
    plot(Acc, Loss)