import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import load_config, init_logger, set_seed
from data import MelDataset
from model import TransformerEncoder
from engine import train_one_epoch, evaluate, print_confusion_matrix

log = logging.getLogger(__name__)


def plot(acc, loss, num_epochs, save_path="runs/training_plot.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    acc_arr = np.array(acc) * 100

    ax1.set_xlim((0, num_epochs)); ax1.set_ylim((0, 100))
    ax1.set_xlabel("epochs"); ax1.set_ylabel("accuracy rate (%)")
    ax1.plot(acc_arr[0], "r-", acc_arr[1], "b-", lw=1)
    ax1.axhline(y=60, c="m", ls="--", lw=0.5)
    ax1.axhline(y=80, c="m", ls="--", lw=0.5)

    ax2.set_xlim((0, num_epochs)); ax2.set_ylim((0, -math.log(1 / 10)))
    ax2.set_xlabel("epochs"); ax2.set_ylabel("Loss")
    ax2.plot(loss[0], "r-", loss[1], "b-", lw=1)
    ax2.axhline(0.5, c="m", ls="--", lw=0.5)
    ax2.axhline(1.0, c="m", ls="--", lw=0.5)
    ax2.legend(("training", "valid"), loc=1)

    plt.savefig(save_path)
    plt.show()


def main():
    cfg = load_config("configs/config.yaml")
    t = cfg["training"]
    m = cfg["model"]
    p = cfg["paths"]

    init_logger(log_dir=p["log_dir"])
    set_seed(t["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(torch.cuda.get_device_name(0))
    log.info(f"Using device: {device}")

    dataset = MelDataset(p["data_x"], p["data_y"])
    train_ds, val_ds = train_test_split(dataset, test_size=t["test_size"], random_state=t["random_state"])
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False)

    model = TransformerEncoder(**m).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler"]["step_size"], gamma=cfg["scheduler"]["gamma"]
    )

    Acc, Loss = [[], []], [[], []]
    all_preds, all_labels = [], []

    for epoch in range(t["num_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

        Acc[0].append(train_acc); Acc[1].append(val_acc)
        Loss[0].append(train_loss); Loss[1].append(val_loss)

        log.info(f"\033[32mEpoch {epoch+1:>3}/{t['num_epochs']}\033[0m  "
                 f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:5.2f}%  |  "
                 f"Val Loss: {val_loss:.4f} Acc: {100*val_acc:5.2f}%")

    print_confusion_matrix(all_preds, all_labels, val_acc)

    Path(p["save_model"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p["save_model"])
    log.info("model saved !")

    plot(Acc, Loss, t["num_epochs"])


if __name__ == "__main__":
    main()
