import logging
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm.rich import tqdm

log = logging.getLogger(__name__)

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total += labels.size(0)
            preds = logits.argmax(-1)
            label = labels.argmax(-1)
            correct += (preds == label).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader), correct / total, all_preds, all_labels


def print_confusion_matrix(all_preds, all_labels, acc):
    genres = ["blues", "classical", "country", "disco", "hiphop",
              "jazz", "metal", "pop", "reggae", "rock"]
    confusion = confusion_matrix(all_preds, all_labels)
    precision = np.around(100 * np.diagonal(confusion) / np.sum(confusion, axis=1), 2)
    recall = np.around(100 * np.diagonal(confusion) / np.sum(confusion, axis=0), 2)

    header = genres + ["precision(%)"]
    print("\033[32mconfusion matrix:\033[0m")
    print(" " * 11, end="")
    for g in header:
        print(f"{g:<10}", end="")
    for i, row in enumerate(confusion):
        print(f"\n{genres[i]:<10}", end="")
        for col in row:
            print(f"{col:< 10}", end="")
        print(f"{precision[i]:< 10}", end="")
    print("\n" + "recall(%)  ", end="")
    for v in np.append(recall, np.around(100 * acc, 2)):
        print(f"{v:<10}", end="")
    print()
