import logging
import torch
from tqdm.rich import tqdm

log = logging.getLogger(__name__)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for batch in tqdm(loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        correct += (logits.argmax(-1) == labels.argmax(-1)).sum().item()

    return total_loss / len(loader), correct / total
