import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm

if torch.cuda.is_available():
    DEVICE = 'cuda'
    DEVICE_N_WORKERS = 4
else:
    DEVICE = 'cpu'
    DEVICE_N_WORKERS = 0


def train(model, dataloader, criterion, optimizer, scaler) -> tuple[float, float]:
    """
    Returns:
        total_loss, total_acc
    """

    model.train()
    total_loss, total_acc = 0, 0

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     leave=False, position=0, desc='Train')

    for i, (feature, label) in enumerate(dataloader):
        optimizer.zero_grad()
        feature = feature.to(DEVICE)
        label = label.to(DEVICE)

        # Forward Propagation
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # print(feature.device)
            # print(next(model.parameters()).device)
            logits = model(feature)
            loss = criterion(logits, label)

        # Backpropagation
        scaler.scale(loss).backward()

        # GD
        scaler.step(optimizer)
        scaler.update()

        # Record
        prediction = (logits >= 0.5).int()
        total_loss += loss.item()
        total_acc += torch.sum(prediction == label).item() / logits.shape[0]
        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))),
                              acc="{:.04f}%".format(float(total_acc*100 / (i + 1))))
        batch_bar.update()

        # Release memory
        del feature, label, logits, prediction
        torch.cuda.empty_cache()

    batch_bar.close()

    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc


def eval(model, dataloader, criterion) -> tuple[float, float, float, float, float]:
    """
    Returns:
        total_loss, total_acc, precision, recall, f1
    """

    model.eval()
    total_loss, total_acc = 0, 0
    predictions, labels = [], []

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                     leave=False, position=0, desc='Val')

    for i, (feature, label) in enumerate(dataloader):
        feature = feature.to(DEVICE)
        label = label.to(DEVICE)

        # Forward Propagation
        with torch.inference_mode():
            logits = model(feature)
            loss = criterion(logits, label)
        # Record
        prediction = (logits >= 0.5).int()
        total_loss += loss.item()
        total_acc += (torch.sum(prediction == label).item()) / logits.shape[0]
        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))),
                              acc="{:.04f}%".format(float(total_acc*100 / (i + 1))))
        batch_bar.update()

        labels.extend(label.tolist())
        predictions.extend(prediction.tolist())

        # Release memory
        del feature, label, logits, prediction
        torch.cuda.empty_cache()

    batch_bar.close()

    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return (total_loss, total_acc, precision, recall, f1)  # type: ignore


def test(model, test_loader) -> list[int]:
    """
    Returns:
        predictions
    """

    model.eval()
    predictions = []

    with torch.no_grad():
        for i, feature in enumerate(tqdm(test_loader)):

            feature = feature.to(DEVICE)
            logits = model(feature)
            prediction = (logits >= 0.5).int()
            predictions.extend(prediction.tolist())

    return predictions
