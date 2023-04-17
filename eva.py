# !!! When first launching script, uncomment lines 273, 274 (or somewhere like that, there is comment in that area also) 

# Setting up env:
# >>> conda create -n vk_contest pytorch torchvision torchaudio pytorch-cuda=11.7 pandas scikit-learn matplotlib seaborn jupyter jupyterlab tqdm -c nvidia -c pytorch
# >>> conda activate vk_contest
# >>> pip install --pre timm

# Getting data
# >>> kaggle competitions download -c vk-made-sports-image-classification
# >>> unzip vk-made-sports-image-classification.zip
# >>> mv train full_train
# >>> mv test full_test


import os
import shutil
from tqdm import tqdm

import timm
import torch
import torchvision
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

torch.set_num_threads(32)

SEED = 44
FULL_TRAIN = "full_train"
FULL_TEST = "full_test"
NUM_CLASSES = 30

torch.manual_seed(SEED)
np.random.seed(SEED)


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None, device="cuda:0", model_ema=None):
    model.train()
    
    losses = []
    all_labels = []
    all_preds = []
    
    for imgs, labels in tqdm(dataloader, desc="train"):
        optimizer.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        
        loss = criterion(logits, labels)
        loss.backward()

        preds = logits.argmax(-1)
        
        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())        
        losses.append(loss.item())
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    microf1 = f1_score(all_labels, all_preds, average="micro")
    microprecision = precision_score(all_labels, all_preds, average="micro")
    microrecall = recall_score(all_labels, all_preds, average="micro")
    
    macrof1 = f1_score(all_labels, all_preds, average="macro")
    macroprecision = precision_score(all_labels, all_preds, average="macro")
    macrorecall = recall_score(all_labels, all_preds, average="macro")
    
    accuracy = (all_labels == all_preds).mean()
    
    metrics = {
        "accuracy": accuracy,
        "micro_f1": microf1,
        "micro_precision": microprecision,
        "micro_recall": microrecall,
        "macro_f1": macrof1,
        "macro_precision": macroprecision,
        "macro_recall": macrorecall,
        "total_loss": np.mean(losses),
    }
    
    return losses, metrics


@torch.no_grad()
def eval_model(model, dataloader, criterion, device="cuda:0"):
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    all_labels = []
    all_preds = []
    
    for imgs, labels in tqdm(dataloader, desc="eval"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        
        preds = logits.argmax(-1)

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    
    microf1 = f1_score(all_labels, all_preds, average="micro")
    microprecision = precision_score(all_labels, all_preds, average="micro")
    microrecall = recall_score(all_labels, all_preds, average="micro")
    
    macrof1 = f1_score(all_labels, all_preds, average="macro")
    macroprecision = precision_score(all_labels, all_preds, average="macro")
    macrorecall = recall_score(all_labels, all_preds, average="macro")
    
    accuracy = (all_labels == all_preds).mean()
    
    total_loss /= total_samples
    
    return {
        "accuracy": accuracy,
        "micro_f1": microf1,
        "micro_precision": microprecision,
        "micro_recall": microrecall,
        "macro_f1": macrof1,
        "macro_precision": macroprecision,
        "macro_recall": macrorecall,
        "total_loss": total_loss,
    }


@torch.no_grad()
def predict_model(model, dataloader, device="cuda:0"):
    model.eval()
    
    all_preds = []
    all_logits = []
    
    for imgs, _ in tqdm(dataloader, desc="predict"):
        imgs = imgs.to(device)
        
        logits = model(imgs)
        
        preds = logits.argmax(-1)

        all_logits.append(logits.detach().cpu())
        all_preds.append(preds.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_logits = torch.cat(all_logits).numpy()
    
    return all_preds, all_logits


def print_metrics(prefix, metrics, subset=("micro_f1", "total_loss")):
    print(prefix, end=": ")
    for name in subset:
        print(f"{name}: {metrics[name]:.3f}  ", end="")
    print()


def prepare_for_imagefolder(whole_train, whole_test):
    f"Creating subdirectories for classes"
    for label in whole_train.label.unique():
        os.mkdir(f"{FULL_TRAIN}/{label}")
    
    f"Moving images to subdirectories..."
    for _, path, label in whole_train.itertuples():
        os.replace(f"{FULL_TRAIN}/{path}", f"{FULL_TRAIN}/{label}/{path}")

    f"Creating dummy subdirectory 'unknown' to use ImageFolder for test dataset"
    os.mkdir(f"{FULL_TEST}/unknown")
    
    f"Moving images to subdirectory..."
    for _, path in whole_test.itertuples():
        os.replace(f"{FULL_TEST}/{path}", f"{FULL_TEST}/{label}/{path}")
    print()
        

def make_split(whole_train):
    trainval, test = train_test_split(whole_train, test_size=0.1, random_state=SEED)
    train, val = train_test_split(trainval, test_size=0.1, random_state=SEED)

    # test dataset was made, but actually never used, but anyway
    part2df = {
        "train": train,
        "val": val,
        "test": test,
    }

    for part in part2df:
        if os.path.exists(part):
            print(f"{part} exists, continuing")
            continue

        print(f"Creating {part} directory")
        os.mkdir(part)
    
        for label in whole_train.label.unique():
            os.mkdir(f"{part}/{label}")
        
        for _, path, label in part2df[part].itertuples():
            shutil.copyfile(f"{FULL_TRAIN}/{label}/{path}", f"{part}/{label}/{path}")

    return train, val, test


def main():
    assert torch.cuda.is_available(), "Expected to have a GPU (and better a decent one)"

    for subdirectory in ("checkpoints", "metrics", "submissions"):
        if not os.path.exists(subdirectory):
            os.mkdir(subdirectory)
    
# Config
    device = "cuda:0"
    n_layers_to_train = 6
    bs = 64
    n_epochs = 2
    n_full_tune_epochs = 4
    learning_rate = 3e-4
    label_smoothing = 0.1
    experiment_name = f"eva"
    # Training is too long to do in one go, so intermediate checkpoints were made,
    # and sometimes training was restarted, modifying n_epochs and n_full_tune_epochs accordingly.
    checkpoint_path = "" 

# Model
    model = timm.create_model('eva_large_patch14_336.in22k_ft_in22k_in1k', pretrained=True)

    for p in model.parameters():
        p.requires_grad_(False)

    for p in model.blocks[-n_layers_to_train:]:
        p.requires_grad_(True)

    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    model.to(device)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    print("Dummy input...")
    out = model(torch.randn(bs, 3, 336, 336).to(device))
    print("Output shape:", out.shape)

    model_ema = timm.utils.ModelEmaV2(model, decay=0.999, device=device)

# Data
    whole_train = pd.read_csv("train.csv")
    whole_test = pd.read_csv("test.csv")
    # Uncomment next 2 lines __once__ -- when you first launch script
    # prepare_for_imagefolder(whole_train, whole_test)
    # train, val, test = make_split(whole_train)

    data_config = timm.data.resolve_model_data_config(model)
    train_transforms = timm.data.create_transform(**data_config, is_training=True)
    test_transforms = timm.data.create_transform(**data_config, is_training=False)

    whole_ds = torchvision.datasets.ImageFolder(FULL_TRAIN, transform=train_transforms)
    train_ds = torchvision.datasets.ImageFolder("train", transform=train_transforms)
    val_ds = torchvision.datasets.ImageFolder("val", transform=test_transforms)

    whole_dataloader = torch.utils.data.DataLoader(whole_ds, batch_size=bs, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False)

# Training on local train set, tracking metrics
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    print("\nStart training")
    all_train_metrics = []
    all_val_metrics = []
    all_val_ema_metrics = []
    batch_losses = []

    for epoch in range(n_epochs):
        losses, train_metrics = train_one_epoch(model, train_dataloader, optimizer, criterion, model_ema=model_ema)
        val_metrics = eval_model(model, val_dataloader, criterion)
        val_ema_metrics = eval_model(model_ema.module, val_dataloader, criterion)
        
        batch_losses.extend(losses)    
        all_train_metrics.append(train_metrics)
        all_val_metrics.append(val_metrics)
        all_val_ema_metrics.append(val_ema_metrics)

        print_metrics("   Train", train_metrics)
        print_metrics("      Val", val_metrics)
        print_metrics("Val (ema)", val_ema_metrics)

        # To save space on disk, either model or it's ema is checkpointed
        if val_metrics["micro_f1"] > val_ema_metrics["micro_f1"]:
            torch.save(model.state_dict(), f"checkpoints/{experiment_name}_{epoch + 1}_local_train.pth")
        else:
            torch.save(model_ema.module.state_dict(), f"checkpoints/{experiment_name}_ema_{epoch + 1}_local_train.pth")

    if n_epochs > 0:
        all_train_metrics = pd.DataFrame(all_train_metrics)
        all_train_metrics.to_csv(f"metrics/{experiment_name}_{n_epochs}_local_train_train_metrics.csv")
        print("All train metrics\n", all_train_metrics)

        all_val_metrics = pd.DataFrame(all_val_metrics)
        all_val_metrics.to_csv(f"metrics/{experiment_name}_{n_epochs}_local_train_val_metrics.csv")
        print("All val metrics\n", all_val_metrics)

# Proceed training on complete dataset
    whole_metrics = []
    whole_losses = []

    for epoch in range(n_full_tune_epochs):
        losses, metrics = train_one_epoch(model, whole_dataloader, optimizer, criterion, model_ema=model_ema)
        
        whole_losses.extend(losses)
        whole_metrics.append(metrics)
        
        print_metrics("Metrics", metrics)

        torch.save(model_ema.module.state_dict(), 
            f"checkpoints/{experiment_name}_ema_{n_epochs}_epochs_local_train_{epoch + 1}_epochs_whole_train.pth")

# Prediction
    full_test_ds = torchvision.datasets.ImageFolder("full_test", transform=test_transforms)
    full_test_dataloader = torch.utils.data.DataLoader(full_test_ds, bs, shuffle=False)

    all_preds, all_logits = predict_model(model_ema.module, full_test_dataloader)
    
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    whole_test = whole_test.sort_values("image_id")
    whole_test["label"] = all_preds
    whole_test["label"] = whole_test["label"].map(idx_to_class)
    whole_test = whole_test.sort_index()
    print("Predictions:\n", whole_test.head())
    whole_test.to_csv(f"submissions/{experiment_name}_ema_{n_epochs}_{n_full_tune_epochs}_sub.csv", index=False)


main()