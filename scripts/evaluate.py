import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm
from tabulate import tabulate
from src.data import CharacterClassificationDataset

import src.utils
import numpy as np
import src.models as vits

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader


def get_dataloaders(args):
    if args.source == "images":
        transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transforms = T.Compose(
            [
                T.Lambda(lambda x: torch.from_numpy(x).float()),
            ]
        )

    train_dataset = CharacterClassificationDataset(
        args.data_dir,
        transform=transforms,
        split="train",
        source=args.source,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = CharacterClassificationDataset(
        args.data_dir,
        transform=transforms,
        split="val",
        source=args.source,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def evaluate(datloader, mode="train"):
    all_latents, all_labels = [], []
    all_valence, valence_mask = [], []
    all_arousal, arousal_mask = [], []

    for character, y, val, aro in tqdm(datloader):
        if not args.no_model:
            with torch.no_grad():
                model.context_features = 0
                character = character.to(device)
                character = model(character)

        all_latents.append(character)
        all_labels.append(y)

        all_valence.append(val)
        valence_mask.append(val != -1)

        all_arousal.append(aro)
        arousal_mask.append(aro != -1)

    all_latents = torch.cat(all_latents, dim=0).detach().cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).detach().cpu().numpy()

    all_valence = torch.cat(all_valence, dim=0).detach().cpu().numpy()
    valence_mask = torch.cat(valence_mask, dim=0).detach().cpu().numpy()

    all_arousal = torch.cat(all_arousal, dim=0).detach().cpu().numpy()
    arousal_mask = torch.cat(arousal_mask, dim=0).detach().cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(all_latents, all_labels)
    all_preds = knn.predict(all_latents)

    character_accuracy = accuracy_score(all_labels, all_preds)
    character_precision = precision_score(all_labels, all_preds, average="weighted")
    character_recall = recall_score(all_labels, all_preds, average="weighted")
    character_f1 = f1_score(all_labels, all_preds, average="weighted")
    character_silhouette = silhouette_score(all_latents, all_labels)

    (
        valence_accuracy,
        valence_precision,
        valence_recall,
        valence_f1,
        valence_silhouette,
        valence_samples,
    ) = (0, 0, 0, 0, 0, 0)

    (
        arousal_accuracy,
        arousal_precision,
        arousal_recall,
        arousal_f1,
        arousal_silhouette,
        arousal_samples,
    ) = (0, 0, 0, 0, 0, 0)

    for c in range(all_labels.max() + 1):
        c_mask = valence_mask & (all_labels == c)

        if c_mask.sum() >= 10:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(all_latents[c_mask], all_valence[c_mask])
            all_preds = knn.predict(all_latents[c_mask])

            valence_accuracy += accuracy_score(all_valence[c_mask], all_preds)
            valence_precision += precision_score(
                all_valence[c_mask], all_preds, average="weighted"
            )
            valence_recall += recall_score(
                all_valence[c_mask], all_preds, average="weighted"
            )
            valence_f1 = f1_score(all_valence[c_mask], all_preds, average="weighted")

            valence_samples += 1

        c_mask = arousal_mask & (all_labels == c)

        if c_mask.sum() >= 10:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(all_latents[c_mask], all_arousal[c_mask])
            all_preds = knn.predict(all_latents[c_mask])

            arousal_accuracy += accuracy_score(all_arousal[c_mask], all_preds)
            arousal_precision += precision_score(
                all_arousal[c_mask], all_preds, average="weighted"
            )
            arousal_recall += recall_score(
                all_arousal[c_mask], all_preds, average="weighted"
            )
            arousal_f1 = f1_score(all_arousal[c_mask], all_preds, average="weighted")

            arousal_samples += 1

    valence_accuracy /= valence_samples
    valence_precision /= valence_samples
    valence_recall /= valence_samples
    valence_f1 /= valence_samples

    arousal_accuracy /= arousal_samples
    arousal_precision /= arousal_samples
    arousal_recall /= arousal_samples
    arousal_f1 /= arousal_samples

    return (
        character_accuracy,
        character_precision,
        character_recall,
        character_f1,
        character_silhouette,
        valence_accuracy,
        valence_precision,
        valence_recall,
        valence_f1,
        valence_silhouette,
        arousal_accuracy,
        arousal_precision,
        arousal_recall,
        arousal_f1,
        arousal_silhouette,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--arch", type=str, default="vit_small")
    parser.add_argument("--checkpoint-key", type=str, default="teacher")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints")

    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_model:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)

        utils.load_pretrained_weights(
            model, args.checkpoint_path, args.checkpoint_key, args.arch, args.patch_size
        )

        model.to(device)
        model.eval()

    train_dataloader, val_dataloader = get_dataloaders(args)

    train_metrics = evaluate(train_dataloader, mode="train")
    val_metrics = evaluate(val_dataloader, mode="val")

    table = [
        [
            "Character",
            f"{100 * train_metrics[0]:.2f}%",
            f"{100 * train_metrics[1]:.2f}%",
            f"{100 * train_metrics[2]:.2f}%",
            f"{100 * train_metrics[3]:.2f}%",
            f"{train_metrics[4]:.4f}",
            f"{100 * val_metrics[0]:.2f}%",
            f"{100 * val_metrics[1]:.2f}%",
            f"{100 * val_metrics[2]:.2f}%",
            f"{100 * val_metrics[3]:.2f}%",
            f"{val_metrics[4]:.4f}",
        ],
        [
            "Valence",
            f"{100 * train_metrics[5]:.2f}%",
            f"{100 * train_metrics[6]:.2f}%",
            f"{100 * train_metrics[7]:.2f}%",
            f"{100 * train_metrics[8]:.2f}%",
            f"{train_metrics[9]:.4f}",
            f"{100 * val_metrics[5]:.2f}%",
            f"{100 * val_metrics[6]:.2f}%",
            f"{100 * val_metrics[7]:.2f}%",
            f"{100 * val_metrics[8]:.2f}%",
            f"{val_metrics[9]:.4f}",
        ],
        [
            "Arousal",
            f"{100 * train_metrics[10]:.2f}%",
            f"{100 * train_metrics[11]:.2f}%",
            f"{100 * train_metrics[12]:.2f}%",
            f"{100 * train_metrics[13]:.2f}%",
            f"{train_metrics[14]:.4f}",
            f"{100 * val_metrics[10]:.2f}%",
            f"{100 * val_metrics[11]:.2f}%",
            f"{100 * val_metrics[12]:.2f}%",
            f"{100 * val_metrics[13]:.2f}%",
            f"{val_metrics[14]:.4f}",
        ],
    ]

    print(
        tabulate(
            table,
            headers=[
                "Metric",
                "Train Accuracy",
                "Train Precision",
                "Train Recall",
                "Train F1",
                "Train Silhouette",
                "Val Accuracy",
                "Val Precision",
                "Val Recall",
                "Val F1",
                "Val Silhouette",
            ],
            tablefmt="fancy_grid",
        )
    )
