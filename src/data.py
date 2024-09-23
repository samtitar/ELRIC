import io
import json
import h5py
import torch
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class CharacterSequenceDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        character_transform=None,
        sequence_transform=None,
        return_metadata=False,
    ):
        super().__init__()

        self.split = split

        self.character_transform = character_transform
        self.sequence_transform = sequence_transform
        self.return_metadata = return_metadata

        self.panels_file = h5py.File(f"{data_dir}/{split}_sequence_features.hdf5", "r")

        with open(f"{data_dir}/{split}_sequence_features_indexing.json", "r") as f:
            indexing = json.load(f)[f"{split}_sequence_features"]

        self.panel_to_idx = {}

        print("Indexing panels for sequence sampling...")
        comics = indexing.keys()
        for comic in comics:
            if "feat_data" not in indexing[comic]:
                continue

            self.panel_to_idx[comic] = {}
            panels = indexing[comic]["feat_data"]

            for i, panel in enumerate(panels):
                self.panel_to_idx[comic][panel] = i

        self.characters_file = h5py.File(f"{data_dir}/{split}_images.hdf5", "r")

        with open(f"{data_dir}/{split}_images_indexing.json", "r") as f:
            indexing = json.load(f)[f"{split}_images"]

        self.character_to_idx = {}
        self.idx_to_character = []

        print("Indexing characters for sampling...")
        comics = indexing.keys()
        for comic in comics:
            if "img_data" not in indexing[comic]:
                continue

            self.character_to_idx[comic] = {}
            characters = indexing[comic]["img_data"]

            for i, character in enumerate(characters):
                panel = "_".join(character.split("_")[:2])

                if (
                    comic not in self.panel_to_idx
                    or panel not in self.panel_to_idx[comic]
                ):
                    continue

                self.idx_to_character.append((comic, panel, character))
                self.character_to_idx[comic][character] = i

    def __len__(self):
        return len(self.idx_to_character)

    def __getitem__(self, idx):
        comic, panel, character = self.idx_to_character[idx]

        character_idx = self.character_to_idx[comic][character]
        panel_idx = self.panel_to_idx[comic][panel]

        character_img = self.characters_file[f"{self.split}_images"][comic]["img_data"][
            character_idx
        ]
        character_img = Image.open(io.BytesIO(character_img))

        panel_feats = self.panels_file[f"{self.split}_sequence_features"][comic][
            "feat_data"
        ][panel_idx]

        if self.character_transform is not None:
            character_img = self.character_transform(character_img)

        if self.sequence_transform is not None:
            panel_feats = self.sequence_transform(panel_feats)

        if self.return_metadata:
            return character_img, panel_feats, (comic, panel, character)
        return character_img, panel_feats

    def close(self):
        self.characters_file.close()
        self.panels_file.close()


class CharacterClassificationDataset(Dataset):
    def __init__(self, data_dir, split="train", source="features", transform=None):
        super().__init__()

        self.split = split
        self.source = source
        self.transform = transform
        self.characters_file = h5py.File(f"{data_dir}/{split}_{source}.hdf5", "r")
        self.emotions_file = pd.read_csv(f"{data_dir}/characters_indexing.csv")

        with open(f"{data_dir}/{split}_{self.source}_indexing.json", "r") as f:
            indexing = json.load(f)[f"{split}_{source}"]

        self.character_to_idx = {}
        self.idx_to_character = []
        self.character_to_class = {}

        self.data_source = "img_data"
        if self.source == "features":
            self.data_source = "feat_data"

        print("Indexing characters for sampling...")
        comics = indexing.keys()
        for comic in tqdm(comics):
            if self.data_source not in indexing[comic]:
                continue

            self.character_to_idx[comic] = {}
            characters = indexing[comic][self.data_source]

            for i, character in enumerate(characters):
                self.idx_to_character.append((comic, character))
                self.character_to_idx[comic][character] = i

                character_id = character.split("_")[-1]
                if character_id not in self.character_to_class:
                    self.character_to_class[character_id] = len(self.character_to_class)

        self.num_classes = len(self.character_to_class)

    def __len__(self):
        return len(self.idx_to_character)

    def __getitem__(self, idx):
        comic, character = self.idx_to_character[idx]

        page, panel, character_id = character.split("_")
        emotion_row = self.emotions_file[
            (self.emotions_file["page"] == float(page))
            & (self.emotions_file["panel"] == f"{page}_{panel}")
            & (self.emotions_file["character"] == f"{comic}_{character_id}")
        ]

        valence = emotion_row["valence"].values[0]
        arousal = emotion_row["arousal"].values[0]
        
        character_idx = self.character_to_idx[comic][character]

        character_data = self.characters_file[f"{self.split}_{self.source}"][comic][
            self.data_source
        ][character_idx]

        if self.data_source == "img_data":
            character_data = Image.open(io.BytesIO(character_data))

        if self.transform is not None:
            character_data = self.transform(character_data)

        return character_data, self.character_to_class[character_id], valence, arousal
