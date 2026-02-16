from __future__ import annotations

from pathlib import Path
import re
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import torch.nn.functional as F
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


TRAIN_PATH = "./dataset/train.jsonl"
TEST_PATH = "./dataset/test.jsonl"

class loading:
    """Simple dataset loader for train/test JSONL files."""

    def __init__(self,dataset: str = TRAIN_PATH,seed: int = 42,ratio: float = 0.9,test_path: str = TEST_PATH) -> None:
        self.data = Path(dataset)
        self.test_data = Path(test_path)
        self.seed = seed
        self.ratio = ratio

    def _load(self, path: Path) -> pd.DataFrame:
        """Load a JSONL dataset and return it as a pandas DataFrame."""
        df = pd.read_json(path, lines=True)
        return df

    def split(self) -> tuple[Dataset, Dataset, Dataset]:
        """Split train into train/dev and return torch Datasets for train/dev/test."""
        df = self._load(self.data)
        test_df = self._load(self.test_data)

        train_df, dev_df = train_test_split(df, test_size= 1 - self.ratio, random_state=self.seed)

        return (DatasetNews(train_df), DatasetNews(dev_df), DatasetNews(test_df, has_label=False))

class Preprocessing:
    """Preprocessing class for text data."""
    _stop_words: set[str] | None = None

    def __init__(self, text, stop_words: set[str] = stopwords.words("english")) -> None:
        self.text = text
        self._stop_words = stop_words

    def preprocess(self): # normilize text by lowercasing, removing punctuation, and extra whitespace
        sentence = self.text.lower().strip()
        sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
        sentence = " ".join(sentence.split())
        return sentence

    def tokenize(self): # tokenize the text and remove stop words using nltk
        tokens = word_tokenize(self.preprocess())
        tokens = [word for word in tokens if word not in self._stop_words]
        return ["<s>"] + tokens + ["</s>"]

class DatasetNews(Dataset):
    """Torch dataset that returns tokenized text and label."""

    def __init__(self, df: pd.DataFrame, text_mode: str = "full") -> None:
        self.df = df.reset_index(drop=True)
        self.text_mode = text_mode

        if self.text_mode not in {"full", "title", "description"}:
            raise ValueError(f"Invalid text_mode: {self.text_mode}. Must be 'full', 'title', or 'description'.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        title = row["title"]
        description = row["description"]
        if self.text_mode == "full":
            text = title + " " + description
        elif self.text_mode == "title":
            text = title
        else:
            text = description

        tokens = Preprocessing(text).tokenize()
        
        label = row["label"]
        label = F.one_hot(torch.tensor(label - 1), num_classes=4)
        return {"tokens": tokens, "label": label}