import re
from collections import Counter
from pathlib import Path
from pprint import pprint

import matplotlib
import numpy as np

from models.load import DatasetNews, loading

matplotlib.use("Agg")

DATA_PATH = Path("./dataset/train.jsonl")
OUTPUT_PLOT = Path("./output_images/text_length_boxplot.png")

UNK = "<unk>"
PAD = "<pad>"

CLASSES = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4,
}


def texts_from_newsdataset(df) -> list[str]:
    dataset = DatasetNews(df, text_mode="full") 
    texts = []
    for idx in range(len(dataset)):
        tokens = dataset[idx]["tokens"]
        words = [
            token
            for token in tokens
            if isinstance(token, str) and token not in {"<s>", "</s>"}
        ]
        texts.append(" ".join(words))
    return texts


def length_of_texts(texts: list, class_label=None):
    """Calculate the length of each text in a list of texts."""
    maximum_length = 0
    minimum_length = -1
    mode_length = 0
    mode_frequency = 0
    counter = Counter()
    lengths = []

    for text in texts:
        length = len(text.split())
        lengths.append(length)
        counter[length] += 1

        if counter[length] > mode_frequency:
            mode_frequency = counter[length]
            mode_length = length
        if length > maximum_length:
            maximum_length = length
        if minimum_length == -1 or length < minimum_length:
            minimum_length = length

    array = np.array(lengths, dtype=float)
    mean_length = np.mean(array)
    median_length = np.median(array)
    interval_length = (np.percentile(array, 5), np.percentile(array, 95))

    return {
        "class_label": class_label,
        "max_length": maximum_length,
        "min_length": minimum_length,
        "mode_length": mode_length,
        "mean_length": mean_length,
        "median_length": median_length,
        "interval_length": interval_length,
    }, array


def vocab_from_texts(texts: list) -> tuple[set, int]:
    """Calculate the vocabulary from a list of texts."""
    vocab = set()
    for text in texts:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        vocab.update(words)
    return vocab, len(vocab)


def most_common_words(texts: list, n=15):
    """Calculate the most common words in a list of texts."""
    counter = Counter()
    for text in texts:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        counter.update(words)
    most_common = counter.most_common(n)
    return most_common


if __name__ == "__main__":
    pprint("running pre-analysis...")
    loader = loading(dataset=str(DATA_PATH))
    df = loader._load(DATA_PATH)
    common_words = Counter()
    class_commons = {}
    text_stats = {}
    data_length = []  # length of texts used for the boxplot
    for class_name, id in CLASSES.items():
        class_texts = texts_from_newsdataset(df[df["label"] == id])
        stats, array = length_of_texts(
            class_texts, class_label=class_name
        )
        text_stats[class_name] = stats
        data_length.append(array)

        class_common = most_common_words(class_texts)
        class_commons[class_name] = class_common
        for word, count in class_common:
            common_words[word] += count

        print("\n")
        print(f"--- {class_name} ---")
        print(f" entries in class in texts: {len(class_texts)}")
        print(f" text length stats:")
        print(f" Mean: {stats['mean_length']}")
        print(f" Median: {stats['median_length']}")
        print(f" Mode: {stats['mode_length']}")
        print(f" Min: {stats['min_length']}")
        print(f" Max: {stats['max_length']}")
        print(f" CLI: {stats['interval_length']}")
        pprint(f" Most common words: {class_common[:10]}...")

    pprint(f"-------------- Overall --------------")
    pprint(
        f"Most common words across all classes: {common_words.most_common(15)}"
    )
    pprint(f"Overall text length stats:")
    all_lengths = np.concatenate(data_length)
    print("Mean:", np.mean(all_lengths))
    print("90th percentile:", np.percentile(all_lengths, 90))
    print("95th percentile:", np.percentile(all_lengths, 95))
    print("99th percentile:", np.percentile(all_lengths, 99))
    print("Max:", np.max(all_lengths))
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(data, labels=CLASSES.keys())
    # plt.title("text Length Distribution by Class")
    # plt.xlabel("Class")
    # plt.ylabel("text Length (number of words)")
    # plt.savefig(OUTPUT_PLOT)
    pprint(f"-------------- End --------------")
