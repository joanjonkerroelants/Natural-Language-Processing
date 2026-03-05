from pathlib import Path
from collections import Counter
from models.load import loading, DatasetNews
import numpy as np 
from pprint import pprint
import re 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = Path("./dataset/train.jsonl")
OUTPUT_PLOT = Path("./output_images/sentence_length_boxplot.png")

CLASSES = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4,
}

def sentences_from_newsdataset(df) -> list[str]:
    dataset = DatasetNews(df, text_mode="full")
    sentences = []
    for idx in range(len(dataset)):
        tokens = dataset[idx]["tokens"]
        words = [
            token
            for token in tokens
            if isinstance(token, str) and token not in {"<s>", "</s>"}
        ]
        sentences.append(" ".join(words))
    return sentences


def length_of_sentences(sentences: list, class_label=None):
    """Calculate the length of each sentence in a list of sentences."""
    maximum_length = 0
    minimum_length = -1
    mode_length = 0
    mode_frequency = 0
    counter = Counter()
    lengths = []

    for sentence in sentences:
        length = len(sentence.split())
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

def vocab_from_sentences(sentences: list) -> set:
    """Calculate the vocabulary from a list of sentences."""
    vocab = set()
    for sentence in sentences:
        sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
        words = sentence.split()
        if words not in vocab:
            vocab.update(words)
    return vocab

def most_common_words(sentences: list, n=15):
    """Calculate the most common words in a list of sentences."""
    counter = Counter()
    for sentence in sentences:
        sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
        words = sentence.split()
        counter.update(words)
    most_common = counter.most_common(n)
    return most_common

if __name__ == "__main__":
    pprint("running pre-analysis...")
    loader = loading(dataset=str(data))
    df = loader._load(data)
    common_words = Counter()
    class_commons = {}
    sentence_stats = {} 
    data = []
    for class_name, id in CLASSES.items():
        class_sentences = sentences_from_newsdataset(df[df["label"] == id])
        stats, array = length_of_sentences(class_sentences, class_label=class_name)
        sentence_stats[class_name] = stats
        data.append(array)

        class_common = most_common_words(class_sentences)
        class_commons[class_name] = class_common
        for word, count in class_common:
            common_words[word] += count

        print(f"--- {class_name} ---")
        print(f" entries in class in sentences: {len(class_sentences)}")
        print(f" Sentence length stats:")
        print(f" Mean: {stats['mean_length']}")
        print(f" Median: {stats['median_length']}")
        print(f" Mode: {stats['mode_length']}")
        print(f" Min: {stats['min_length']}")
        print(f" Max: {stats['max_length']}")
        print(f" CLI: {stats['interval_length']}")
        pprint(f" Most common words: {class_common[:10]}...")

    pprint(f"-------------- Overall --------------")
    pprint(f"Most common words across all classes: {common_words.most_common(15)}")
    
    intervals = [stats['interval_length'] for stats in sentence_stats.values()]
    interval_mins = [interval[0] for interval in intervals]
    interval_maxs = [interval[1] for interval in intervals]
    
    pprint(f"total CLI: ({(np.percentile(interval_mins, 5))}, {(np.percentile(interval_maxs, 95))})")
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=CLASSES.keys())
    plt.title("Sentence Length Distribution by Class")
    plt.xlabel("Class")
    plt.ylabel("Sentence Length (number of words)")
    plt.savefig(OUTPUT_PLOT)
    pprint(f"-------------- End --------------")


