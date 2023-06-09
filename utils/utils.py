import os
import random

from datasets import Dataset, DatasetDict
from utils.corpus import Corpus


def get_text(f):
    lines = f.readlines()
    text = ""

    # Remove CSS warning if present
    if "This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n" in lines:
        lines.remove(
            "This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled. While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience. Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.\n")

    for line in lines:
        # Skip paragraphs with less than 2 declarative sentences, likely to be irrelevant or meta text which may affect training
        if line.count('.') > 1:
            text += line

    return text


def read_weebit(path: str):
    data = {0: [], 1: [], 2: [], 3: [], 4: []}

    # Get texts for 7-8 year olds
    for file_name in os.listdir(path + "\\WRLevel2"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WRLevel2\\" + file_name)
        text = get_text(f)

        if text:
            data[0].append({"text": text, "label": 0})


    # Get texts for 8-9 year olds
    for file_name in os.listdir(path + "\\WRLevel3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WRLevel3\\" + file_name)
        text = get_text(f)

        if text:
            data[1].append({"text": text, "label": 1})

    # Get texts for 9-10 year olds
    for file_name in os.listdir(path + "\\WRLevel4"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WRLevel4\\" + file_name)
        text = get_text(f)

        if text:
            data[2].append({"text": text, "label": 2})

    # Get texts for 11-14 year olds
    for file_name in os.listdir(path + "\\BitKS3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\BitKS3\\" + file_name)
        text = get_text(f)

        if text:
            data[3].append({"text": text, "label": 3})

    # Get texts for 15-16 year olds
    for file_name in os.listdir(path + "\\BitGCSE"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\BitGCSE\\" + file_name)
        text = get_text(f)

        if text:
            data[4].append({"text": text, "label": 4})

    min_length = -1

    for label in data:
        if min_length == -1 or len(data[label]) < min_length:
            min_length = len(data[label])

    train = []
    test = []

    for label in data:
        data[label] = random.sample(data[label], min_length)
        random.shuffle(data[label])
        split_no = int(0.8 * min_length)
        train += data[label][:split_no]
        test += data[label][split_no:]

    random.shuffle(train)
    random.shuffle(test)

    return train, test


def read_newsela(path: str):
    pass


def get_dataset(corpus: Corpus, path: str, reload: False):
    # Process WeeBit dataset if no cached option is found or the data should be reloaded
    if not os.path.isfile("weebit-cache-train.csv") or not os.path.isfile("weebit-cache-test.csv") or reload:
        if corpus == Corpus.WEEBIT:
            train, test = read_weebit(path)
        else:
            train, test = read_newsela(path)

        train_dataset = Dataset.from_list(train)
        test_dataset = Dataset.from_list(test)
        train_dataset.to_csv("weebit-cache-train.csv")
        test_dataset.to_csv("weebit-cache-test.csv")

        return DatasetDict({"train": train_dataset, "test": test_dataset})

    train_dataset = Dataset.from_csv("weebit-cache-train.csv")
    test_dataset = Dataset.from_csv("weebit-cache-test.csv")

    return DatasetDict({"train": train_dataset, "test": test_dataset})
