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
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel2"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel2\\" + file_name)
        text = get_text(f)

        if text:
            data[0].append({"text": text, "label": 0})


    # Get texts for 8-9 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel3\\" + file_name)
        text = get_text(f)

        if text:
            data[1].append({"text": text, "label": 1})

    # Get texts for 9-10 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\WRLevel4"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\WRLevel4\\" + file_name)
        text = get_text(f)

        if text:
            data[2].append({"text": text, "label": 2})

    # Get texts for 11-14 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\BitKS3"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\BitKS3\\" + file_name)
        text = get_text(f)

        if text:
            data[3].append({"text": text, "label": 3})

    # Get texts for 15-16 year olds
    for file_name in os.listdir(path + "\\WeeBit-TextOnly\\BitGCSE"):
        if ".txt" not in file_name:
            continue

        f = open(path + "\\WeeBit-TextOnly\\BitGCSE\\" + file_name)
        text = get_text(f)

        if text:
            data[4].append({"text": text, "label": 4})

    min_length = -1

    for label in data:
        if min_length == -1 or len(data[label]) < min_length:
            min_length = len(data[label])

    flattened_data = []

    for label in data:
        data[label] = random.sample(data[label], min_length)
        flattened_data += data[label]

    return flattened_data


def read_newsela(path: str):
    pass


def get_dataset(corpus: Corpus, path: str, reload: False):
    # Process WeeBit dataset if no cached option is found or the data should be reloaded
    if not os.path.isfile("weebit-cache-train.csv") or not os.path.isfile("weebit-cache-test.csv") or reload:
        if corpus == Corpus.WEEBIT:
            data = read_weebit(path)
        else:
            data = read_newsela(path)

        dataset = Dataset.from_list(data)
        dataset = dataset.train_test_split(train_size=0.8, shuffle=True)
        dataset["train"].to_csv("weebit-cache-train.csv")
        dataset["test"].to_csv("weebit-cache-test.csv")

        return dataset

    train_dataset = Dataset.from_csv("weebit-cache-train.csv")
    test_dataset = Dataset.from_csv("weebit-cache-test.csv")

    return DatasetDict({"train": train_dataset, "test": test_dataset})
