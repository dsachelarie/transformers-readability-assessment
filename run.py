from transformer_model import TransformerModel
from utils.corpus import Corpus
from utils.model import Model
from utils import utils
import readnet
import numpy as np
import torch


WEEBIT_PATH = "WeeBit-NoSpecialChar"
NEWSELA_PATH = ""


def run_on_corpus(corpus: Corpus, num_labels: int, models: [Model], reload=False):
    if corpus == Corpus.WEEBIT:
        dataset = utils.get_dataset(corpus, WEEBIT_PATH, reload)
    else:
        dataset = utils.get_dataset(corpus, NEWSELA_PATH, reload)

    for model in models:
        if model == Model.READNET:
            predictions = readnet.run_readnet(dataset["test"].to_dict()["text"], evaluate_only=True)
            labels = dataset["test"].to_dict()["label"]
            file = "predictions/readnet_predictions.txt"

        elif model == Model.BERT:
            transformer_model = TransformerModel("bert-base-uncased", num_labels)
            file = "predictions/bert_predictions.txt"
            dataset = transformer_model.tokenize(dataset)
            predictions, labels = transformer_model.run(dataset["train"], dataset["test"])

        elif model == Model.ROBERTA:
            transformer_model = TransformerModel("roberta-base", num_labels)
            file = "predictions/roberta_predictions.txt"
            dataset = transformer_model.tokenize(dataset)
            predictions, labels = transformer_model.run(dataset["train"], dataset["test"])

        elif model == Model.BART:
            transformer_model = TransformerModel("facebook/bart-base", num_labels)
            file = "predictions/bart_predictions.txt"
            dataset = transformer_model.tokenize(dataset)
            predictions, labels = transformer_model.run(dataset["train"], dataset["test"])

        elif model == Model.GPT2:
            transformer_model = TransformerModel("gpt2", num_labels)
            file = "predictions/gpt2_predictions.txt"
            dataset = transformer_model.tokenize(dataset)
            predictions, labels = transformer_model.run(dataset["train"], dataset["test"])

        else:
            raise Exception("No such model")

        f = open(file, "a")

        for i in range(len(predictions)):
            f.write(str(np.argmax(predictions[i])) + ' ' + str(labels[i]) + "\n")

        f.close()

        # results.append({"model": model, "results": transformer_model.run(dataset["train"], dataset["test"])})


# Before evaluating using ReadNet, please train it first by running readnet.py
run_on_corpus(corpus=Corpus.WEEBIT, num_labels=5, models=[Model.BERT])
