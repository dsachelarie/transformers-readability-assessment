from datasets import Dataset

from transformer_model import TransformerModel
from utils.corpus import Corpus
from utils.model import Model
from utils import utils


WEEBIT_PATH = "WeeBitBalanced"
NEWSELA_PATH = ""


def run_on_corpus(corpus: Corpus, models: [Model], compute_per_label=False):
    results = []

    if corpus == Corpus.WEEBIT:
        data = utils.read_weebit(WEEBIT_PATH)
    elif corpus == Corpus.NEWSELA:
        data = utils.read_newsela(NEWSELA_PATH)
    else:
        raise Exception("No such corpus")

    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(train_size=0.8, shuffle=True)

    for model in models:
        if model == Model.READNET:
            # TODO change to readnet code
            transformer_model = TransformerModel("bert-base-uncased")
        elif model == Model.BERT:
            transformer_model = TransformerModel("bert-base-uncased")
        elif model == Model.ROBERTA:
            transformer_model = TransformerModel("roberta-base")
        elif model == Model.BART:
            transformer_model = TransformerModel("facebook/bart-base")
        elif model == Model.GPT2:
            transformer_model = TransformerModel("gpt2")
        else:
            raise Exception("No such model")

        dataset = transformer_model.tokenize(dataset)
        results.append({"model": model, "results": transformer_model.run(dataset["train"], dataset["test"], compute_per_label)})

    return results


print(run_on_corpus(corpus=Corpus.WEEBIT, models=[Model.ROBERTA]))
