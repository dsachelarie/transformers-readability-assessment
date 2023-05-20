from transformer_model import TransformerModel
from utils.corpus import Corpus
from utils.model import Model
from utils import utils


WEEBIT_PATH = "WeeBitBalanced"
NEWSELA_PATH = ""


def run_on_corpus(corpus: Corpus, num_labels: int, models: [Model], reload=False):
    results = []

    if corpus == Corpus.WEEBIT:
        dataset = utils.get_dataset(corpus, WEEBIT_PATH, reload)
    else:
        dataset = utils.get_dataset(corpus, NEWSELA_PATH, reload)

    for model in models:
        if model == Model.READNET:
            # TODO change to readnet code
            transformer_model = TransformerModel("bert-base-uncased", num_labels)
        elif model == Model.BERT:
            transformer_model = TransformerModel("bert-base-uncased", num_labels)
        elif model == Model.ROBERTA:
            transformer_model = TransformerModel("roberta-base", num_labels)
        elif model == Model.BART:
            transformer_model = TransformerModel("facebook/bart-base", num_labels)
        elif model == Model.GPT2:
            transformer_model = TransformerModel("gpt2", num_labels)
        else:
            raise Exception("No such model")

        dataset = transformer_model.tokenize(dataset)
        results.append({"model": model, "results": transformer_model.run(dataset["train"], dataset["test"])})

    return results


print(run_on_corpus(corpus=Corpus.WEEBIT, num_labels=5, models=[Model.BERT]))
