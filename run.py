from transformer_model import TransformerModel
from utils.corpus import Corpus
from utils.model import Model
from utils import utils


WEEBIT_PATH = "WeeBit-TextOnly"
NEWSELA_PATH = ""


def run_on_corpus(corpus: Corpus, models: [Model], compute_rmse=False, compute_accuracy=True, compute_per_label=False):
    results = []

    if corpus == Corpus.WEEBIT:
        data = utils.read_weebit(WEEBIT_PATH)
    elif corpus == Corpus.NEWSELA:
        data = utils.read_newsela(NEWSELA_PATH)
    else:
        raise Exception("No such corpus")

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

        tokenized_data = transformer_model.tokenize(data)
        train_data, test_data = transformer_model.split_train_test(tokenized_data)
        results.append({"model": model, "results": transformer_model.run(train_data, test_data, compute_rmse, compute_accuracy, compute_per_label)})

    return results


print(run_on_corpus(corpus=Corpus.WEEBIT, models=[Model.BART]))
