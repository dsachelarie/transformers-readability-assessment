from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
import evaluate
import numpy as np


class TransformerModel:
    def __init__(self, model_name: str, num_labels: int):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.num_labels = num_labels

    def tokenize(self, dataset: Dataset):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # define pad token for GPT-2, since no default pad token is provided
        if self.model_name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        return dataset.map(
            lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

    def run(self, train_data: Dataset, test_data: Dataset):
        # def compute_metrics(eval_pred):
        #     metric1 = evaluate.load("accuracy")
        #     metric2 = evaluate.load("mse")
        #     logits, labels = eval_pred
        #
        #     # BART provides more predictions
        #     if self.model_name == "facebook/bart-base":
        #         predictions = np.argmax(logits[0], axis=-1)
        #     else:
        #         predictions = np.argmax(logits, axis=-1)
        #
        #     return {"accuracy": metric1.compute(predictions=predictions, references=labels)["accuracy"],
        #             "mse": metric2.compute(predictions=predictions, references=labels)["mse"]}

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="trainer"),
            train_dataset=train_data,
        )
        trainer.train()
        # model_results = [{"overall_results": trainer.evaluate()}]
        #
        # for label in range(self.num_labels):
        #     data_division = test_data.filter(lambda element: element["label"] == label)
        #
        #     model_results.append({"label": label, "per_label_results": trainer.evaluate(eval_dataset=data_division)})
        #
        # return model_results

        output = trainer.predict(test_data)

        if self.model_name == "facebook/bart-base":
            return output.predictions[0]

        return output.predictions, output.label_ids
