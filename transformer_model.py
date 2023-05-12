from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, GPT2Config, GPT2ForSequenceClassification
from datasets import Dataset
import evaluate
import numpy as np


class TransformerModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

    def tokenize(self, data):
        dataset = Dataset.from_list(data)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # define pad token for GPT-2, since no default pad token is provided
        if self.model_name == "gpt2":
            tokenizer.pad_token = tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        return dataset.map(
            lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

    @staticmethod
    def split_train_test(dataset: Dataset):
        dataset = dataset.train_test_split(train_size=0.8, shuffle=True)

        return dataset['train'], dataset['test']

    def run(self, train_data: Dataset, test_data: Dataset, compute_rmse: bool, compute_accuracy: bool, compute_per_label: bool):
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="no")

        # Evaluate
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data.select(range(10)),
            # eval_dataset=test_data.select(range(1)),
            # compute_metrics=compute_metrics,
        )

        trainer.train()
        selected_data = test_data.select(range(10))
        output = trainer.predict(selected_data)

        if self.model_name == "facebook/bart-base":
            return output.predictions[0]

        return output.predictions
