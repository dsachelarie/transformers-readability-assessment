from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
import evaluate
import numpy as np


class TransformerModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        """, num_labels=3"""

    def tokenize(self, data):
        dataset = Dataset.from_list(data)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return dataset.map(
            lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

    @staticmethod
    def split_train_test(dataset):
        dataset = dataset.train_test_split(train_size=0.8, shuffle=True)

        return dataset['train'], dataset['test']

    def run(self, train_data, test_data, compute_rmse, compute_accuracy, compute_per_label):
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="steps", eval_steps=1)

        # Evaluate
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
        )

        # trainer.train()
