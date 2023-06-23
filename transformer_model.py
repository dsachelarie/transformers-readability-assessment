from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset


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
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="trainer"),
            train_dataset=train_data,
        )
        trainer.train()
        output = trainer.predict(test_data)

        if self.model_name == "facebook/bart-base":
            return output.predictions[0], output.label_ids

        return output.predictions, output.label_ids
