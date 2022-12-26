from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from transformers import RobertaTokenizer

class MiSeComDataLoader:
    def __init__(self, pretrained_ck: str, max_length: int):
        dataset = load_dataset('csv', data_files=["data/format_data.csv"])['train']
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        test_valid_dataset = dataset.pop('test')
        test_valid_dataset = test_valid_dataset.train_test_split(test_size=0.5, seed=42)
        dataset['validation'] = test_valid_dataset.pop('train')
        dataset['test'] = test_valid_dataset.pop('test')
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_ck)
        self.max_length = max_length
        self.dataset = dataset.map(
            self.__tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

    def __tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["Input"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        all_labels = examples["Output"]
        tokenized_inputs["labels"] = [[int(l) for i, l in enumerate(labels) if i != 0] for labels in all_labels]
        return tokenized_inputs

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in encoded_inputs.items() if k != 'labels'}
        batch['labels'] = torch.tensor(encoded_inputs['labels'], dtype=torch.float32)
        return batch

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "test", "validation"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
