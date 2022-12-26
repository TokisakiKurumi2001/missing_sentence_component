import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from MiSeCom import MiSeComModel, MiSeComConfig, LabelConverter
import evaluate
import numpy as np

class LitMiSeCom(pl.LightningModule):
    def __init__(self, pretrained_ck: str, layers_use_from_last: int, method_for_layers: str, lr: float):
        super(LitMiSeCom, self).__init__()
        label_converter = LabelConverter()
        self.id2label = label_converter.id2label
        config = MiSeComConfig.from_pretrained(
            pretrained_ck,
            pretrained_ck=pretrained_ck,
            layers_use_from_last=layers_use_from_last,
            method_for_layers=method_for_layers,
            id2label={i: label for i, label in enumerate(self.id2label)},
            label2id=label_converter.label2id)
        self.model = MiSeComModel(config)
        self.num_labels = config.num_labels
        self.loss = nn.BCELoss()
        self.lr = lr
        self.valid_metric = evaluate.load("roc_auc", "multilabel")
        self.test_metric = evaluate.load("roc_auc", "multilabel")
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def __postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy().astype(int)
        return predictions, labels

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)
        loss = self.loss(logits, labels)
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)

        decoded_preds, decoded_labels = self.__postprocess(logits, labels)
        self.valid_metric.add_batch(prediction_scores=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.valid_metric.compute()
        self.log('valid/roc_auc', results['roc_auc'], on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        logits = self.model(**batch)

        decoded_preds, decoded_labels = self.__postprocess(logits, labels)
        self.test_metric.add_batch(prediction_scores=decoded_preds, references=decoded_labels)

    def test_epoch_end(self, outputs):
        results = self.test_metric.compute()
        self.log('test/roc_auc', results['roc_auc'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer