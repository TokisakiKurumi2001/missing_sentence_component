from MiSeCom import MiSeComModel, LabelConverter
from transformers import RobertaTokenizer
import torch

class MissSentComp:
    def __init__(self, ckpt: str, tokenizer_ckpt: str) :
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_ckpt)
        self.model = MiSeComModel.from_pretrained(ckpt)
        self.model.eval()
        self.label_converter = LabelConverter()

    def __call__(self, sent: str):
        tokenized_inputs = self.tokenizer(sent, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**tokenized_inputs).squeeze(0)
        threshold = 0.5
        masks = [self.label_converter.id2label[i] for i, logit in enumerate(logits.tolist()) if logit > threshold]
        return sent + " " + " ".join(masks)
        