# Missing Sentence Components

Multilabel classification with RoBERTa from huggingface library

## How to use

```bash
pip install MiSeCom evaluate
```

```python
from MiSeCom import MissSentComp
miss_sentence_component = MissSentComp('transZ/misecom', 'roberta-base')
sent = "I education company."
print(miss_sentence_component(sent)) # I education company <ma> <mp> <mv>
```
