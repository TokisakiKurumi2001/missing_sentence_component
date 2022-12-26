from MiSeCom import MissSentComp
miss_sentence_component = MissSentComp('misecom_model/v1', 'roberta-base')
sent = "I education company."
print(miss_sentence_component(sent))