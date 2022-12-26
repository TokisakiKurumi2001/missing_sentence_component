class LabelConverter:
    def __init__(self):
        self.label2id = {'<ma>': 0, '<madv>': 1, '<mn>': 2, '<mp>': 3, '<ms>': 4, '<mv>': 5}
        self.id2label = [k for k in self.label2id.keys()]
        self.abbr2full = {
            "<ma>": "Missing article",
            "<madv>": "Missing adverb",
            "<mn>": "Missing noun phrase",
            "<mp>": "Missing preposition",
            "<ms>": "Missing subject",
            "<mv>": "Missing verb",
        }