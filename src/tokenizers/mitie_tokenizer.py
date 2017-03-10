import logging
import re

from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component


class MitieTokenizer(Tokenizer, Component):
    name = "tokenizer_mitie"

    context_provides = ["tokens"]

    def __init__(self):
        pass

    def tokenize(self, text):
        import mitie

        return [w.decode('utf-8') for w in mitie.tokenize(text.encode('utf-8'))]

    def process(self, text):
        return {
            "tokens": self.tokenize(text)
        }

    def tokenize_with_offsets(self, text):
        import mitie

        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w.decode('utf-8') for w in mitie.tokenize(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            if m is None:
                message = "Invalid MITIE offset. Token '{}' in message '{}'.".format(str(tok),
                                                                                     str(text.encode('utf-8')))
                raise ValueError(message)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets
