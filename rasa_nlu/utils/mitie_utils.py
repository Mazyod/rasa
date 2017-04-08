from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from builtins import str
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.model import Metadata


class MitieNLP(Component):
    name = "nlp_mitie"

    context_provides = {
        "pipeline_init": ["mitie_feature_extractor"],
    }

    def __init__(self, mitie_file, extractor=None):
        self.extractor = extractor
        self.mitie_file = mitie_file
        MitieNLP.ensure_proper_language_model(self.extractor)

    @classmethod
    def create(cls, mitie_file):
        import mitie
        extractor = mitie.total_word_feature_extractor(mitie_file)
        return MitieNLP(mitie_file, extractor)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        mitie_file = model_metadata.metadata.get("mitie_file", None)
        return cls.name + "-" + str(os.path.abspath(mitie_file)) if mitie_file else None

    def pipeline_init(self, mitie_file):
        # type: (Text) -> dict
        return {"mitie_feature_extractor": self.extractor}

    @staticmethod
    def ensure_proper_language_model(extractor):
        # type: (Optional[mitie.total_word_feature_extractor]) -> None
        import mitie

        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. Loading the model returned 'None'.")

    @classmethod
    def load(cls, mitie_file):
        # type: (Text, Text) -> MitieNLP
        return cls.create(mitie_file)

    def persist(self, model_dir):
        # type: (Text) -> dict

        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint,
            "mitie_file": self.mitie_file
        }
