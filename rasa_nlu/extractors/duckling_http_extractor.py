from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import requests
import simplejson
from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.extractors.duckling_extractor import extract_value
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)


class DucklingHTTPExtractor(EntityExtractor):
    """Searches for structured entites, e.g. dates, using a duckling server."""

    name = "ner_duckling_http"

    provides = ["entities"]

    defaults = {
        # by default all dimensions recognized by duckling are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None,

        # http url of the running duckling server
        "url": None
    }

    def __init__(self, component_config=None, language=None):
        # type: (Text, Text, Optional[List[Text]]) -> None

        super(DucklingHTTPExtractor, self).__init__(component_config)
        self.language = language

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> DucklingHTTPExtractor

        return DucklingHTTPExtractor(config.for_component(cls.name,
                                                          cls.defaults),
                                     config.language)

    def _duckling_parse(self, text):
        """Sends the request to the duckling server and parses the result."""

        try:
            # TODO: this is king of a quick fix to generate a proper locale
            #       for duckling. and might not always create correct
            #       locales. We should rather introduce a new config value.
            locale = "{}_{}".format(self.language, self.language.upper())
            payload = {"text": text, "locale": locale}
            headers = {"Content-Type": "application/x-www-form-urlencoded; "
                                       "charset=UTF-8"}
            response = requests.post(self.duckling_url + "/parse",
                                     data=payload,
                                     headers=headers)
            if response.status_code == 200:
                return simplejson.loads(response.text)
            else:
                logger.error("Failed to get a proper response from remote "
                             "duckling. Status Code: {}. Response: {}"
                             "".format(response.status_code, response.text))
                return []
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to duckling http server. Make sure "
                         "the duckling server is running and the proper host "
                         "and port are set in the configuration. More "
                         "information on how to run the server can be found on "
                         "github: "
                         "https://github.com/facebook/duckling#quickstart "
                         "Error: {}".format(e))
            return []

    def _filter_irrelevant_matches(self, matches):
        """Only return dimensions the user configured"""

        if self.dimensions:
            return [match
                    for match in matches
                    if match["dim"] in self.dimensions]
        else:
            return matches

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = []
        if self.component_config["duckling_url"] is not None:

            matches = self._duckling_parse(message.text)
            relevant_matches = self._filter_irrelevant_matches(matches)
            for match in relevant_matches:
                value = extract_value(match)
                entity = {
                    "start": match["start"],
                    "end": match["end"],
                    "text": match["body"],
                    "value": value,
                    "additional_info": match["value"],
                    "entity": match["dim"]}

                extracted.append(entity)
        else:
            logger.warn("Duckling HTTP component in pipeline, but no "
                        "`duckling_http_url` configuration in the config "
                        "file.")

        extracted = self.add_extractor_name(extracted)
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[DucklingHTTPExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> DucklingHTTPExtractor

        component_config = model_metadata.get(cls.name)
        return cls(component_config, model_metadata.get("language"))
