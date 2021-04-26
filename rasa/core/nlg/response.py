import copy
import logging

from rasa.shared.core.trackers import DialogueStateTracker
from typing import Text, Any, Dict, Optional, List

from rasa.core.nlg import interpolator
from rasa.core.nlg.generator import NaturalLanguageGenerator

logger = logging.getLogger(__name__)


class TemplatedNaturalLanguageGenerator(NaturalLanguageGenerator):
    """Natural language generator that generates messages based on responses.

    The responses can use variables to customize the utterances based on the
    state of the dialogue.
    """

    def __init__(self, responses: Dict[Text, List[Dict[Text, Any]]]) -> None:
        """Creates a Template Natural Language Generator.

        Args:
            responses: responses that will be used to generate messages.
        """
        self.responses = responses
        print(self.responses)

    def _matches_filled_slots(
        self, filled_slots: Dict[Text, Any], response: Dict[Text, Any],
    ) -> bool:
        """Checks if the conditional response variation matches the filled slots."""
        condition_list = response.get("condition")
        for condition in condition_list:
            name = condition["name"]
            value = condition["value"]
            if filled_slots.get(name) == value:
                return True
        else:
            return False

    def _responses_for_utter_action(
        self, utter_action: Text, output_channel: Text, filled_slots: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        """Return array of responses that fit the channel, action and condition if any."""
        default_responses = []
        conditional_responses = []

        for response in self.responses[utter_action]:
            if response.get("condition") is None:
                default_responses.append(response)
            else:
                matched_response = self._matches_filled_slots(
                    filled_slots=filled_slots, response=response
                )
                if matched_response:
                    conditional_responses.append(response)

        if conditional_responses:
            potential_responses = conditional_responses[:]
        else:
            potential_responses = default_responses[:]

        channel_responses = list(
            filter(lambda x: (x.get("channel") == output_channel), potential_responses)
        )

        # always prefer channel specific responses over default ones
        if channel_responses:
            return channel_responses
        else:
            return potential_responses

    # noinspection PyUnusedLocal
    def _random_response_for(
        self, utter_action: Text, output_channel: Text, filled_slots: Dict[Text, Any]
    ) -> Optional[Dict[Text, Any]]:
        """Select random response for the utter action from available ones.

        If channel-specific responses for the current output channel are given,
        only choose from channel-specific ones.
        """
        import numpy as np

        if utter_action in self.responses:
            suitable_responses = self._responses_for_utter_action(
                utter_action, output_channel, filled_slots
            )

            if suitable_responses:
                return np.random.choice(suitable_responses)
            else:
                return None
        else:
            return None

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action."""
        filled_slots = tracker.current_slot_values()
        return self.generate_from_slots(
            utter_action, filled_slots, output_channel, **kwargs
        )

    def generate_from_slots(
        self,
        utter_action: Text,
        filled_slots: Dict[Text, Any],
        output_channel: Text,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action."""
        # Fetching a random response for the passed utter action
        r = copy.deepcopy(
            self._random_response_for(utter_action, output_channel, filled_slots)
        )
        # Filling the slots in the response with placeholders and returning the response
        if r is not None:
            return self._fill_response(r, filled_slots, **kwargs)
        else:
            return None

    def _fill_response(
        self,
        response: Dict[Text, Any],
        filled_slots: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> Dict[Text, Any]:
        """Combine slot values and key word arguments to fill responses."""
        # Getting the slot values in the response variables
        response_vars = self._response_variables(filled_slots, kwargs)

        keys_to_interpolate = [
            "text",
            "image",
            "custom",
            "buttons",
            "attachment",
            "quick_replies",
        ]
        if response_vars:
            for key in keys_to_interpolate:
                if key in response:
                    response[key] = interpolator.interpolate(
                        response[key], response_vars
                    )
        return response

    @staticmethod
    def _response_variables(
        filled_slots: Dict[Text, Any], kwargs: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        """Combine slot values and key word arguments to fill responses."""
        if filled_slots is None:
            filled_slots = {}

        # Copying the filled slots in the response variables.
        response_vars = filled_slots.copy()
        response_vars.update(kwargs)
        return response_vars
