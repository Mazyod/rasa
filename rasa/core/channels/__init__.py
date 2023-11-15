from typing import Text, Dict, List, Type

from rasa.core.channels.channel import (  # noqa: F401
    InputChannel,
    OutputChannel,
    UserMessage,
    CollectingOutputChannel,
)

# this prevents IDE's from optimizing the imports - we need to import the
# above first, otherwise we will run into import cycles
from rasa.core.channels.callback import CallbackInput
from rasa.core.channels.console import CmdlineInput
from rasa.core.channels.rasa_chat import RasaChatInput
from rasa.core.channels.rest import RestInput

input_channel_classes: List[Type[InputChannel]] = [
    CmdlineInput,
    RasaChatInput,
    CallbackInput,
    RestInput,
]

# Mapping from an input channel name to its class to allow name based lookup.
BUILTIN_CHANNELS: Dict[Text, Type[InputChannel]] = {
    c.name(): c for c in input_channel_classes
}
