from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
from multiprocessing import Queue
from threading import Thread
from time import sleep

from rasa_core import utils
from typing import Text, List, Dict, Any, Optional, Callable, Iterable
from flask import Blueprint, jsonify, request, Flask, Response

try:
    from urlparse import urljoin
except ImportError:
    from urllib.parse import urljoin


class UserMessage(object):
    """Represents an incoming message.

     Includes the channel the responses should be sent to."""

    DEFAULT_SENDER_ID = "default"

    def __init__(self, text, output_channel=None, sender_id=None):
        # type: (Optional[Text], Optional[OutputChannel], Text) -> None

        self.text = text

        if output_channel is not None:
            self.output_channel = output_channel
        else:
            self.output_channel = CollectingOutputChannel()

        if sender_id is not None:
            self.sender_id = sender_id
        else:
            self.sender_id = self.DEFAULT_SENDER_ID


# TODO: TB - ensure backward compatibility with old webhooks
def register_blueprints(input_channels, app, on_new_message, route):
    # type: (List[InputChannel], Flask, Callable[[UserMessage], None]) -> None

    for channel in input_channels:
        p = urljoin(route, channel.url_prefix())
        app.register_blueprint(channel.blueprint(on_new_message), url_prefix=p)


class InputChannel(object):

    def url_prefix(self):
        return type(self).__name__

    def blueprint(self, on_new_message):
        # type: (Callable[[UserMessage], None])-> None
        """Defines a Flask blueprint.

        The blueprint will be attached to a running flask server and handel
        incoming routes it registered for."""
        raise NotImplementedError(
                "Component listener needs to provide blueprint.")


class OutputChannel(object):
    """Output channel base class.

    Provides sane implementation of the send methods
    for text only output channels."""

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Send a message through this channel."""

        raise NotImplementedError("Output channel needs to implement a send "
                                  "message for simple texts.")

    def send_image_url(self, recipient_id, image_url):
        # type: (Text, Text) -> None
        """Sends an image. Default will just post the url as a string."""

        self.send_text_message(recipient_id, "Image: {}".format(image_url))

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        # type: (Text, Text, List[Dict[Text, Any]], **Any) -> None
        """Sends buttons to the output.

        Default implementation will just post the buttons as a string."""

        self.send_text_message(recipient_id, message)
        for idx, button in enumerate(buttons):
            button_msg = "{idx}: {title} ({val})".format(
                    idx=idx + 1, title=button['title'], val=button['payload'])
            self.send_text_message(recipient_id, button_msg)

    def send_custom_message(self, recipient_id, elements):
        # type: (Text, Iterable[Dict[Text, Any]]) -> None
        """Sends elements to the output.

        Default implementation will just post the elements as a string."""

        for element in elements:
            element_msg = "{title} : {subtitle}".format(
                    title=element['title'], subtitle=element['subtitle'])
            self.send_text_with_buttons(
                    recipient_id, element_msg, element['buttons'])


class CollectingOutputChannel(OutputChannel):
    """Output channel that collects send messages in a list

    (doesn't send them anywhere, just collects them)."""

    def __init__(self):
        self.messages = []

    def latest_output(self):
        if self.messages:
            return self.messages[-1]
        else:
            return None

    def send_text_message(self, recipient_id, message):
        self.messages.append({"recipient_id": recipient_id,
                              "text": message})

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        self.messages.append({"recipient_id": recipient_id,
                              "text": message,
                              "data": buttons})


class QueueOutputChannel(OutputChannel):
    """Output channel that collects send messages in a list

    (doesn't send them anywhere, just collects them)."""

    def __init__(self, message_queue=None):
        # type: (Queue) -> None
        self.messages = Queue() if not message_queue else message_queue

    def send_text_message(self, recipient_id, message):
        self.messages.put({"recipient_id": recipient_id,
                           "text": message})

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        self.messages.put({"recipient_id": recipient_id,
                           "text": message,
                           "data": buttons})


class RestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa Core and
    retrieve responses from the agent."""

    def url_prefix(self):
        return "rest"

    @staticmethod
    def on_message_wrapper(on_new_message, text, queue, sender_id):
        collector = QueueOutputChannel(queue)

        message = UserMessage(text, collector, sender_id)
        on_new_message(message)

        queue.put("DONE")

    def stream_response(self, on_new_message, text, sender_id):
        from multiprocessing import Queue

        q = Queue()

        t = Thread(target=self.on_message_wrapper,
                   args=(on_new_message, text, q, sender_id))
        t.start()
        while True:
            response = q.get()
            if response == "DONE":
                break
            else:
                yield json.dumps(response) + "\n"

    def blueprint(self, on_new_message):
        custom_webhook = Blueprint('custom_webhook', __name__)

        @custom_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @custom_webhook.route("/webhook", methods=['POST'])
        def receive():
            payload = request.json
            sender_id = payload.get("sender", None)
            text = payload.get("message", None)
            should_use_stream = utils.bool_arg("stream", default=False)

            if should_use_stream:
                return Response(
                        self.stream_response(on_new_message, text, sender_id),
                        content_type='text/event-stream')
            else:
                collector = CollectingOutputChannel()
                on_new_message(UserMessage(text, collector, sender_id))
                return jsonify(collector.messages)

        return custom_webhook
