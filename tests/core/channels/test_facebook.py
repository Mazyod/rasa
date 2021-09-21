import logging
from rasa.core.channels.facebook import MessengerBot
from fbmessenger import MessengerClient

from _pytest.monkeypatch import MonkeyPatch

import rasa.core.run
from rasa.core import utils


# this is needed so that the tests included as code examples look better
from tests.utilities import json_of_latest_request, latest_request

logger = logging.getLogger(__name__)


def test_facebook_channel():
    from rasa.core.channels.facebook import FacebookInput

    input_channel = FacebookInput(
        fb_verify="YOUR_FB_VERIFY",
        # you need tell facebook this token, to confirm your URL
        fb_secret="YOUR_FB_SECRET",  # your app secret
        fb_access_token="YOUR_FB_PAGE_ACCESS_TOKEN"
        # token for the page you subscribed to
    )

    s = rasa.core.run.configure_app([input_channel], port=5004)

    routes_list = utils.list_routes(s)

    assert routes_list["fb_webhook.health"].startswith("/webhooks/facebook")
    assert routes_list["fb_webhook.webhook"].startswith("/webhooks/facebook/webhook")


async def test_facebook_send_custom_json():
    # This function tests cases when the custom json is a list
    # The send_custom_json function doesn't return anything. Rather
    # it calls an object MessengerClient, that will
    # then make a post request.
    # Since the purpose is to test the extraction of the recipient_id
    # by the MessengerBot.send_custom_json_list we
    # modify MessengerClient (from the fbmessenger pypackage) to
    # return the recipient ID.

    json_without_id = {
        "blocks": [
            {"type": "title", "text": {"text": "Conversation progress"}},
            {
                "type": "progression_bar",
                "text": {"text": "progression 1", "level": "1"},
            },
        ]
    }
    json_with_id = {
        "blocks": [
            {"type": "title", "text": {"text": "Conversation progress"}},
            {
                "type": "progression_bar",
                "text": {"text": "progression 1", "level": "1"},
            },
        ],
        "sender": {"id": "test_json_id"},
    }

    json_with_id_and_blocks_as_dict = {
        "blocks": {
            "type": "progression_bar",
            "text": {"text": "progression 1", "level": "1"},
        },
        "sender": {"id": "test_json_id_2"},
    }

    class TestableMessengerClient(MessengerClient):
        def __init__(self, page_access_token, **kwargs):
            self.recipient_id = ""
            super(TestableMessengerClient, self).__init__(page_access_token, **kwargs)

        def send(
            self,
            payload,
            recipient_id,
            messaging_type="RESPONSE",
            notification_type="REGULAR",
            timeout=None,
            tag=None,
        ):
            self.recipient_id = recipient_id

    messenger_client = TestableMessengerClient(page_access_token="test_token")
    messenger_bot = MessengerBot(messenger_client)
    await messenger_bot.send_custom_json(
        recipient_id="test_id", json_message=json_without_id
    )
    assert messenger_bot.messenger_client.recipient_id == "test_id"
    await messenger_bot.send_custom_json(
        recipient_id="test_id", json_message=json_with_id
    )
    assert messenger_bot.messenger_client.recipient_id == "test_json_id"
    await messenger_bot.send_custom_json(
        recipient_id="not me", json_message=json_with_id_and_blocks_as_dict
    )
    assert messenger_bot.messenger_client.recipient_id == "test_json_id_2"
