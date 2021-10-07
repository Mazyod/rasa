import logging
import pytest
from rasa.core.channels.facebook import MessengerBot
from fbmessenger import MessengerClient

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            {
                "blocks": [
                    {"type": "title", "text": {"text": "Conversation progress"}},
                    {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    },
                ]
            },
            "test_id",
        ),
        (
            {
                "blocks": [
                    {"type": "title", "text": {"text": "Conversation progress"}},
                    {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    },
                ],
                "sender": {"id": "test_json_id"},
            },
            "test_json_id",
        ),
        (
            {
                "blocks": {
                    "type": "progression_bar",
                    "text": {"text": "progression 1", "level": "1"},
                },
                "sender": {"id": "test_json_id_2"},
            },
            "test_json_id_2",
        ),
        (
            [
                {
                    "blocks": {
                        "type": "progression_bar",
                        "text": {"text": "progression 1", "level": "1"},
                    }
                },
                {"sender": {"id": "test_json_id_3"}},
            ],
            "test_json_id_3",
        ),
    ],
)
async def test_facebook_send_custom_json(test_input, expected):
    # This function tests cases when the custom json is a list
    # The send_custom_json function doesn't return anything. Rather
    # it calls an object MessengerClient, that will
    # then make a post request.
    # Since the purpose is to test the extraction of the recipient_id
    # by the MessengerBot.send_custom_json_list we
    # modify MessengerClient (from the fbmessenger pypackage) to
    # return the recipient ID.

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
        recipient_id="test_id", json_message=test_input
    )
    assert messenger_bot.messenger_client.recipient_id == expected
