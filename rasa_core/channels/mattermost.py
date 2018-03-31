from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from flask import Blueprint, request, jsonify
import requests
from mattermostwrapper import MattermostAPI
from rasa_core.channels.channel import UserMessage, OutputChannel
from rasa_core.channels.rest import HttpInputComponent

logger = logging.getLogger(__name__)


class MattermostBot(MattermostAPI, OutputChannel):
    """A Mattermost communication channel"""

    def __init__(self, url, team, user, pw, bot_channel):
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw
        self.bot_channel = bot_channel
        super(MattermostBot, self).__init__(url, team)
        super(MattermostBot, self).login(user, pw)
        print(self.bot_channel)


    def send_text_message(self, recipient_id, message):
        super(MattermostBot, self).post_channel(bot_channel, message)


class MattermostInput(HttpInputComponent):
    """Mattermost input channel implemenation. Based on the HTTPInputChannel."""

    def __init__(self, url, team, user, pw):
        # type: (Text, Text) -> None
        """Create a Mattermost input channel.
        Needs a couple of settings to properly authenticate and validate
        messages.
        :param url: Your Mattermost team url including /v4 example https://mysite.example.com/api/v4

        :param team: Your mattermost team name

        :param user: Your mattermost userid that will post messages

        :param pw: Your mattermost password for your user
        """
        self.url = url
        self.team = team
        self.user = user
        self.pw = pw

    def blueprint(self, on_new_message):
        mattermost_webhook = Blueprint('mattermost_webhook', __name__)

        @mattermost_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @mattermost_webhook.route("/webhook", methods=['POST'])
        def webhook():
            request.get_data()
            if request.json:
                output = request.json
                text = output['text']
                sender_id = output['user_id']
                bot_channel = output['channel_id']
                print(output)
                out_channel = MattermostBot(self.url, self.team, self.user, self.pw, bot_channel)
                user_msg = UserMessage(text, out_channel, sender_id)
                on_new_message(user_msg)
                return "success"

        return mattermost_webhook
