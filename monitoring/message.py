import os
import requests, json
from dotenv import load_dotenv

load_dotenv(verbose=True)

def message(text):
    SLACK_WEBHOOK_TOKEN = os.getenv('SLACK_WEBHOOK_TOKEN')
    CHANNEL = os.getenv('CHANNEL')
    URL = 'https://hooks.slack.com/services/' + SLACK_WEBHOOK_TOKEN
    print(URL)
    data = {"username": "Training-bot", "channel": CHANNEL, "text": text, "icon_emoji": ":gem:"}
    requests.post(URL, data=json.dumps(data))