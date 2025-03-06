"""
jarvus.agent
"""

from datetime import datetime
from TTS.api import TTS
from typing import Optional
from vosk import Model, KaldiRecognizer
import configparser
import json
from nltk.tokenize import sent_tokenize
import os
import pygame
import queue
import random
import re
import requests
import threading
import time
import torch

from jarvus import SPEAKERS
from jarvus.posix_paths import detect_posix_path
from jarvus.ap_news import get_article, get_latest_news


class Agent:
    """
    Agent

    A class to interact with the v1/chats/completion API hosted at http://localhost:5000.

    Attributes:
        api_url (str): The URL of the API endpoint.
    """

    def load_config(self, file_path="./config.ini"):
        config = configparser.ConfigParser()
        try:
            config.read(file_path)
            agent_voice = config.getint("agent", "voice")
            return {"agent": {"voice": agent_voice}}
        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            print(f"Configuration error: {e}")
            return None

    def __init__(self, api_url: str = "http://localhost:5000/v1/chat/completions"):
        """
        Initializes a new instance of GPT4AllAgent.

        Args:
            api_url (str): The URL of the API endpoint. Defaults to 'http://localhost:5000/v1/chats/completion'.
        """
        conf = self.load_config()

        self.api_url = api_url
        self.short_term_memory = []
        self.random_speaker_idx = -1
        self.configured_speaker_id = conf.get("agent", {"voice": -1}).get("voice", -1)
        self.urls = []

        # TTS Model
        self.TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

        # Get device
        self.tts_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)

        # Init TTS
        self.tts_model = TTS(self.TTS_MODEL).to(self.tts_device)
        print("TTS initialized")

        self.is_playing = threading.Lock()

        # Init STT
        self.stt_model = Model("vosk_model")
        self.recognizer = KaldiRecognizer(self.stt_model, 16000)
        self.audio_queue = queue.Queue()
        print("STT initialized")

    @property
    def speaker_voice(self) -> int:
        if self.configured_speaker_id == -1:
            # Change the speaker - for testing
            self.random_speaker_idx += 1
            if self.random_speaker_idx == len(SPEAKERS) - 1:
                self.random_speaker_idx = 0
            print(
                f"Speaker {self.random_speaker_idx}: "
                f"{SPEAKERS[self.random_speaker_idx]}"
            )

            return self.random_speaker_idx
        else:
            print(
                f"Speaker {self.configured_speaker_id}: "
                f"{SPEAKERS[self.configured_speaker_id]}"
            )
            return self.configured_speaker_id

    def query_api(self, chat_query: str) -> Optional[str]:
        """
        Sends a chat query to the GPT4All API and returns the response.

        Args:
            chat_query (str): The chat query to send to the API.

        Returns:
            Optional[str]: The response from the API if successful; otherwise, None.
        """
        chat_query = chat_query.upper()
        do_bash = False
        do_python = False
        do_syscheck_result = False
        do_daily_feed = False
        model = "Llama 3 8B Instruct"

        self.short_term_memory.append({"role": "user"})

        if "SYSTEMCHECKRESULT" in chat_query:
            chat_query = chat_query.replace("SYSTEMCHECKRESULT", "")
            do_syscheck_result = True

        if do_syscheck_result:
            self.short_term_memory[-1]["prepend"] = (
                "These are the results from the code you provided."
                "Summarize this information tersely (do not include any markdown or conversation). "
                "Response may be conversational or technical. Replace commonly abreviated terms with their full names. "
                "Do not explain that you are simply an AI assistant. "
                "Results: "
            )

        if (
            "DAILY FEED" in chat_query
            or "TODAY'S HEADLINES" in chat_query
            or "DAILY NEWS" in chat_query
            or "TODAY'S NEWS" in chat_query
        ):
            do_daily_feed = True
            self.short_term_memory[-1]["prepend"] = (
                "In LocalDocs there is a file named today.txt, which contains today's feed of Associated Press headlines. "
                "The file is structured like this:\n"
                "[headline:] Dogs and Cats Best Friends\n[url:] https://stories.com/dogs-and-cats-best-friends\n\n"
                "Review the document, as I have questions about its contents. Respond only with relevant content from this file.  "
                "Do not refer to the file by name when responding. "
                "Only respond with information found in this file. "
                "If no corresponding headlines are found, respond with 'I did not find anything about xxx' where xxx is the requested subject."
                "When corresponding content is found, respond with the corresponding text from the file, including the tags for [headline:] and [url]."
                "Finally, summarize any findings.  \nA response might look like this:\n"
                "```[headline:] Dogs and Cats Best Friends\n[url:] https://stories.com/dogs-and-cats-best-friends\nApparently, dogs and cats like each other.```\n"
            )

            model = "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF"

        if "SUMMARIZE STORY" in chat_query:
            self.get_articles()
            today = datetime.now().strftime(r"%Y%m%d")
            self.short_term_memory[-1]["prepend"] = (
                f"In LocalDocs under folder {today} there is an article related to what we have been discussing. "
                "Review the article and provide a summary of it.  Do not give the name of the article."
                "Do mention date and author if present in article."
            )
            model = "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF"

        if "SYSTEM CHECK" in chat_query:
            chat_query = chat_query.replace("SYSTEM CHECK", "")
            do_bash = True

        if "BASH" in chat_query:
            do_bash = True

        if do_bash:
            self.short_term_memory[-1]["prepend"] = (
                (
                    "Respond only in the first BASH command that satisfies this "
                    "question (do not include back-ticks, markdown or conversation): "
                ),
            )

        if "PYTHON" in chat_query:
            do_python = True

        if do_python:
            self.short_term_memory[-1]["prepend"] = (
                "Respond only with Python code that satisfies this request using best "
                "practices for docstring comments and test coverage (do not include "
                "any markdown or conversation): "
            )

        chat_query = chat_query.capitalize().strip()
        self.short_term_memory[-1]["content"] = chat_query

        # if the short term memory is too long, remove the oldest two entries
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)
            self.short_term_memory.pop(0)

        print(model, self.short_term_memory)

        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model,
                "messages": [
                    {
                        "role": m["role"],
                        "content": f"{m.get('prepend', '')}{m['content']}",
                    }
                    for m in self.short_term_memory
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
            }

            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            resp = json.loads(response.text)
            output = resp["choices"][0]["message"]["content"]

            self.short_term_memory.append({"role": "assistant", "content": output})

            if do_python or do_bash or do_syscheck_result:
                # strip out the prepend so it does not confuse future queries
                self.short_term_memory = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.short_term_memory
                ]
            return output
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def remove_markdown_list_periods(text):
        # Regular expression to match numbered lists (1., 2., 3., ...)
        # This will match any number followed by a period and a space (e.g., "1. ", "2. ")
        return re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)

    def say(self, what_to_say):
        """
        say the text using coquie and pygame

        """
        #  remove asterisks
        what_to_say = what_to_say.replace("*", "")

        # Remove list periods
        what_to_say = self.remove_markdown_list_periods(what_to_say)

        # Split the text into sentences
        sentences = [ss for s in sent_tokenize(what_to_say) for ss in s.split("\n")]

        def playback(filename):
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove(filename)

        def play_tts(sentence: str, sentence_nbr: int):
            with self.is_playing:
                tmp_spoken_output = f"/tmp/temp_{sentence_nbr}.wav"

                self.tts_model.tts_to_file(
                    text=detect_posix_path(sentence),
                    file_path=tmp_spoken_output,
                    language="en",
                    speaker=SPEAKERS[self.speaker_voice],
                )
                playback(tmp_spoken_output)

        if len(sentences) <= 3:
            for sentence_nbr, sentence in enumerate(sentences):
                if sentence:
                    if sentence.startswith("[url:] "):
                        self.urls = [sentence.replace("[url:] ", ""), *self.urls]
                        print(sentence)
                        continue
                    tts_thread = threading.Thread(
                        target=play_tts,
                        args=(sentence, sentence_nbr),
                    )
                    tts_thread.start()
        else:
            for sentence_nbr, sentence in enumerate(sentences):
                if sentence:
                    if sentence.startswith("[url:] "):
                        self.urls = [sentence.replace("[url:] ", ""), *self.urls]
                        print(sentence)
                        continue
                    play_tts(sentence, sentence_nbr)
                    # catch your breath - you have a lot to say
                    # gives user a chance to interrupt (IE Goodbye)
                    time.sleep(0.5)

    def callback(self, indata, frames, time, status):
        """
        callback function for the sounddevice stream
        suppresses the input if speech is playing
        """
        # If speech is playing, don't process the input
        with self.is_playing:
            # Process the input if it's not playing
            self.audio_queue.put(bytes(indata))

    def say_goodbye(self):
        """
        say_goodbye

        """
        goodbyes = [
            "Good bye, now!",
            "Goodbye!",
            "See you later!",
            "Toodles!",
            "Bye!",
            "Later, Gator!",
            "In a while, Crocodile",
            "Adios!",
            "Farewell!",
            "Bye-bye!",
            "Cheerio!",
            "So long!",
            "Till next time!",
            "Take care!",
            "Peace out!",
            "I'm out!",
            "Jarvus out!",
            "Don't be a stranger!",
            "Catch you later!",
            "Parting is such sweet sorrow!",
            "Goodbye, friend!",
            "Anon!",
        ]
        sign_off = goodbyes[random.randint(0, len(goodbyes) - 1)]
        print(f"{sign_off}")
        self.say(sign_off)
        return

    def say_hello(self):
        greetings = [
            "Hello!  I am Jarvus!",
            "Greetings and salutations!",
            "Hi! I am ready!",
            "Salutations!",
            "Ni Hao Ma!",
        ]

        sign_on = greetings[random.randint(0, len(greetings) - 1)]
        print(f"{sign_on}")
        self.say(sign_on)
        return

    def get_articles(self):
        for _ in range(len(self.urls)):
            story_url = self.urls.pop(0)
            try:
                get_article(story_url)
            except Exception as e:
                print(f"could not download story {story_url}: {e}")

    @staticmethod
    def get_latest_news():
        try:
            get_latest_news()
        except Exception as e:
            print(f"Could not download daily news feed: {e}")
