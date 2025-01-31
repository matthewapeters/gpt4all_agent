"""jarvus.py: A simple voice assistant that uses the GPT-4-All API for natural language processing and TTS for speech synthesis."""

from TTS.api import TTS
from typing import Optional
from vosk import Model, KaldiRecognizer
import json
import os
import pygame
import queue
import re
import requests
import sounddevice as sd
import subprocess
import threading
import torch


import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


pygame.mixer.init()

SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Chandra MacFarland",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
    "Ige Behringer",
    "Filip Traverse",
    "Damjan Chapman",
    "Wulf Carlevaro",
    "Aaron Dreschner",
    "Kumar Dahl",
    "Eugenio Mataracı",
    "Ferran Simen",
    "Xavier Hayasaka",
    "Luis Moray",
    "Marcos Rudaski",
]

Speaker_idx = 0

# TTS Model
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Get device
tts_device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL)

# Init TTS
tts_model = TTS(TTS_MODEL).to(tts_device)
print("TTS initialized")

is_playing = threading.Lock()

# Init STT
stt_model = Model("model")
recognizer = KaldiRecognizer(stt_model, 16000)
audio_queue = queue.Queue()
print("STT initialized")


class GPT4AllAgent:
    """
    A class to interact with the v1/chats/completion API hosted at http://localhost:5000.

    Attributes:
        api_url (str): The URL of the API endpoint.
    """

    def __init__(self, api_url: str = "http://localhost:5000/v1/chat/completions"):
        """
        Initializes a new instance of GPT4AllAgent.

        Args:
            api_url (str): The URL of the API endpoint. Defaults to 'http://localhost:5000/v1/chats/completion'.
        """
        self.api_url = api_url
        self.short_term_memory = []

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

        if "SYSTEM CHECK" in chat_query:
            chat_query = chat_query.replace("SYSTEM CHECK", "")
            do_bash = True

        if "BASH" in chat_query:
            do_bash = True

        if do_bash:
            self.short_term_memory.append(
                {
                    "role": "user",
                    "remove": True,
                    "content": (
                        "Respond only in the first BASH command that satisfies this "
                        "question (do not include back-ticks):"
                    ),
                }
            )
        if "PYTHON" in chat_query:
            do_python = True

        if do_python:
            self.short_term_memory.append(
                {
                    "role": "system",
                    "content": (
                        "Respond only with Python code that satisfies this request using best "
                        "practices for docstring comments and test coverage (do not include "
                        "any markdown or conversation)"
                    ),
                }
            )

        chat_query = chat_query.capitalize()

        self.short_term_memory.append({"role": "user", "content": chat_query})

        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)

        print(self.short_term_memory)

        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "Llama 3 8B Instruct",
                "messages": [
                    {"role": m["role"], "content": m["content"]}
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

            if do_python or do_bash:
                self.short_term_memory = [
                    m for m in self.short_term_memory if "remove" not in m
                ]

            return output
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


def remove_markdown_list_periods(text):
    # Regular expression to match numbered lists (1., 2., 3., ...)
    # This will match any number followed by a period and a space (e.g., "1. ", "2. ")
    return re.sub(r"^\d+\.\s*", "", text, flags=re.MULTILINE)


def say(what_to_say):
    """
    say the text using gTTS and pygame

    """
    global Speaker_idx

    # Remove list periods
    what_to_say = remove_markdown_list_periods(what_to_say)

    # Split the text into sentences
    sentences = sent_tokenize(what_to_say)

    # Change the speaker - for testing
    Speaker_idx += 1
    if Speaker_idx == len(SPEAKERS) - 1:
        Speaker_idx = 0
    print(f"Speaker {Speaker_idx}: {SPEAKERS[Speaker_idx]}")

    def playback(filename):
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove(filename)

    def play_tts(sentence: str, sentence_nbr: int):
        global is_playing
        global Speaker_idx
        with is_playing:
            tmp_spoken_output = f"/tmp/temp_{sentence_nbr}.wav"

            tts_model.tts_to_file(
                text=sentence,
                file_path=tmp_spoken_output,
                language="en",
                speaker=SPEAKERS[Speaker_idx],
            )
            playback(tmp_spoken_output)

    for sentence_nbr, parsed_sentence in enumerate(sentences):
        for sentence in parsed_sentence.split("\n"):
            tts_thread = threading.Thread(
                target=play_tts, args=(sentence, sentence_nbr)
            )
            tts_thread.start()


def callback(indata, frames, time, status):
    """
    callback function for the sounddevice stream
    suppresses the input if speech is playing
    """
    global is_playing

    # If speech is playing, don't process the input
    with is_playing:
        # Process the input if it's not playing
        audio_queue.put(bytes(indata))


def main():
    agent = GPT4AllAgent()
    pygame.mixer.init()
    listening = True

    with sd.RawInputStream(
        samplerate=16000, blocksize=8000, dtype="int16", channels=1, callback=callback
    ):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                what_i_heard = json.loads(recognizer.Result())["text"]
                if "goodbye" in what_i_heard:
                    print("Goodbye!")
                    say("Goodbye!")
                    return

                if "jarvis" in what_i_heard:
                    what_i_heard = what_i_heard.replace("jarvis", "")
                    if "system check" in what_i_heard:
                        do_system = True
                    else:
                        do_system = False
                    what_i_do = agent.query_api(what_i_heard)
                    if do_system:
                        if what_i_do is not None:
                            print("checking...", end="")
                            print(f"{what_i_do}...", end="")
                            try:
                                result = subprocess.run(
                                    what_i_do,
                                    capture_output=True,
                                    text=True,
                                    shell=True,
                                )
                                print(f"\n{result.stdout}")
                                output = result.stdout.split("\n")
                                if len(output) <= 2:
                                    say(result.stdout)
                                else:
                                    print(output)

                                output_to_read = "\n".joiuns(output)
                                ack = agent.query_api(
                                    f"I ran what you suggested and got this result: \n {output_to_read}"
                                )
                                print(ack)
                            except Exception as e:
                                print(e)
                    else:
                        print(what_i_do)
                        if what_i_do is not None:
                            say(what_i_do.replace("*", ""))

                else:
                    if what_i_heard != "huh" and what_i_heard != "":
                        print("you talking to me?  " + what_i_heard)


if __name__ == "__main__":
    main()
