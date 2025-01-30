from gtts import gTTS
from typing import Optional
from vosk import Model, KaldiRecognizer
import json
import os
import pygame
import queue
import requests
import sounddevice as sd
import subprocess
import sys
import threading

is_playing = False

model = Model("model")
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()


def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


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

    def query_api(self, chat_query: str) -> Optional[str]:
        """
        Sends a chat query to the GPT4All API and returns the response.

        Args:
            chat_query (str): The chat query to send to the API.

        Returns:
            Optional[str]: The response from the API if successful; otherwise, None.
        """
        chat_query = chat_query.upper()
        if "SYSTEM CHECK" in chat_query:
            chat_query = f"BASH {chat_query}"
            chat_query = chat_query.replace("SYSTEM CHECK", "")

        chat_query = chat_query.replace(
            "BASH",
            (
                "Respond only in the first BASH command that satisfies this "
                "question (do not include back-ticks):"
            ),
        )
        chat_query = chat_query.replace(
            "PYTHON",
            (
                "Respond only with Python code that satisfies this request "
                "using best practices for docstring comments and test coverage "
                "(do not include any back-ticks, markdown or conversation): "
            ),
        )

        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "Llama 3 8B Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": chat_query,
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
            }

            response = requests.post(self.api_url, json=data, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            resp = json.loads(response.text)
            return resp["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None


def say(what_to_say):
    global is_playing

    is_playing = True

    def play_tts():
        tts = gTTS(text=what_to_say, lang="en-au", slow=False)
        tts.save("/tmp/temp.mp3")
        pygame.mixer.music.load("/tmp/temp.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.remove("/tmp/temp.mp3")
        global is_playing
        is_playing = False

    tts_thread = threading.Thread(target=play_tts)
    tts_thread.start()


def callback(indata, frames, time, status):
    global is_playing

    # If speech is playing, don't process the input
    if is_playing:
        return

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
                    if "system" in what_i_heard:
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
                            except Exception as e:
                                print(e)
                    else:
                        print(what_i_do)
                        say(what_i_do.replace("*", ""))

                else:
                    print("you talking to me?")
                    print(what_i_heard)


if __name__ == "__main__":
    main()
