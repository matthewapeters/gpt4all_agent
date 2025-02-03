"""jarvus.py: A simple voice assistant that uses the GPT-4-All API for natural language processing and TTS for speech synthesis."""

import json
import pygame
import sounddevice as sd
import subprocess


import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
from .agent import Agent

pygame.mixer.init()


def systemcheck(what_i_do: str, agent: Agent):
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
        output_to_read = "\n".join(output)
        ack = agent.query_api(f"SYSTEMCHECKRESULT: \n{output_to_read}")
        if ack:
            print(ack)
            agent.say(ack)
    except Exception as e:
        print(e)


def main():
    agent = Agent()
    pygame.mixer.init()
    listening = True
    agent.say_hello()

    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=agent.callback,
    ):
        while True:
            data = agent.audio_queue.get()
            if agent.recognizer.AcceptWaveform(data):
                # extract the text from the STT
                what_i_heard = json.loads(agent.recognizer.Result())["text"]

                # change of topic clears short-term memory
                if "change of topic" in what_i_heard:
                    agent.short_term_memory = []
                    continue

                # goodbye ends the application
                if "goodbye" in what_i_heard:
                    agent.say_goodbye()
                    return

                # jarvus (and homophones) triggers the assistant
                # system check prompts the assistant to generate a system command (BASH)
                if "jarvis" in what_i_heard or what_i_heard.startswith(
                    ("jarvis", "jarvus", "drivers", "system check")
                ):
                    # remove the trigger word
                    what_i_heard = what_i_heard.replace("jarvis", "")
                    # what does the GPT respond with?
                    what_i_do = agent.query_api(what_i_heard)

                    if "system check" in what_i_heard and what_i_do is not None:
                        systemcheck(what_i_do, agent)
                    else:
                        print(what_i_do)
                        if what_i_do is not None:
                            agent.say(what_i_do)
                else:
                    if what_i_heard != "huh" and what_i_heard != "":
                        print("noise?  " + what_i_heard)


if __name__ == "__main__":
    main()
