# GPT4All Agent -- Jarvus

- [GPT4All Agent -- Jarvus](#gpt4all-agent----jarvus)
  - [Summary](#summary)
  - [Platform Requirements](#platform-requirements)
  - [Installing the gpt4All Application](#installing-the-gpt4all-application)
    - [Download and Install](#download-and-install)
    - [Configure the Server](#configure-the-server)
  - [Installing the Vosk Speech-To-Text Model](#installing-the-vosk-speech-to-text-model)
  - [Installing the Virtual Environment](#installing-the-virtual-environment)
  - [Running](#running)
  - [Where Are You Going With This?](#where-are-you-going-with-this)
  - [What About Artwork](#what-about-artwork)

## Summary

Ever wanted your own AI assistant like Tony Stark had?  This project cobles together existing resources to create something similar.  It utilizes the gpt4All application, which, along with a desktop interface, provides a backend server that Javus will talk to.  It uses Vosk to provide speech-to-text, but ignores anything it hears unless it hears its name (sounds like "Jarvis").  The agent runs until it hears "Goodbye".  Javus uses Vosk for speech-to-text (stt) and Coqui.ai for text-to-speech (tts).

Javus is being outfitted with the ability to "know" a bit about itself: if it hears "Jarvus, system check ...." it will attempt to translate your request to a BASH statement, execute it, and read the results back to you.  

**NOTE**: it is not the AI that is invoking BASH - it is the python application, *so do not run it as root!*  

This is really just a proof of concept - build and run it at your own risk, but I am happy to hear if this project sparks some creative moments on your end!

## Platform Requirements

Javus was built at tested on the following platform.

```bash

   OS: Ubuntu 24.10 x86_64 
   Kernel: 6.11.0-14-generic 
   Shell: bash 5.2.32 
   CPU: AMD Ryzen 9 7950X (32) @ 5.881G 
   GPU: AMD ATI 6a:00.0 Raphael 
   GPU: NVIDIA GeForce RTX 4090 
   Memory: 8471MiB / 62414MiB 

```

## Installing the gpt4All Application

### Download and Install

[https://www.nomic.ai/gpt4all](https://www.nomic.ai/gpt4all)

### Configure the Server

Make note of the port, or change the port to one you will use from your Jarvus Agent.

## Installing the Vosk Speech-To-Text Model

```bash
 wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
 unzip vosk-model-small-en-us-0.15.zip
 mv vosk-model-small-en-us-0.15 model
```

## Installing the Virtual Environment

I used Python 3.10.16

```bash
pyenv virtualenv 3.10.16 gpt4all_agent
pyenv activate gpt4all_agent
pip install -U pip
pip install -r requirements.txt
```

## Running

1. Start gpt4All
2. Navigate to Chats and click Server Chat.  Note the antena icon in the lower-left (above the Nomic logo).  When it animates, the gpt4All server is running
3. Launch the Jarvus agent

    ```bash
    ./jarvus.sh
    ```  

4. Talk to Jarvus. Try things like:  
     a. `Good Morning, Jarvus`  
     b. `system check: what is the current date and time?`  
     c. `system check: how much free space is left on my root file system?`  
5. When you are done, say "Goodbye".

## Where Are You Going With This?

Current AI services like gpt4All and OpenAI's chatGPT use models that are trained on large volumes of data, but the services do not go out and find new information to work with -- it relies on what it was trained on.  There are some features of gpt4All such as the `LocalDocs` feature that allows you to expand the model's knowledge base.  By creating a custom agent between ourselves and the AI, we can create controled automation to help our productivity.

**Just... do not run as root -- No Voltrons, please!**

## What About Artwork

I might look into integrating with A1111 Stable Diffusion WebUI at some point in the future.
