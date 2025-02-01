
SHELL=/bin/bash
nullstring =
pyenv_path = $(shell which pyenv)
three_ten_sixteen = $(shell pyenv versions | grep 3.10.16|wc -l)
vosk_model_path = $(shell ls | grep vosk_model)
pyenv_jarvus = $(shell pyenv versions | grep jarvus|wc -l)

prerequisites:
	sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev


pyenv:
	if $(pyenv_path) ; then \
		echo "pyenv already installed"; \
	else \

		curl https://pyenv.run | bash

		echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
		echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
	fi

python_3_10_16 pyenv:
	if [ $(three_ten_sixteen) -gt 0 ] ; then \
		echo "Python 3.10.16 already installed"; \
	else \
		pyenv install 3.10.16; \
	fi

virtualenv python_3_10_16:
	if [ $(pyenv_jarvus) -gt 0 ] ; then \
		echo "virtualenv jarvus already installed"; \
	else \
		pyenv 	virtualenv 3.10.16 jarvus; \
		pyenv activate jarvus; \
		pip install -U pip; \
	fi

pips virtualenv:
	pip install -r requirements.txt; \


vosk:
	if [ ! -d ./$(vosk_model_path) ]; then \
		mkdir -p ./vosk_model; \
		pushd ./vosk_model; \
		wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip ; \
		unzip vosk-model-small-en-us-0.15.zip; \
		rm vosk-model-small-en-us-0.15.zip; \
		popd; \
	else \
		echo "vosk model already downloaded"; \
	fi

clean:
	rm -rf */__pycache__

build: prerequisites vosk pips

all: clean build
