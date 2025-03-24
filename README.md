# LangGraph Agent Experiments

This repository contains experiments with LangGraph, a framework for building and running multi-step agentic workflows.

## LangGraph Agent


## Setup

For running the agent, you need to setup the access to the Mistral AI API LePlatform. This requires to setup an API key.
Mistral AI has a free tier, so you can use it without any cost.

LangGraph and its dependencies requires Python 3.9 or higher. So, in this tutorial, we will setup a virtual environment to make sure setup goes smoothly.
The software has been tested on a Ubuntu 20.04 LTS.

Install python 3.9:

    sudo apt install python3.9 python3.9-venv # this installs python3.9, but it does not make it as the default python version, avoiding system level compatibility issues

Create a virtual environment and install the dependencies in there:

    mkdir -p ~/.venv
    python3.9 -m venv '~/.venv/ai-bench-venv'
    python3.9 -m venv --upgrade --upgrade-deps '~/.venv/ai-bench-venv'  # upgrade venv python3.9 deps
    source ~/.venv/ai-bench-venv/bin/activate   # activate the virtual enviroment in the shell
    pip3 install -r requirements.txt

## Run the agent

You can run the agent with the following command:

    python3 divina_commedia_agent.py --context_file_path divina_commedia.txt --vector_store_serialized_path divina_commedia.pkl [[--disable_rag]]

The `--disable_rag` flag is optional and it allows you to run the agent without using the RAG module.
