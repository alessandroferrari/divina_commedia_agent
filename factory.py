import getpass
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
import os


def model_factory(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    if "mistral" in model_name:
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = getpass.getpass(
                "Enter your Mistral API key:")
    if "gpt" in model_name:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass(
                "Enter your OpenAI API key:")
    if model_name == "mistral-small":
        return ChatMistralAI(model="mistral-small", temperature=temperature)
    elif model_name == "mistral-large":
        return ChatMistralAI(model="mistral-large", temperature=temperature)
    elif model_name == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    elif model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif model_name == "gpt-3.5-turbo":
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
    else:
        raise ValueError(f"Model {model_name} not found")
