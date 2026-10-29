"""Creates the LangChain chat model for your Azure deployment."""
from langchain_openai import AzureChatOpenAI

from . import config


def get_llm(temperature=None, max_tokens=None):
    """Main chat model (creative-ish, streaming)."""
    return AzureChatOpenAI(
        azure_endpoint=config.AZURE_ENDPOINT,
        api_key=config.API_KEY,
        api_version=config.API_VERSION,
        azure_deployment=config.DEPLOYMENT,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        max_tokens=config.MAX_TOKENS if max_tokens is None else max_tokens,
    )


def get_extractor_llm():
    """Same deployment, but deterministic. Used for memory extraction and summaries."""
    return get_llm(temperature=0, max_tokens=600)
