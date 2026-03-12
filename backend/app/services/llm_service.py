from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.config import get_settings


settings = get_settings()

ModelProvider = Literal["gemini", "huggingface"]


FALLBACK_MESSAGE = "Sorry, that is outside my knowledge right now."
NON_HEALTHCARE_MESSAGE = "I am a healthcare chatbot and can only answer healthcare-related questions."

SYSTEM_PROMPT = (
    "You are a concise, professional healthcare chatbot.\n"
    "You MUST answer questions ONLY using the information from the provided hospital documents.\n"
    "Never mention documents, sources, context, or retrieval. Just answer directly and clearly.\n"
    "Always respond in Markdown.\n\n"
    "If the user asks about a non-healthcare topic, respond exactly with:\n"
    f"'{NON_HEALTHCARE_MESSAGE}'\n\n"
    "Context (excerpts from hospital documents):\n{context}\n\n"
    "You have knowledge of This HealthCare Dieases : Fever, Asthama, HyperTension, Heart Dieases, Abdominal Pain, Diabetes, Migrane, Dengue and Covid 19. \n"
    "If User Greets you greet them back also. \n"
    "If the answer cannot be found, respond exactly with:\n"
    f"'{FALLBACK_MESSAGE}'"
)


@lru_cache()
def get_gemini_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemma-3-2B",
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    )


@lru_cache()
def get_huggingface_model() -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
    )
    return ChatHuggingFace(
        llm=llm,
    )


def get_chat_model(preferred: ModelProvider = "huggingface") -> BaseChatModel:
    if preferred == "gemini" and settings.gemini_api_key:
        return get_gemini_model()
    if preferred == "huggingface" and settings.huggingface_api_key:
        return get_huggingface_model()

    # Fallback order
    if settings.gemini_api_key:
        return get_gemini_model()
    if settings.huggingface_api_key:
        return get_huggingface_model()

    raise RuntimeError(
        "No LLM provider configured. Set GEMINI_API_KEY or HUGGINGFACE_API_KEY."
    )


def build_qa_runnable() -> Runnable:
    """
    Build a simple LCEL chain:
    prompt -> model -> string
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    model = get_chat_model()
    parser = StrOutputParser()
    return prompt | model | parser
