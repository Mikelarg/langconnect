import os
import json
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain.embeddings import init_embeddings
from starlette.config import Config, undefined

env = Config()

IS_TESTING = env("IS_TESTING", cast=str, default="").lower() == "true"

if IS_TESTING:
    SUPABASE_URL = ""
    SUPABASE_KEY = ""
else:
    SUPABASE_URL = env("SUPABASE_URL", cast=str, default=undefined)
    SUPABASE_KEY = env("SUPABASE_KEY", cast=str, default=undefined)


_EMBEDDINGS_SINGLETON: Optional[object] = None
GIGACHAT_PROVIDER = "gigachat:"

def load_gigachat_embeddings():
    from langchain_gigachat import GigaChatEmbeddings
    llm_str = os.getenv("LANGCONNECT_EMBEDDINGS")
    return GigaChatEmbeddings(
        model=llm_str[len(GIGACHAT_PROVIDER) :],
    )

def load_embeddings():
    global _EMBEDDINGS_SINGLETON

    if _EMBEDDINGS_SINGLETON is not None:
        return _EMBEDDINGS_SINGLETON

    emb_str = os.getenv("LANGCONNECT_EMBEDDINGS")
    if emb_str is None:
        raise RuntimeError("LANGCONNECT_EMBEDDINGS is empty! Fill it with your model")

    if emb_str.startswith(GIGACHAT_PROVIDER):
        embeddings = load_gigachat_embeddings()
    else:
        embeddings = init_embeddings(emb_str)

    _EMBEDDINGS_SINGLETON = embeddings
    return embeddings


def get_embeddings() -> Embeddings:
    """Get the embeddings instance based on the environment."""
    if IS_TESTING:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=512)
    return load_embeddings()


DEFAULT_EMBEDDINGS = get_embeddings()
DEFAULT_COLLECTION_NAME = "default_collection"


# Database configuration
POSTGRES_HOST = env("POSTGRES_HOST", cast=str, default="localhost")
POSTGRES_PORT = env("POSTGRES_PORT", cast=int, default="5432")
POSTGRES_USER = env("POSTGRES_USER", cast=str, default="langchain")
POSTGRES_PASSWORD = env("POSTGRES_PASSWORD", cast=str, default="langchain")
POSTGRES_DB = env("POSTGRES_DB", cast=str, default="langchain_test")

# Read allowed origins from environment variable
ALLOW_ORIGINS_JSON = env("ALLOW_ORIGINS", cast=str, default="")

if ALLOW_ORIGINS_JSON:
    ALLOWED_ORIGINS = json.loads(ALLOW_ORIGINS_JSON.strip())
    print(f"ALLOW_ORIGINS environment variable set to: {ALLOW_ORIGINS_JSON}")
else:
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8502"]
    print("ALLOW_ORIGINS environment variable not set.")
