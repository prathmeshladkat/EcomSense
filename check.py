from config.settings import settings
import langchain
import langgraph
import chromadb
import redis
import fastapi
import streamlit

print("✓ settings loaded:", settings.environment)
print("✓ langchain:", langchain.__version__)
print("✓ chromadb:", chromadb.__version__)
print("✓ all imports working")
