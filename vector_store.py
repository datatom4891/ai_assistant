import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv(dotenv_path=".env", override=True)

chroma_dir = os.path.join(os.getcwd(),'ltm_chroma_db')
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

longterm_memory_vs = Chroma(collection_name="ltm",
                      embedding_function = embeddings,
                      persist_directory = chroma_dir)