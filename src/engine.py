from llama_index.core import VectorStoreIndex, Settings 
from llama_index.llms.ollama import Ollama 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config.settings import Settings as ConfigSettings

class RAG:    
    def __init__(self, model_name=ConfigSettings.model_name, embedding_model_name=ConfigSettings.embedding_model_name):

        """Initialize the RAG engine with the specified model and embedding model."""

        self.llm = Ollama(model=model_name, 
                          temperature=ConfigSettings.temperature, 
                          request_timeout=ConfigSettings.request_timeout
                          )
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None
        self.chat_engine = None
    
    def create_index(self, documents):
        """Create a vector store index from the provided documents."""
        self.index = VectorStoreIndex.from_documents(documents)

        self.chat_engine = self.index.as_chat_engine(
            chat_model="condense_plus_context",
            system_prompt=(
                "Você é um assistente técnico especializado em análise de documentos. "
                "Sua tarefa é extrair requisitos, riscos e recomendações de forma precisa. "
                "Responda sempre em Português (Brasil) de forma profissional e direta."
                )
        )

    def query(self, query):
        """Query the RAG engine with a user query and return the response."""
        if not self.chat_engine:
            raise ValueError("O índice de documentos não foi criado. Por favor, crie o índice antes de fazer uma consulta.")
        
        response = self.chat_engine.query(query)

        return getattr(response, "response", str(response))

