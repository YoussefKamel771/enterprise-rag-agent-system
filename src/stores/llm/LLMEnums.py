from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    SENTENCE_TRANSFORMER = "SENTENCE_TRANSFORMER"
    
    
class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class COHEREEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"  

    DOCUMENT = "search_document"
    QUERY = "search_query"

class DocumentTypeEnums(str, Enum):
    QUERY = "query"
    DOCUMENT = "document"
