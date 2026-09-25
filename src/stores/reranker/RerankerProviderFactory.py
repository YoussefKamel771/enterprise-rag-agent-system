from .RerankerEnums import RerankerEnums
from .providers import CohereRerankerProvider, SentenceTransformerRerankerProvider, FastEmbedRerankerProvider

class RerankerProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == RerankerEnums.COHERE.value:
            return CohereRerankerProvider(
                api_key=self.config.COHERE_API_KEY,)
            
        if provider == RerankerEnums.SENTENCE_TRANSFORMERS.value:
            return SentenceTransformerRerankerProvider()

        if provider == RerankerEnums.FASTEMBED.value:
            return FastEmbedRerankerProvider()

        return None