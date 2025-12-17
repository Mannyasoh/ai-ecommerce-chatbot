from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from ..config import settings
from ..database.models import Product


class EmbeddingManager:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.embedding_model

    def generate_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model, input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model, input=list(texts), encoding_format="float"
        )
        return [data.embedding for data in response.data]

    def prepare_product_text(self, product: Product) -> str:
        text_parts = [
            product.name,
            product.description,
            f"Price: ${product.price}",
            f"Category: {product.category}",
            f"Stock: {product.stock_status}",
        ]

        if product.specifications:
            spec_text = [
                f"{key}: {value}" for key, value in product.specifications.items()
            ]
            text_parts.append("Specifications: " + ", ".join(spec_text))

        return " | ".join(text_parts)

    def cosine_similarity(
        self, embedding1: Sequence[float], embedding2: Sequence[float]
    ) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


embedding_manager = EmbeddingManager()
