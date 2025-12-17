"""Embedding generation and vector search functionality"""

import os
from typing import List, Optional

import numpy as np
from openai import OpenAI

from ..config import settings
from ..database.models import Product, VectorSearchResult


class EmbeddingManager:
    """Manages embedding generation using OpenAI"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize embedding manager"""
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.embedding_model

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise Exception(f"Failed to generate batch embeddings: {str(e)}")

    def prepare_product_text(self, product: Product) -> str:
        """Prepare product text for embedding"""
        text_parts = [
            product.name,
            product.description,
            f"Price: ${product.price}",
            f"Category: {product.category}",
            f"Stock: {product.stock_status}"
        ]

        # Add specifications if available
        if product.specifications:
            spec_text = []
            for key, value in product.specifications.items():
                spec_text.append(f"{key}: {value}")
            text_parts.append("Specifications: " + ", ".join(spec_text))

        return " | ".join(text_parts)

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            raise Exception(f"Failed to calculate cosine similarity: {str(e)}")


# Global embedding manager instance
embedding_manager = EmbeddingManager()