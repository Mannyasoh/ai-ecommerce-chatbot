import json
import os
from collections.abc import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from ..database.models import Product, VectorSearchResult
from .embeddings import embedding_manager


def _prepare_product_metadata(product: Product) -> dict[str, str | int | float]:
    metadata = {
        "product_id": product.product_id,
        "name": product.name,
        "price": product.price,
        "category": product.category,
        "stock_status": product.stock_status,
        "description": product.description[:500],
    }

    if product.specifications:
        metadata["specifications"] = json.dumps(product.specifications)

    return metadata


def _reconstruct_product_from_metadata(
    metadata: dict[str, str | int | float],
) -> Product:
    specifications = None
    if "specifications" in metadata:
        try:
            specifications = json.loads(str(metadata["specifications"]))
        except (json.JSONDecodeError, TypeError):
            specifications = None

    return Product(
        product_id=str(metadata["product_id"]),
        name=str(metadata["name"]),
        description=str(metadata["description"]),
        price=float(metadata["price"]),
        stock_status=str(metadata["stock_status"]),
        category=str(metadata["category"]),
        specifications=specifications,
    )


class ChromaVectorStore:
    def __init__(self, persist_directory: str | None = None) -> None:
        self.persist_directory = persist_directory or settings.vector_db_path
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection_name = settings.vector_collection_name
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Product embeddings for e-commerce chatbot"},
            )

    def add_product(self, product: Product) -> bool:
        product_text = embedding_manager.prepare_product_text(product)
        embedding = embedding_manager.generate_embedding(product_text)
        metadata = _prepare_product_metadata(product)

        self.collection.add(
            documents=[product_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[product.product_id],
        )
        return True

    def add_products_batch(self, products: Sequence[Product]) -> bool:
        if not products:
            return True

        texts = [
            embedding_manager.prepare_product_text(product) for product in products
        ]
        metadatas = [_prepare_product_metadata(product) for product in products]
        ids = [product.product_id for product in products]
        embeddings = embedding_manager.generate_embeddings_batch(texts)

        self.collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        return True

    def search_products(
        self,
        query: str,
        n_results: int = 5,
        category_filter: str | None = None,
        price_filter: dict[str, float] | None = None,
    ) -> list[VectorSearchResult]:
        query_embedding = embedding_manager.generate_embedding(query)

        where_clause: dict[str, dict[str, str | float]] = {}
        if category_filter:
            where_clause["category"] = {"$eq": category_filter}

        if price_filter:
            price_conditions: dict[str, float] = {}
            if "min_price" in price_filter:
                price_conditions["$gte"] = price_filter["min_price"]
            if "max_price" in price_filter:
                price_conditions["$lte"] = price_filter["max_price"]
            if price_conditions:
                where_clause["price"] = price_conditions

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, 20),
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, metadata in enumerate(results["metadatas"][0]):
                product = _reconstruct_product_from_metadata(metadata)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity_score = max(0, 1 - distance)

                search_results.append(
                    VectorSearchResult(
                        product=product,
                        score=similarity_score,
                        metadata={"distance": distance, "rank": i + 1},
                    )
                )

        return search_results

    def delete_product(self, product_id: str) -> bool:
        self.collection.delete(ids=[product_id])
        return True

    def update_product(self, product: Product) -> bool:
        self.delete_product(product.product_id)
        return self.add_product(product)

    def get_collection_info(self) -> dict[str, str | int]:
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }

    def clear_collection(self) -> bool:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Product embeddings for e-commerce chatbot"},
        )
        return True


vector_store = ChromaVectorStore()
