"""ChromaDB vector store implementation"""

import json
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from ..database.models import Product, VectorSearchResult
from .embeddings import embedding_manager


class ChromaVectorStore:
    """ChromaDB-based vector store for product search"""

    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB vector store"""
        self.persist_directory = persist_directory or settings.vector_db_path
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection_name = settings.vector_collection_name
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Product embeddings for e-commerce chatbot"}
            )

    def add_product(self, product: Product) -> bool:
        """Add product to vector store"""
        try:
            # Prepare text for embedding
            product_text = embedding_manager.prepare_product_text(product)
            
            # Generate embedding
            embedding = embedding_manager.generate_embedding(product_text)
            
            # Prepare metadata
            metadata = {
                "product_id": product.product_id,
                "name": product.name,
                "price": product.price,
                "category": product.category,
                "stock_status": product.stock_status,
                "description": product.description[:500],  # Truncate for metadata
            }
            
            # Add specifications to metadata if they exist
            if product.specifications:
                metadata["specifications"] = json.dumps(product.specifications)
            
            # Add to collection
            self.collection.add(
                documents=[product_text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[product.product_id]
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to add product to vector store: {str(e)}")

    def add_products_batch(self, products: List[Product]) -> bool:
        """Add multiple products to vector store"""
        try:
            if not products:
                return True
            
            # Prepare texts and metadata
            texts = []
            metadatas = []
            ids = []
            
            for product in products:
                product_text = embedding_manager.prepare_product_text(product)
                texts.append(product_text)
                
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
                
                metadatas.append(metadata)
                ids.append(product.product_id)
            
            # Generate embeddings in batch
            embeddings = embedding_manager.generate_embeddings_batch(texts)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to add products batch to vector store: {str(e)}")

    def search_products(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        price_filter: Optional[Dict[str, float]] = None
    ) -> List[VectorSearchResult]:
        """Search products using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = embedding_manager.generate_embedding(query)
            
            # Prepare where clause for filtering
            where_clause = {}
            if category_filter:
                where_clause["category"] = {"$eq": category_filter}
            
            # Add price filtering if specified
            if price_filter:
                if "min_price" in price_filter:
                    where_clause["price"] = {"$gte": price_filter["min_price"]}
                if "max_price" in price_filter:
                    if "price" not in where_clause:
                        where_clause["price"] = {}
                    where_clause["price"]["$lte"] = price_filter["max_price"]
            
            # Perform vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 20),  # Limit to reasonable number
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to VectorSearchResult objects
            search_results = []
            if results["metadatas"] and results["metadatas"][0]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    # Reconstruct Product from metadata
                    specifications = None
                    if "specifications" in metadata:
                        try:
                            specifications = json.loads(metadata["specifications"])
                        except (json.JSONDecodeError, TypeError):
                            specifications = None
                    
                    product = Product(
                        product_id=metadata["product_id"],
                        name=metadata["name"],
                        description=metadata["description"],
                        price=metadata["price"],
                        stock_status=metadata["stock_status"],
                        category=metadata["category"],
                        specifications=specifications
                    )
                    
                    # Calculate similarity score (ChromaDB returns distances, convert to similarity)
                    distance = results["distances"][0][i] if results["distances"] else 0
                    similarity_score = max(0, 1 - distance)  # Convert distance to similarity
                    
                    search_results.append(VectorSearchResult(
                        product=product,
                        score=similarity_score,
                        metadata={"distance": distance, "rank": i + 1}
                    ))
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Failed to search products: {str(e)}")

    def delete_product(self, product_id: str) -> bool:
        """Delete product from vector store"""
        try:
            self.collection.delete(ids=[product_id])
            return True
        except Exception as e:
            raise Exception(f"Failed to delete product: {str(e)}")

    def update_product(self, product: Product) -> bool:
        """Update product in vector store"""
        try:
            # Delete existing product
            self.delete_product(product.product_id)
            
            # Add updated product
            return self.add_product(product)
            
        except Exception as e:
            raise Exception(f"Failed to update product: {str(e)}")

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")

    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Product embeddings for e-commerce chatbot"}
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to clear collection: {str(e)}")


# Global vector store instance
vector_store = ChromaVectorStore()