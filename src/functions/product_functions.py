"""Function definitions for product-related operations"""

import json
from typing import Any, Dict, List, Optional

from ..config import settings
from ..database.database import db_manager
from ..database.models import Product, ProductSearchRequest
from ..vector_store.chroma_store import vector_store


def search_products(
    query: str,
    category: Optional[str] = None,
    max_results: int = 5,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Search for products using vector similarity and optional filters.
    
    Args:
        query: Product search query (e.g., "laptop", "iPhone", "gaming headphones")
        category: Optional category filter (e.g., "laptops", "smartphones")
        max_results: Maximum number of results to return (1-20)
        price_min: Minimum price filter
        price_max: Maximum price filter
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        # Validate input
        search_request = ProductSearchRequest(
            query=query,
            category=category,
            max_results=max_results,
            price_min=price_min,
            price_max=price_max
        )
        
        # Prepare price filter for vector search
        price_filter = {}
        if search_request.price_min is not None:
            price_filter["min_price"] = search_request.price_min
        if search_request.price_max is not None:
            price_filter["max_price"] = search_request.price_max
        
        # Perform vector search
        search_results = vector_store.search_products(
            query=search_request.query,
            n_results=search_request.max_results,
            category_filter=search_request.category,
            price_filter=price_filter if price_filter else None
        )
        
        # Format results for function response
        products = []
        for result in search_results:
            product_dict = {
                "product_id": result.product.product_id,
                "name": result.product.name,
                "description": result.product.description,
                "price": result.product.price,
                "stock_status": result.product.stock_status,
                "category": result.product.category,
                "similarity_score": round(result.score, 3)
            }
            
            # Add specifications if available
            if result.product.specifications:
                product_dict["specifications"] = result.product.specifications
            
            products.append(product_dict)
        
        return {
            "success": True,
            "query": search_request.query,
            "products_found": len(products),
            "products": products,
            "search_metadata": {
                "category_filter": search_request.category,
                "price_filter": {
                    "min": search_request.price_min,
                    "max": search_request.price_max
                },
                "max_results": search_request.max_results
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "products_found": 0,
            "products": []
        }


def get_product_details(product_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific product.
    
    Args:
        product_id: Unique product identifier
    
    Returns:
        Dictionary with product details
    """
    try:
        # Get product from database
        product = db_manager.get_product(product_id)
        
        if not product:
            return {
                "success": False,
                "error": f"Product with ID '{product_id}' not found",
                "product": None
            }
        
        return {
            "success": True,
            "product": {
                "product_id": product.product_id,
                "name": product.name,
                "description": product.description,
                "price": product.price,
                "stock_status": product.stock_status,
                "category": product.category,
                "specifications": product.specifications,
                "created_at": product.created_at.isoformat() if product.created_at else None
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "product": None
        }


def get_products_by_category(category: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get products filtered by category.
    
    Args:
        category: Product category (e.g., "laptops", "smartphones")
        limit: Maximum number of products to return
    
    Returns:
        Dictionary with category products
    """
    try:
        # Search products by category in database
        products = db_manager.search_products(query="", category=category, limit=limit)
        
        products_list = []
        for product in products:
            products_list.append({
                "product_id": product.product_id,
                "name": product.name,
                "description": product.description[:200] + "..." if len(product.description) > 200 else product.description,
                "price": product.price,
                "stock_status": product.stock_status,
                "category": product.category
            })
        
        return {
            "success": True,
            "category": category,
            "products_found": len(products_list),
            "products": products_list
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "category": category,
            "products_found": 0,
            "products": []
        }


def check_product_availability(product_name: str) -> Dict[str, Any]:
    """
    Check if a product is available for purchase.
    
    Args:
        product_name: Name of the product to check
    
    Returns:
        Dictionary with availability information
    """
    try:
        # Search for product by name
        products = db_manager.search_products(query=product_name, limit=5)
        
        if not products:
            return {
                "success": False,
                "product_name": product_name,
                "available": False,
                "message": f"Product '{product_name}' not found in our catalog",
                "alternatives": []
            }
        
        # Find exact or closest match
        best_match = None
        alternatives = []
        
        for product in products:
            if product.name.lower() == product_name.lower():
                best_match = product
                break
            elif product_name.lower() in product.name.lower():
                if not best_match:
                    best_match = product
                else:
                    alternatives.append(product)
            else:
                alternatives.append(product)
        
        if best_match:
            available = best_match.stock_status == "in_stock"
            return {
                "success": True,
                "product_name": best_match.name,
                "product_id": best_match.product_id,
                "available": available,
                "stock_status": best_match.stock_status,
                "price": best_match.price,
                "message": f"Product found: {best_match.name} - {'Available' if available else 'Not available'} ({best_match.stock_status})",
                "alternatives": [
                    {
                        "name": alt.name,
                        "product_id": alt.product_id,
                        "price": alt.price,
                        "stock_status": alt.stock_status
                    } for alt in alternatives[:3]  # Limit alternatives
                ]
            }
        else:
            return {
                "success": False,
                "product_name": product_name,
                "available": False,
                "message": f"Exact product '{product_name}' not found",
                "alternatives": [
                    {
                        "name": alt.name,
                        "product_id": alt.product_id,
                        "price": alt.price,
                        "stock_status": alt.stock_status
                    } for alt in alternatives[:5]
                ]
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "product_name": product_name,
            "available": False,
            "alternatives": []
        }


# Function schemas for OpenAI Function Calling
PRODUCT_FUNCTION_SCHEMAS = [
    {
        "name": "search_products",
        "description": "Search for products using natural language queries with optional filters",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product search query (e.g., 'laptop', 'iPhone 15', 'gaming headphones under $200')"
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g., 'laptops', 'smartphones', 'headphones')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-20)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "price_min": {
                    "type": "number",
                    "description": "Minimum price filter in USD"
                },
                "price_max": {
                    "type": "number",
                    "description": "Maximum price filter in USD"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_product_details",
        "description": "Get detailed information about a specific product using its ID",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "Unique product identifier"
                }
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "check_product_availability",
        "description": "Check if a product is available for purchase by name",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to check availability"
                }
            },
            "required": ["product_name"]
        }
    },
    {
        "name": "get_products_by_category",
        "description": "Get products from a specific category",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Product category (e.g., 'laptops', 'smartphones', 'headphones')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of products to return",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 10
                }
            },
            "required": ["category"]
        }
    }
]