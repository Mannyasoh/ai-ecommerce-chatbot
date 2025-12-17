import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")
sys.path.insert(0, str(project_root))

from src.config import validate_environment
from src.database.database import db_manager
from src.database.models import Product
from src.logging_config import configure_logging
from src.vector_store.chroma_store import vector_store


def load_products_from_json(file_path: str) -> list[Product]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            products_data = json.load(f)

        products: list[Product] = []
        for product_data in products_data:
            try:
                product = Product(**product_data)
                products.append(product)
            except Exception as e:
                name = product_data.get("name", "Unknown")
                print(f"Error creating product {name}: {str(e)}")
                continue

        return products

    except FileNotFoundError:
        print(f"Product data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {str(e)}")
        return []
    except Exception as e:
        print(f"Error loading products: {str(e)}")
        return []


def load_data():
    try:
        print("Validating environment...")
        validate_environment()

        data_file = project_root / "data" / "products.json"
        print(f"Loading products from: {data_file}")

        products = load_products_from_json(str(data_file))

        if not products:
            print("No products loaded from file")
            return False

        print(f"Loaded {len(products)} products from file")

        print("Clearing existing data...")
        vector_store.clear_collection()

        print("Adding products to database...")
        db_success_count = 0
        db_error_count = 0

        for product in products:
            try:
                success = db_manager.add_product(product)
                if success:
                    db_success_count += 1
                else:
                    db_error_count += 1
                    print(f"Failed to add {product.name} to database")
            except Exception as e:
                db_error_count += 1
                print(f"Error adding {product.name} to database: {str(e)}")

        print(f"Database: {db_success_count} products added, {db_error_count} errors")

        print("Adding products to vector store...")
        try:
            vector_success = vector_store.add_products_batch(products)
            if vector_success:
                print(f"Vector store: {len(products)} products added successfully")
            else:
                print("Failed to add products to vector store")
                return False
        except Exception as e:
            print(f"Error adding products to vector store: {str(e)}")
            return False

        print("Verifying data...")
        collection_info = vector_store.get_collection_info()
        all_products = db_manager.get_all_products()

        print("Verification complete:")
        print(f"   Database: {len(all_products)} products")
        print(f"   Vector store: {collection_info['document_count']} documents")

        doc_count = collection_info["document_count"]
        if len(all_products) == doc_count == len(products):
            print("All products loaded successfully!")
            return True
        else:
            msg = (
                "Mismatch in product counts - "
                "some products may not have loaded correctly"
            )
            print(msg)
            return False

    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        return False


def test_search():
    print("\nTesting search functionality...")

    test_queries = [
        "iPhone",
        "laptop under $1500",
        "wireless headphones",
        "gaming",
        "MacBook",
    ]

    for query in test_queries:
        try:
            print(f"\nTesting query: '{query}'")
            results = vector_store.search_products(query, n_results=3)

            if results:
                print(f"   Found {len(results)} results:")
                for i, result in enumerate(results[:2], 1):
                    name = result.product.name
                    price = result.product.price
                    score = result.score
                    print(f"   {i}. {name} - ${price} (score: {score:.3f})")
            else:
                print("   No results found")

        except Exception as e:
            print(f"   Error testing query '{query}': {str(e)}")

    print("\nSearch testing complete!")


def main():
    print("AI E-Commerce Chatbot - Data Loading Script")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable not set")
        print("   Please set your OpenAI API key before running this script")
        print("   Example: export OPENAI_API_KEY=your_api_key_here")
        return

    try:
        success = load_data()

        if success:
            test_search()
            print("\nData loading completed successfully!")
            print("   You can now run the chatbot with: python main.py")
        else:
            print("\nData loading failed - please check the errors above")

    except KeyboardInterrupt:
        print("\nData loading cancelled by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")


if __name__ == "__main__":
    # Configure logging for script execution
    configure_logging()
    logger.info("Starting data loading script")
    main()
    logger.info("Data loading script completed")
