"""Integration tests for the AI E-Commerce Chatbot system"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from src.agents.orchestrator import ConversationOrchestrator
from src.agents.order_agent import OrderAgent
from src.agents.rag_agent import RAGAgent
from src.database.database import DatabaseManager
from src.database.models import OrderModel, OrderStatus, Product


class TestProductModels:
    """Test Pydantic models"""

    def test_product_model_validation(self):
        """Test product model validation"""
        product_data = {
            "product_id": "TEST-001",
            "name": "Test Product",
            "description": "A test product for validation",
            "price": 99.99,
            "stock_status": "in_stock",
            "category": "test",
        }

        product = Product(**product_data)
        assert product.product_id == "TEST-001"
        assert product.price == 99.99
        assert product.stock_status == "in_stock"

    def test_product_model_invalid_price(self):
        """Test product model with invalid price"""
        product_data = {
            "product_id": "TEST-002",
            "name": "Test Product",
            "description": "A test product",
            "price": -10.00,  # Invalid negative price
            "stock_status": "in_stock",
            "category": "test",
        }

        with pytest.raises(ValueError):
            Product(**product_data)

    def test_order_model_validation(self):
        """Test order model validation"""
        order_data = {
            "order_id": "ORD-TEST-001",
            "product_name": "Test Product",
            "quantity": 2,
            "unit_price": 50.00,
            "total_price": 100.00,
        }

        order = OrderModel(**order_data)
        assert order.order_id == "ORD-TEST-001"
        assert order.quantity == 2
        assert order.total_price == 100.00

    def test_order_model_invalid_total(self):
        """Test order model with invalid total price"""
        order_data = {
            "order_id": "ORD-TEST-002",
            "product_name": "Test Product",
            "quantity": 2,
            "unit_price": 50.00,
            "total_price": 150.00,  # Should be 100.00
        }

        with pytest.raises(ValueError, match="Total price must equal"):
            OrderModel(**order_data)


class TestDatabaseOperations:
    """Test database operations"""

    def setup_method(self):
        """Setup test database"""
        # Use in-memory SQLite for testing
        self.db_manager = DatabaseManager("sqlite:///:memory:")

    def test_add_and_get_product(self):
        """Test adding and retrieving products"""
        product = Product(
            product_id="TEST-001",
            name="Test Laptop",
            description="A high-performance laptop for testing",
            price=999.99,
            stock_status="in_stock",
            category="laptops",
        )

        # Add product
        success = self.db_manager.add_product(product)
        assert success is True

        # Retrieve product
        retrieved = self.db_manager.get_product("TEST-001")
        assert retrieved is not None
        assert retrieved.name == "Test Laptop"
        assert retrieved.price == 999.99

    def test_create_and_get_order(self):
        """Test creating and retrieving orders"""
        order = OrderModel(
            order_id="",  # Will be generated
            product_name="Test Laptop",
            quantity=1,
            unit_price=999.99,
            total_price=999.99,
        )

        # Create order
        order_id = self.db_manager.create_order(order)
        assert order_id is not None
        assert len(order_id) > 0

        # Retrieve order
        retrieved = self.db_manager.get_order(order_id)
        assert retrieved is not None
        assert retrieved.product_name == "Test Laptop"
        assert retrieved.total_price == 999.99

    def test_update_order_status(self):
        """Test updating order status"""
        order = OrderModel(
            order_id="",
            product_name="Test Product",
            quantity=1,
            unit_price=50.00,
            total_price=50.00,
        )

        order_id = self.db_manager.create_order(order)

        # Update status
        success = self.db_manager.update_order_status(order_id, OrderStatus.CONFIRMED)
        assert success is True

        # Verify update
        retrieved = self.db_manager.get_order(order_id)
        assert retrieved.status == OrderStatus.CONFIRMED


class TestFunctionCalling:
    """Test function calling implementations"""

    def setup_method(self):
        """Setup test environment"""
        # Use in-memory database for testing
        self.db_manager = DatabaseManager("sqlite:///:memory:")

        # Add sample product for testing
        self.test_product = Product(
            product_id="TEST-FUNC-001",
            name="Test iPhone",
            description="A test smartphone for function testing",
            price=799.99,
            stock_status="in_stock",
            category="smartphones",
        )
        self.db_manager.add_product(self.test_product)

    @patch("src.functions.product_functions.vector_store")
    @patch("src.functions.product_functions.db_manager")
    def test_search_products_function(self, mock_db, mock_vector_store):
        """Test search_products function"""
        from src.database.models import VectorSearchResult
        from src.functions.product_functions import search_products

        # Create mock vector search result
        mock_search_result = VectorSearchResult(
            product=self.test_product, score=0.95, metadata={}
        )

        # Mock vector store response
        mock_vector_store.search_products.return_value = [mock_search_result]

        result = search_products("iPhone", max_results=5)

        assert result["success"] is True
        assert result["products_found"] == 1
        assert len(result["products"]) == 1
        assert result["products"][0]["name"] == "Test iPhone"
        assert result["products"][0]["price"] == 799.99
        assert result["products"][0]["similarity_score"] == 0.95

    @patch("src.functions.order_functions.db_manager")
    def test_create_order_function(self, mock_db):
        """Test create_order function"""
        from src.functions.order_functions import create_order

        # Mock database responses
        mock_db.search_products.return_value = [self.test_product]
        mock_db.create_order.return_value = "ORD-TEST-12345"

        result = create_order("Test iPhone", quantity=1)

        assert result["success"] is True
        assert result["order_id"] == "ORD-TEST-12345"
        assert result["product_name"] == "Test iPhone"
        assert result["total_price"] == 799.99


class TestAgents:
    """Test agent implementations"""

    def setup_method(self):
        """Setup test environment"""
        # Mock OpenAI client to avoid API calls during testing
        self.mock_client = Mock()

    def test_rag_agent_intent_detection(self):
        """Test RAG agent intent detection"""
        agent = RAGAgent()

        # Test product queries
        assert agent.detect_product_intent("What laptops do you have?") is True
        assert agent.detect_product_intent("Show me iPhone prices") is True
        assert (
            agent.detect_product_intent("Tell me about MacBook specifications") is True
        )

        # Test non-product queries
        assert agent.detect_product_intent("Hello") is False
        assert agent.detect_product_intent("Thank you") is False

    def test_order_agent_intent_detection(self):
        """Test order agent intent detection"""
        agent = OrderAgent()

        # Test order queries
        assert agent.detect_order_intent("I'll take it") is True
        assert agent.detect_order_intent("I want to buy this") is True
        assert agent.detect_order_intent("Place the order") is True

        # Test non-order queries
        assert agent.detect_order_intent("What is the price?") is False
        assert agent.detect_order_intent("Tell me more") is False

    def test_orchestrator_agent_selection(self):
        """Test orchestrator agent selection logic"""
        from src.database.models import ChatMessage

        orchestrator = ConversationOrchestrator()

        # Test product query routing
        chat_history = [
            ChatMessage(role="user", content="What laptops do you have?"),
            ChatMessage(role="assistant", content="We have MacBook Pro for $1999"),
        ]

        # Product query should go to RAG agent
        agent = orchestrator.determine_agent("Tell me more about MacBook", chat_history)
        assert agent == "rag_agent"

        # Order query with product context should go to order agent
        agent = orchestrator.determine_agent("I'll take it", chat_history)
        assert agent == "order_agent"


class TestIntegration:
    """Integration tests with real data/components"""

    def test_search_products_integration(self):
        """Integration test for search_products with real vector store"""
        from src.functions.product_functions import search_products

        result = search_products("laptop", max_results=3)

        assert result["success"] is True
        assert result["products_found"] > 0
        assert len(result["products"]) > 0

        # Check that results have proper structure and reasonable content
        for product in result["products"]:
            assert "name" in product
            assert "description" in product
            assert "price" in product
            assert "category" in product
            assert "similarity_score" in product
            assert 0 <= product["similarity_score"] <= 1

        # For laptop search, expect laptops category (semantic search should work)
        categories = [p["category"] for p in result["products"]]
        assert (
            "laptops" in categories
        ), f"Expected laptops in categories, got: {categories}"

    def test_create_order_integration(self):
        """Integration test for create_order with real database"""
        from src.functions.order_functions import create_order

        # Use a known product from the sample data
        result = create_order("DJI Mini 3", quantity=2)

        assert result["success"] is True
        assert result["order_id"] is not None
        assert result["product_name"] == "DJI Mini 3"
        assert result["quantity"] == 2
        assert result["total_price"] > 0
        assert "message" in result

    def test_product_availability_integration(self):
        """Integration test for product availability check"""
        from src.functions.product_functions import check_product_availability

        result = check_product_availability("iPhone")

        assert result["success"] is True
        assert result["available"] in [True, False]  # Could be either
        assert result["product_name"] is not None
        assert "message" in result

    def test_get_products_by_category_integration(self):
        """Integration test for category-based product search"""
        from src.functions.product_functions import get_products_by_category

        result = get_products_by_category("smartphones", limit=5)

        assert result["success"] is True
        assert result["category"] == "smartphones"
        assert result["products_found"] > 0

        # All products should be smartphones
        for product in result["products"]:
            assert product["category"] == "smartphones"

    def test_orchestrator_integration(self):
        """Integration test for orchestrator with real agents"""
        from src.database.models import ChatMessage

        orchestrator = ConversationOrchestrator()

        # Test product inquiry routing
        chat_history = []
        result = orchestrator.determine_agent("What laptops do you have?", chat_history)
        assert result == "rag_agent"

        # Test order intent with product context
        chat_history = [
            ChatMessage(
                role="assistant", content="We have MacBook Pro for $1999 - Available"
            ),
        ]
        result = orchestrator.determine_agent("I'll take it", chat_history)
        assert result == "order_agent"

        # Test conversation summary
        summary = orchestrator.get_conversation_summary(chat_history)
        assert "total_messages" in summary
        assert "current_agent" in summary


class TestEndToEnd:
    """End-to-end integration tests"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_full_conversation_flow(self):
        """Test complete conversation flow (requires API key)"""
        from src.database.models import ChatMessage

        orchestrator = ConversationOrchestrator()
        chat_history = []

        # Simulate product inquiry
        user_message = "What smartphones do you have under $800?"
        result = orchestrator.process_message(user_message, chat_history)

        # Should use RAG agent
        assert result.get("success") is True
        assert result.get("agent") == "rag_agent"

        # Add to chat history
        chat_history.append(ChatMessage(role="user", content=user_message))
        chat_history.append(ChatMessage(role="assistant", content=result["response"]))

        # Simulate order intent
        order_message = "I want to buy the iPhone 15"
        result = orchestrator.process_message(order_message, chat_history)

        # Should use order agent
        assert result.get("success") is True
        assert result.get("agent") == "order_agent"

    def test_full_system_robustness(self):
        """Test system handles edge cases gracefully"""
        from src.functions.order_functions import create_order
        from src.functions.product_functions import search_products

        # Test empty query - should return proper error
        result = search_products("", max_results=1)
        assert result["success"] is False  # Should properly reject empty query
        assert "error" in result
        assert "validation error" in result["error"].lower()

        # Test non-existent product order
        result = create_order("NonExistentProduct12345", quantity=1)
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

        # Test search with very specific non-existent product
        result = search_products("SuperSpecificNonExistentGadget2024XYZ", max_results=5)
        assert (
            result["success"] is True
        )  # Search succeeds but finds no relevant results
        # May return 0 results or unrelated results with low similarity scores


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
