from collections.abc import Sequence

from ..database.models import ChatMessage
from .order_agent import order_agent
from .rag_agent import rag_agent


class ConversationOrchestrator:
    def __init__(self) -> None:
        self.rag_agent = rag_agent
        self.order_agent = order_agent
        self.current_agent = None
        self.conversation_state = "product_inquiry"

    def determine_agent(self, message: str, chat_history: Sequence[ChatMessage]) -> str:
        # Check for order intent first
        if self.order_agent.detect_order_intent(message):
            if self._has_product_context(chat_history):
                return "order_agent"

        if self.rag_agent.should_handoff_to_order_agent(message, chat_history):
            return "order_agent"

        order_id_patterns = ["order", "ord-", "#ord", "status"]
        message_lower = message.lower()
        if any(pattern in message_lower for pattern in order_id_patterns):
            import re

            if re.search(r"ord-\d{8}-[a-z0-9]{8}|#ord[\d-]+", message_lower):
                return "order_agent"

        return "rag_agent"

    def _has_product_context(self, chat_history: Sequence[ChatMessage]) -> bool:
        if not chat_history:
            return False

        # Look for product context in last 5 messages
        recent_messages = chat_history[-5:]
        product_indicators = [
            "price",
            "$",
            "product",
            "available",
            "stock",
            "category",
            "spec",
        ]

        for message in recent_messages:
            content_lower = message.content.lower()
            if any(indicator in content_lower for indicator in product_indicators):
                return True

        return False

    def process_message(
        self, message: str, chat_history: Sequence[ChatMessage]
    ) -> dict[str, str | bool | dict]:
        try:
            # Determine which agent to use
            selected_agent = self.determine_agent(message, chat_history)

            # Update conversation state
            if selected_agent == "order_agent":
                self.conversation_state = "order_processing"
            else:
                self.conversation_state = "product_inquiry"

            # Process message with selected agent
            if selected_agent == "order_agent":
                result = self.order_agent.process_message(message, chat_history)

                # Check if order was successfully created to provide handoff message
                if result.get("success") and result.get("function_calls"):
                    for func_call in result["function_calls"]:
                        is_create_order = func_call["name"] == "create_order"
                        is_success = func_call["result"].get("success")
                        if is_create_order and is_success:
                            result["handoff_occurred"] = True
                            result["handoff_from"] = "rag_agent"
                            result["handoff_to"] = "order_agent"

            else:
                result = self.rag_agent.process_message(message, chat_history)

                should_handoff = self.rag_agent.should_handoff_to_order_agent(
                    message, chat_history
                )
                if should_handoff:
                    result["handoff_suggested"] = True
                    result["suggested_handoff_to"] = "order_agent"

            result["orchestrator"] = {
                "selected_agent": selected_agent,
                "conversation_state": self.conversation_state,
                "has_product_context": self._has_product_context(chat_history),
            }

            self.current_agent = selected_agent
            return result

        except Exception as e:
            return {
                "success": False,
                "agent": "orchestrator",
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e),
                "orchestrator": {
                    "selected_agent": None,
                    "conversation_state": self.conversation_state,
                    "error": "orchestrator_error",
                },
            }

    def get_agent_capabilities(self) -> dict[str, list[str]]:
        return {
            "rag_agent": [
                "Search products by name, category, or features",
                "Get product details and specifications",
                "Check product availability and pricing",
                "Compare products and suggest alternatives",
                "Answer product-related questions",
            ],
            "order_agent": [
                "Create new orders from conversation context",
                "Validate order details before creation",
                "Check order status and history",
                "Cancel existing orders",
                "Extract order information from chat history",
            ],
        }

    def get_conversation_summary(
        self, chat_history: Sequence[ChatMessage]
    ) -> dict[str, str | int | list]:
        summary = {
            "total_messages": len(chat_history),
            "current_agent": self.current_agent,
            "conversation_state": self.conversation_state,
            "products_mentioned": [],
            "orders_created": [],
            "agent_switches": 0,
        }

        # Analyze conversation for patterns
        previous_agent = None
        for message in chat_history:
            if hasattr(message, "metadata") and message.metadata:
                agent = message.metadata.get("agent")
                if agent and agent != previous_agent:
                    summary["agent_switches"] += 1
                    previous_agent = agent

        import re

        for message in chat_history:
            if message.role == "assistant":
                product_matches = re.findall(r"Product:\s*([^,\n]+)", message.content)
                summary["products_mentioned"].extend(product_matches)

                order_matches = re.findall(r"Order ID:\s*([#\w\-]+)", message.content)
                summary["orders_created"].extend(order_matches)

        summary["products_mentioned"] = list(set(summary["products_mentioned"]))
        summary["orders_created"] = list(set(summary["orders_created"]))

        return summary

    def reset_conversation_state(self) -> None:
        self.current_agent = None
        self.conversation_state = "product_inquiry"


orchestrator = ConversationOrchestrator()
