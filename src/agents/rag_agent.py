import json
from collections.abc import Sequence

from openai import OpenAI

from ..config import settings
from ..database.models import ChatMessage
from ..functions.product_functions import (
    PRODUCT_FUNCTION_SCHEMAS,
    check_product_availability,
    get_product_details,
    get_products_by_category,
    search_products,
)


class RAGAgent:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.openai_model
        self.function_schemas = PRODUCT_FUNCTION_SCHEMAS

        self.function_map = {
            "search_products": search_products,
            "get_product_details": get_product_details,
            "check_product_availability": check_product_availability,
            "get_products_by_category": get_products_by_category,
        }

    def detect_product_intent(self, message: str) -> bool:
        product_keywords = [
            "price",
            "cost",
            "how much",
            "show me",
            "find",
            "search",
            "what",
            "which",
            "tell me about",
            "do you have",
            "available",
            "spec",
            "specification",
            "feature",
            "detail",
            "info",
            "information",
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in product_keywords)

    def process_message(
        self, message: str, chat_history: Sequence[ChatMessage]
    ) -> dict[str, str | bool | list]:
        try:
            # Build messages for OpenAI API
            messages = self._build_messages(message, chat_history)

            # Call OpenAI with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._format_tools(),
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1000,
            )

            response_message = response.choices[0].message

            # Handle function calls
            if response_message.tool_calls:
                return self._handle_function_calls(response_message, messages)
            else:
                # Regular response without function calls
                return {
                    "success": True,
                    "agent": "rag_agent",
                    "response": response_message.content,
                    "function_calls": [],
                    "metadata": {"model": self.model, "type": "direct_response"},
                }

        except Exception as e:
            return {
                "success": False,
                "agent": "rag_agent",
                "response": f"I apologize, but I encountered an error while searching for product information: {str(e)}",
                "error": str(e),
                "function_calls": [],
            }

    def _build_messages(
        self, message: str, chat_history: Sequence[ChatMessage]
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful e-commerce product specialist. Your primary role is to:

1. Answer questions about products using the available search functions
2. Provide accurate product information including prices, specifications,
   and availability
3. Help customers find the right products for their needs
4. Be conversational and helpful

Always use the search functions when users ask about products.
When presenting product information:
- Include prices clearly
- Mention stock availability
- Highlight key specifications
- Suggest alternatives when appropriate
- Be natural and conversational in your responses

If you detect that a user wants to make a purchase (phrases like "I'll take it",
"buy", "place order", "confirm"), respond conversationally but let them know
that you'll transfer them to our order specialist.""",
            }
        ]

        # Add chat history (last 10 messages to avoid token limit)
        for chat_msg in chat_history[-10:]:
            messages.append({"role": chat_msg.role, "content": chat_msg.content})

        # Add current message
        messages.append({"role": "user", "content": message})

        return messages

    def _format_tools(self) -> list[dict[str, dict | str]]:
        return [
            {"type": "function", "function": schema} for schema in self.function_schemas
        ]

    def _handle_function_calls(
        self, response_message, messages: list[dict[str, str]]
    ) -> dict[str, str | bool | list]:
        try:
            # Add assistant message with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in response_message.tool_calls
                    ],
                }
            )

            function_results = []

            # Execute each function call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Execute function
                if function_name in self.function_map:
                    result = self.function_map[function_name](**function_args)
                    function_results.append(
                        {
                            "name": function_name,
                            "arguments": function_args,
                            "result": result,
                        }
                    )

                    # Add function result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    # Function not found
                    error_result = {
                        "success": False,
                        "error": f"Unknown function: {function_name}",
                    }
                    function_results.append(
                        {
                            "name": function_name,
                            "arguments": function_args,
                            "result": error_result,
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(error_result),
                        }
                    )

            # Get final response from OpenAI
            final_response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1, max_tokens=1000
            )

            return {
                "success": True,
                "agent": "rag_agent",
                "response": final_response.choices[0].message.content,
                "function_calls": function_results,
                "metadata": {
                    "model": self.model,
                    "type": "function_calling",
                    "functions_used": [fc["name"] for fc in function_results],
                },
            }

        except Exception as e:
            return {
                "success": False,
                "agent": "rag_agent",
                "response": f"I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "function_calls": (
                    function_results if "function_results" in locals() else []
                ),
            }

    def should_handoff_to_order_agent(
        self, message: str, chat_history: Sequence[ChatMessage]
    ) -> bool:
        order_intent_phrases = [
            "i'll take it",
            "i'll buy",
            "i want to buy",
            "purchase",
            "order",
            "add to cart",
            "checkout",
            "place order",
            "confirm",
            "yes please",
            "i want",
            "get me",
            "buy now",
            "i need",
        ]

        message_lower = message.lower()

        # Check for direct order intent
        for phrase in order_intent_phrases:
            if phrase in message_lower:
                # Make sure there's product context in recent messages
                recent_messages = chat_history[-5:]  # Check last 5 messages
                for chat_msg in recent_messages:
                    if any(
                        word in chat_msg.content.lower()
                        for word in ["price", "product", "$", "available"]
                    ):
                        return True

        return False


rag_agent = RAGAgent()
