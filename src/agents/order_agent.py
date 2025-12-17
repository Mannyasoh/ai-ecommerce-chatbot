"""Order Agent for processing customer orders"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from ..config import settings
from ..database.models import ChatMessage
from ..functions.order_functions import (
    ORDER_FUNCTION_SCHEMAS,
    cancel_order,
    create_order,
    get_order_status,
    validate_order_details,
)


class OrderAgent:
    """Order Agent for handling order creation and management"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Order Agent"""
        self.client = OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.openai_model
        self.function_schemas = ORDER_FUNCTION_SCHEMAS
        
        # Function mapping
        self.function_map = {
            "create_order": create_order,
            "get_order_status": get_order_status,
            "validate_order_details": validate_order_details,
            "cancel_order": cancel_order,
        }

    def detect_order_intent(self, message: str) -> bool:
        """
        Detect if message contains order intent.
        
        Args:
            message: User message
            
        Returns:
            True if message contains order intent, False otherwise
        """
        order_keywords = [
            "buy", "purchase", "order", "confirm", "take it", "add to cart",
            "checkout", "place order", "i want", "get me", "i'll take",
            "yes please", "proceed", "continue"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in order_keywords)

    def extract_order_context(self, chat_history: List[ChatMessage]) -> Dict[str, Any]:
        """
        Extract order details from chat history.
        
        Args:
            chat_history: Conversation history
            
        Returns:
            Dictionary with extracted order context
        """
        context = {
            "product_name": None,
            "quantity": 1,
            "price": None,
            "product_mentioned": [],
            "quantities_mentioned": [],
            "customer_info": {}
        }
        
        # Analyze recent messages for product and quantity information
        for message in chat_history[-10:]:  # Look at last 10 messages
            content = message.content
            
            # Extract product names (look for products mentioned in responses)
            product_patterns = [
                r"(\w+(?:\s+\w+)*)\s*-\s*\$[\d,]+(?:\.\d{2})?",  # "iPhone 15 - $999"
                r"Product:\s*([^,\n]+)",  # "Product: MacBook Pro"
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*(?:\s+\d+)?)\s*\$[\d,]+",  # "MacBook Pro $1999"
            ]
            
            for pattern in product_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    context["product_mentioned"].append(match.strip())
            
            # Extract prices
            price_matches = re.findall(r"\$[\d,]+(?:\.\d{2})?", content)
            if price_matches and not context["price"]:
                try:
                    price_str = price_matches[0].replace("$", "").replace(",", "")
                    context["price"] = float(price_str)
                except (ValueError, IndexError):
                    pass
            
            # Extract quantities
            quantity_patterns = [
                r"(\d+)\s*(?:x|×|pieces?|units?|items?)",
                r"quantity[:\s]*(\d+)",
                r"(\d+)\s*of\s+",
                r"take\s+(\d+)",
                r"want\s+(\d+)",
                r"buy\s+(\d+)"
            ]
            
            for pattern in quantity_patterns:
                matches = re.findall(pattern, content.lower())
                for match in matches:
                    try:
                        qty = int(match)
                        if 1 <= qty <= 100:  # Reasonable quantity range
                            context["quantities_mentioned"].append(qty)
                    except ValueError:
                        pass
            
            # Extract customer information
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content)
            if email_match:
                context["customer_info"]["email"] = email_match.group()
            
            phone_match = re.search(r"\b\d{3}-?\d{3}-?\d{4}\b", content)
            if phone_match:
                context["customer_info"]["phone"] = phone_match.group()
        
        # Select most likely product and quantity
        if context["product_mentioned"]:
            # Use most recently mentioned product
            context["product_name"] = context["product_mentioned"][-1]
        
        if context["quantities_mentioned"]:
            # Use most recently mentioned quantity
            context["quantity"] = context["quantities_mentioned"][-1]
        
        return context

    def process_message(
        self, 
        message: str, 
        chat_history: List[ChatMessage]
    ) -> Dict[str, Any]:
        """
        Process order-related message and execute appropriate actions.
        
        Args:
            message: User message
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with agent response and metadata
        """
        try:
            # Extract order context from chat history
            order_context = self.extract_order_context(chat_history)
            
            # Build messages for OpenAI API
            messages = self._build_messages(message, chat_history, order_context)
            
            # Call OpenAI with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._format_tools(),
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1000
            )
            
            response_message = response.choices[0].message
            
            # Handle function calls
            if response_message.tool_calls:
                return self._handle_function_calls(response_message, messages, order_context)
            else:
                # Regular response without function calls
                return {
                    "success": True,
                    "agent": "order_agent",
                    "response": response_message.content,
                    "function_calls": [],
                    "order_context": order_context,
                    "metadata": {"model": self.model, "type": "direct_response"}
                }
                
        except Exception as e:
            return {
                "success": False,
                "agent": "order_agent",
                "response": f"I apologize, but I encountered an error while processing your order: {str(e)}",
                "error": str(e),
                "function_calls": []
            }

    def _build_messages(
        self, 
        message: str, 
        chat_history: List[ChatMessage], 
        order_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build message list for OpenAI API"""
        context_info = ""
        if order_context["product_name"]:
            context_info += f"Product mentioned: {order_context['product_name']}\n"
        if order_context["quantity"] > 1:
            context_info += f"Quantity mentioned: {order_context['quantity']}\n"
        if order_context["price"]:
            context_info += f"Price mentioned: ${order_context['price']}\n"
        
        system_message = f"""You are an expert order processing specialist for an e-commerce platform. Your role is to:

1. Create orders when customers confirm their purchase intent
2. Extract order details from conversation context
3. Validate orders before creation
4. Provide order status information
5. Handle order cancellations

IMPORTANT CONTEXT FROM CONVERSATION:
{context_info}

Guidelines:
- Always confirm order details before creating an order
- Use the conversation context to extract product names and quantities
- If information is missing or ambiguous, ask for clarification
- Provide clear order confirmations with order IDs
- Be helpful and professional

When creating orders:
- Use validate_order_details first if you're unsure about product availability
- Extract product name from context when user says "I'll take it", "confirm", etc.
- Default quantity is 1 unless specified otherwise
- Always confirm the order details in your response"""
        
        messages = [{"role": "system", "content": system_message}]
        
        # Add relevant chat history (last 8 messages to include context)
        for chat_msg in chat_history[-8:]:
            messages.append({
                "role": chat_msg.role,
                "content": chat_msg.content
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages

    def _format_tools(self) -> List[Dict[str, Any]]:
        """Format function schemas for OpenAI tools parameter"""
        return [{"type": "function", "function": schema} for schema in self.function_schemas]

    def _handle_function_calls(
        self, 
        response_message: Any, 
        messages: List[Dict[str, Any]],
        order_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle function calls and generate final response"""
        try:
            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in response_message.tool_calls
                ]
            })
            
            function_results = []
            
            # Execute each function call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Auto-fill missing arguments from context
                if function_name == "create_order" and "product_name" not in function_args:
                    if order_context["product_name"]:
                        function_args["product_name"] = order_context["product_name"]
                
                if function_name == "create_order" and "quantity" not in function_args:
                    function_args["quantity"] = order_context["quantity"]
                
                # Execute function
                if function_name in self.function_map:
                    result = self.function_map[function_name](**function_args)
                    function_results.append({
                        "name": function_name,
                        "arguments": function_args,
                        "result": result
                    })
                    
                    # Add function result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                else:
                    # Function not found
                    error_result = {
                        "success": False,
                        "error": f"Unknown function: {function_name}"
                    }
                    function_results.append({
                        "name": function_name,
                        "arguments": function_args,
                        "result": error_result
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(error_result)
                    })
            
            # Get final response from OpenAI
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return {
                "success": True,
                "agent": "order_agent",
                "response": final_response.choices[0].message.content,
                "function_calls": function_results,
                "order_context": order_context,
                "metadata": {
                    "model": self.model,
                    "type": "function_calling",
                    "functions_used": [fc["name"] for fc in function_results]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent": "order_agent",
                "response": f"I encountered an error while processing your order: {str(e)}",
                "error": str(e),
                "function_calls": function_results if 'function_results' in locals() else [],
                "order_context": order_context
            }


# Global order agent instance
order_agent = OrderAgent()