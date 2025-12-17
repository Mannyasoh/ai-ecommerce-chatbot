"""Main application entry point for AI E-Commerce Chatbot"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import orchestrator
from src.config import settings, validate_environment
from src.database.database import db_manager
from src.database.models import ChatMessage
from src.vector_store.chroma_store import vector_store


class ECommerceChatbot:
    """Main chatbot application class"""

    def __init__(self):
        """Initialize the chatbot"""
        try:
            # Validate environment
            validate_environment()
            
            # Initialize chat history
            self.chat_history: List[ChatMessage] = []
            
            # Initialize components
            self.orchestrator = orchestrator
            self.db_manager = db_manager
            self.vector_store = vector_store
            
            print("✅ E-Commerce AI Chatbot initialized successfully!")
            print(f"📊 Vector DB: {self.vector_store.get_collection_info()}")
            
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {str(e)}")
            raise

    def add_message_to_history(self, role: str, content: str, metadata: Dict = None) -> None:
        """Add message to chat history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        
        # Add metadata if provided
        if metadata:
            # Note: In a real implementation, you'd want to properly handle metadata
            # For now, we'll store it as part of the message context
            pass
            
        self.chat_history.append(message)
        
        # Keep chat history manageable
        if len(self.chat_history) > settings.max_chat_history:
            self.chat_history = self.chat_history[-settings.max_chat_history:]

    def process_user_message(self, message: str) -> Dict:
        """
        Process user message and return chatbot response.
        
        Args:
            message: User input message
            
        Returns:
            Dictionary with chatbot response and metadata
        """
        try:
            # Add user message to history
            self.add_message_to_history("user", message)
            
            # Process message through orchestrator
            result = self.orchestrator.process_message(message, self.chat_history)
            
            # Add assistant response to history
            if result.get("success"):
                self.add_message_to_history(
                    "assistant", 
                    result["response"],
                    metadata={
                        "agent": result.get("agent"),
                        "function_calls": result.get("function_calls", []),
                        "orchestrator": result.get("orchestrator", {})
                    }
                )
            
            return result
            
        except Exception as e:
            error_response = {
                "success": False,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e),
                "agent": "system"
            }
            
            self.add_message_to_history("assistant", error_response["response"])
            return error_response

    def get_conversation_summary(self) -> Dict:
        """Get conversation summary"""
        return self.orchestrator.get_conversation_summary(self.chat_history)

    def reset_conversation(self) -> None:
        """Reset conversation state"""
        self.chat_history = []
        self.orchestrator.reset_conversation_state()
        print("🔄 Conversation reset successfully!")

    def run_interactive_chat(self) -> None:
        """Run interactive chat session"""
        print("\n" + "="*60)
        print("🤖 Welcome to the AI E-Commerce Chatbot!")
        print("="*60)
        print("Ask me about products, prices, or place orders.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'reset' to start a new conversation.")
        print("Type 'summary' to see conversation summary.")
        print("Type 'help' for more commands.")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💬 You: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thank you for using our AI E-Commerce Chatbot!")
                    break
                
                elif user_input.lower() == 'reset':
                    self.reset_conversation()
                    continue
                
                elif user_input.lower() == 'summary':
                    summary = self.get_conversation_summary()
                    print(f"\n📊 Conversation Summary:")
                    print(f"   Messages: {summary['total_messages']}")
                    print(f"   Current Agent: {summary.get('current_agent', 'None')}")
                    print(f"   State: {summary.get('conversation_state', 'Unknown')}")
                    print(f"   Products Mentioned: {', '.join(summary.get('products_mentioned', []))}")
                    print(f"   Orders Created: {', '.join(summary.get('orders_created', []))}")
                    continue
                
                elif user_input.lower() == 'help':
                    print(f"\n🔧 Available Commands:")
                    print(f"   quit/exit/bye - End conversation")
                    print(f"   reset - Start new conversation")
                    print(f"   summary - Show conversation summary")
                    print(f"   help - Show this help message")
                    print(f"\n🛍️ Example Queries:")
                    print(f"   'What laptops do you have under $1500?'")
                    print(f"   'Show me iPhone pricing'")
                    print(f"   'I want to buy the MacBook Pro'")
                    print(f"   'Check my order status #ORD-20241213-ABC12345'")
                    continue
                
                elif not user_input:
                    print("⚠️ Please enter a message.")
                    continue
                
                # Process the message
                print("\n🤔 Processing...")
                result = self.process_user_message(user_input)
                
                # Display response
                if result.get("success"):
                    agent_name = result.get("agent", "assistant")
                    agent_emoji = "🔍" if agent_name == "rag_agent" else "🛒" if agent_name == "order_agent" else "🤖"
                    
                    print(f"\n{agent_emoji} Assistant ({agent_name}): {result['response']}")
                    
                    # Show function calls if in debug mode
                    if settings.debug and result.get("function_calls"):
                        print(f"\n🔧 Functions Used: {[fc['name'] for fc in result['function_calls']]}")
                        
                    # Show handoff information
                    if result.get("handoff_occurred"):
                        print(f"🔄 Handoff: {result['handoff_from']} → {result['handoff_to']}")
                        
                else:
                    print(f"\n❌ Error: {result.get('response', 'Unknown error occurred')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
                
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                if settings.debug:
                    import traceback
                    traceback.print_exc()


def main():
    """Main function"""
    try:
        # Create chatbot instance
        chatbot = ECommerceChatbot()
        
        # Check if we have sample data
        collection_info = chatbot.vector_store.get_collection_info()
        if collection_info["document_count"] == 0:
            print("\n⚠️ No products found in vector database.")
            print("   Run the data loading script to add sample products.")
            print("   Example: python scripts/load_sample_data.py")
        
        # Start interactive chat
        chatbot.run_interactive_chat()
        
    except Exception as e:
        print(f"❌ Application failed to start: {str(e)}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()