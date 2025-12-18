import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.orchestrator import orchestrator
from src.config import settings, validate_environment
from src.database.database import db_manager
from src.database.models import ChatMessage
from src.logging_config import configure_logging
from src.vector_store.chroma_store import vector_store


class ECommerceChatbot:
    def __init__(self) -> None:
        validate_environment()
        self.chat_history: list[ChatMessage] = []
        self.orchestrator = orchestrator
        self.db_manager = db_manager
        self.vector_store = vector_store
        self.logger = logger.bind(component="chatbot")

        self.logger.info("E-Commerce AI Chatbot initialized successfully")
        collection_info = self.vector_store.get_collection_info()
        self.logger.info("Vector DB status", **collection_info)

    def add_message_to_history(
        self,
        role: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            function_call=None,
            tool_calls=None,
        )
        self.chat_history.append(message)
        if len(self.chat_history) > settings.max_chat_history:
            self.chat_history = self.chat_history[-settings.max_chat_history :]

    def process_user_message(self, message: str) -> dict[str, str | bool | dict | list]:
        try:
            self.add_message_to_history("user", message)
            result = self.orchestrator.process_message(message, self.chat_history)
            if result.get("success"):
                response_content = str(result.get("response", ""))
                self.add_message_to_history("assistant", response_content)
            return result
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            error_response = {
                "success": False,
                "response": error_message,
                "error": str(e),
                "agent": "system",
            }
            self.add_message_to_history("assistant", error_message)
            return error_response

    def get_conversation_summary(self) -> dict[str, str | int | list]:
        return self.orchestrator.get_conversation_summary(self.chat_history)

    def reset_conversation(self) -> None:
        self.chat_history = []
        self.orchestrator.reset_conversation_state()
        self.logger.info("Conversation reset successfully")
        print("Conversation reset successfully!")


    def run_interactive_chat(self) -> None:
        print("\n" + "=" * 60)
        print("Welcome to the AI E-Commerce Chatbot!")
        print("=" * 60)
        print("Ask me about products, prices, or place orders.")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'reset' to start a new conversation.")
        print("Type 'summary' to see conversation summary.")
        print("Type 'help' for more commands.")
        print("-" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\nThank you for using our AI E-Commerce Chatbot!")
                    break

                elif user_input.lower() == "reset":
                    self.reset_conversation()
                    continue

                elif user_input.lower() == "summary":
                    summary = self.get_conversation_summary()
                    print("\nConversation Summary:")
                    print(f"   Messages: {summary['total_messages']}")
                    print(f"   Current Agent: {summary.get('current_agent', 'None')}")
                    print(f"   State: {summary.get('conversation_state', 'Unknown')}")
                    products: list[str] = summary.get("products_mentioned", [])
                    print(f"   Products Mentioned: {', '.join(products)}")
                    orders: list[str] = summary.get("orders_created", [])
                    print(f"   Orders Created: {', '.join(orders)}")
                    continue

                elif user_input.lower() == "help":
                    print("\nAvailable Commands:")
                    print("   quit/exit/bye - End conversation")
                    print("   reset - Start new conversation")
                    print("   summary - Show conversation summary")
                    print("   help - Show this help message")
                    print("\nExample Queries:")
                    print("   'What laptops do you have under $1500?'")
                    print("   'Show me iPhone pricing'")
                    print("   'I want to buy the MacBook Pro'")
                    print("   'Check my order status #ORD-20241213-ABC12345'")
                    continue

                elif not user_input:
                    print("Please enter a message.")
                    continue

                print("\nProcessing...")
                result = self.process_user_message(user_input)

                if result.get("success"):
                    agent_name = result.get("agent", "assistant")
                    print(f"\nAssistant ({agent_name}): {result['response']}")

                    if settings.debug and result.get("function_calls"):
                        function_calls = result.get("function_calls", [])
                        if isinstance(function_calls, list):
                            func_names = [fc["name"] for fc in function_calls]
                            print(f"\nFunctions Used: {func_names}")

                    if result.get("handoff_occurred"):
                        handoff_from = result["handoff_from"]
                        handoff_to = result["handoff_to"]
                        handoff_msg = f"{handoff_from} -> {handoff_to}"
                        print(f"Handoff: {handoff_msg}")

                else:
                    print(
                        f"\nError: {result.get('response', 'Unknown error occurred')}"
                    )

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break

            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                if settings.debug:
                    import traceback

                    traceback.print_exc()


def main() -> None:
    try:
        # Configure logging first
        configure_logging()
        logger.info("Application starting up")

        chatbot = ECommerceChatbot()
        collection_info = chatbot.vector_store.get_collection_info()
        if collection_info["document_count"] == 0:
            logger.warning("No products found in vector database", **collection_info)
            print("\nNo products found in vector database.")
            print("   Run the data loading script to add sample products.")
            print("   Example: python scripts/load_sample_data.py")

        logger.info("Starting interactive chat session")
        chatbot.run_interactive_chat()
    except Exception as e:
        logger.critical("Application failed to start", error=str(e))
        print(f"Application failed to start: {str(e)}")
        if settings.debug:
            import traceback

            logger.debug("Full traceback", traceback=traceback.format_exc())
            traceback.print_exc()
        sys.exit(1)
    finally:
        logger.info("Application shutting down")


if __name__ == "__main__":
    main()
