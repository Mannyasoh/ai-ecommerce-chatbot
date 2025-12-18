from dataclasses import asdict

from ..database.database import db_manager
from ..database.models import OrderModel, OrderRequest, OrderStatus
from ..models.responses import OrderCreationResponse

try:
    from langfuse.decorators import observe

    LANGFUSE_AVAILABLE = True
except ImportError:

    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    LANGFUSE_AVAILABLE = False


@observe(name="create-order")
def create_order(
    product_name: str,
    quantity: int = 1,
    customer_info: dict[str, str] | None = None,
) -> dict:
    try:
        OrderRequest(
            product_name=product_name, quantity=quantity, customer_info=customer_info
        )

        # Use vector store search for consistency with RAG agent
        from ..vector_store.chroma_store import vector_store

        search_results = vector_store.search_products(query=product_name, n_results=5)
        products = [result.product for result in search_results]

        if not products:
            response = OrderCreationResponse(
                success=False,
                error=f"Product '{product_name}' not found in catalog",
                order_id=None,
                message=f"Sorry, I couldn't find '{product_name}' in our product catalog. Please check the product name and try again.",
            )
            return asdict(response)

        target_product = None
        for product in products:
            if product.name.lower() == product_name.lower():
                target_product = product
                break
            elif product_name.lower() in product.name.lower():
                target_product = product
                break

        if not target_product:
            target_product = products[0]

        if target_product.stock_status != "in_stock":
            response = OrderCreationResponse(
                success=False,
                error=f"Product '{target_product.name}' is currently {target_product.stock_status}",
                order_id=None,
                message=f"Sorry, {target_product.name} is currently {target_product.stock_status}. Please try again later or choose a different product.",
            )
            return asdict(response)

        unit_price = target_product.price
        total_price = unit_price * quantity

        order = OrderModel(
            order_id="",
            product_name=target_product.name,
            product_id=target_product.product_id,
            quantity=quantity,
            unit_price=unit_price,
            total_price=total_price,
            status=OrderStatus.PENDING,
            customer_info=customer_info,
        )

        order_id = db_manager.create_order(order)

        message = f"Order confirmed! Your order for {quantity}x {target_product.name} has been placed successfully."
        if total_price > 0:
            message += f" Total: ${total_price:.2f}"
        message += f" Order ID: #{order_id}"

        response = OrderCreationResponse(
            success=True,
            order_id=order_id,
            product_name=target_product.name,
            product_id=target_product.product_id,
            quantity=quantity,
            unit_price=unit_price,
            total_price=total_price,
            status="pending",
            message=message,
        )
        return asdict(response)

    except Exception as e:
        response = OrderCreationResponse(
            success=False,
            error=str(e),
            order_id=None,
            message=f"Sorry, there was an error processing your order: {str(e)}",
        )
        return asdict(response)


@observe(name="get-order-status")
def get_order_status(order_id: str) -> dict[str, str | bool | dict | None]:
    try:
        order = db_manager.get_order(order_id)

        if not order:
            return {
                "success": False,
                "error": f"Order with ID '{order_id}' not found",
                "order": None,
                "message": f"I couldn't find an order with ID {order_id}. Please check the order ID and try again.",
            }

        return {
            "success": True,
            "order": {
                "order_id": order.order_id,
                "product_name": order.product_name,
                "quantity": order.quantity,
                "total_price": order.total_price,
                "status": order.status.value,
                "created_at": (
                    order.created_at.isoformat() if order.created_at else None
                ),
                "updated_at": (
                    order.updated_at.isoformat() if order.updated_at else None
                ),
            },
            "message": f"Order #{order.order_id}: {order.quantity}x {order.product_name} - Status: {order.status.value.title()}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "order": None,
            "message": f"Error retrieving order status: {str(e)}",
        }


def update_order_status(order_id: str, new_status: str) -> dict[str, str | bool]:
    try:
        try:
            status_enum = OrderStatus(new_status.lower())
        except ValueError:
            valid_statuses = [status.value for status in OrderStatus]
            return {
                "success": False,
                "error": f"Invalid status '{new_status}'. Valid statuses: {valid_statuses}",
                "message": f"Invalid order status. Please use one of: {', '.join(valid_statuses)}",
            }

        success = db_manager.update_order_status(order_id, status_enum)

        if not success:
            return {
                "success": False,
                "error": f"Order with ID '{order_id}' not found",
                "message": f"Could not find order {order_id} to update.",
            }

        return {
            "success": True,
            "order_id": order_id,
            "new_status": status_enum.value,
            "message": f"Order #{order_id} status updated to {status_enum.value.title()}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error updating order status: {str(e)}",
        }


def cancel_order(
    order_id: str, reason: str | None = None
) -> dict[str, str | bool | None]:
    try:
        order = db_manager.get_order(order_id)

        if not order:
            return {
                "success": False,
                "error": f"Order with ID '{order_id}' not found",
                "message": f"Could not find order {order_id} to cancel.",
            }

        if order.status in [
            OrderStatus.SHIPPED,
            OrderStatus.DELIVERED,
            OrderStatus.CANCELLED,
        ]:
            return {
                "success": False,
                "error": f"Order {order_id} cannot be cancelled (current status: {order.status.value})",
                "message": f"Order #{order_id} cannot be cancelled because it is already {order.status.value}.",
            }

        success = db_manager.update_order_status(order_id, OrderStatus.CANCELLED)

        if not success:
            return {
                "success": False,
                "error": f"Failed to cancel order {order_id}",
                "message": f"There was an error cancelling order {order_id}. Please try again.",
            }

        message = f"Order #{order_id} has been cancelled successfully."
        if reason:
            message += f" Reason: {reason}"

        return {
            "success": True,
            "order_id": order_id,
            "status": "cancelled",
            "reason": reason,
            "message": message,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Error cancelling order: {str(e)}",
        }


def validate_order_details(
    product_name: str, quantity: int
) -> dict[str, str | bool | int | float | dict | None]:
    try:
        # Use vector store search for consistency with RAG agent
        from ..vector_store.chroma_store import vector_store

        search_results = vector_store.search_products(query=product_name, n_results=3)
        products = [result.product for result in search_results]

        if not products:
            return {
                "success": False,
                "valid": False,
                "error": f"Product '{product_name}' not found",
                "message": f"I couldn't find '{product_name}' in our catalog.",
            }

        target_product = None
        for product in products:
            if product.name.lower() == product_name.lower():
                target_product = product
                break
            elif product_name.lower() in product.name.lower():
                target_product = product
                break

        if not target_product:
            target_product = products[0]

        available = target_product.stock_status == "in_stock"
        total_price = target_product.price * quantity

        validation_message = f"Product: {target_product.name} (${target_product.price})"
        validation_message += f"\nQuantity: {quantity}"
        validation_message += f"\nTotal Price: ${total_price:.2f}"
        validation_message += f"\nAvailability: {'Available' if available else 'Not available'} ({target_product.stock_status})"

        return {
            "success": True,
            "valid": available,
            "product": {
                "name": target_product.name,
                "product_id": target_product.product_id,
                "price": target_product.price,
                "stock_status": target_product.stock_status,
                "available": available,
            },
            "quantity": quantity,
            "total_price": total_price,
            "message": validation_message,
        }

    except Exception as e:
        return {
            "success": False,
            "valid": False,
            "error": str(e),
            "message": f"Error validating order: {str(e)}",
        }


ORDER_FUNCTION_SCHEMAS = [
    {
        "name": "create_order",
        "description": "Create a new order for a specified product when customer confirms purchase",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to order",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Quantity to order",
                    "minimum": 1,
                    "default": 1,
                },
                "customer_info": {
                    "type": "object",
                    "description": "Optional customer information (name, email, phone, address)",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "address": {"type": "string"},
                    },
                },
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Get the current status of an existing order",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Unique order identifier in format ORD-YYYYMMDD-XXXXXXXX (without # prefix if provided by user)",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "validate_order_details",
        "description": "Validate order details before creating order (check availability, calculate total price)",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product to validate",
                },
                "quantity": {
                    "type": "integer",
                    "description": "Quantity to validate",
                    "minimum": 1,
                    "default": 1,
                },
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an existing order",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Unique order identifier to cancel",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional reason for cancellation",
                },
            },
            "required": ["order_id"],
        },
    },
]
