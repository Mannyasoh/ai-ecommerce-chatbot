from dataclasses import dataclass
from typing import Any


@dataclass
class BaseResponse:
    success: bool
    error: str | None = None


@dataclass
class ProductSearchResponse:
    success: bool
    query: str
    products_found: int
    products: list[dict[str, Any]]
    error: str | None = None
    search_metadata: dict[str, Any] | None = None


@dataclass
class ProductDetailsResponse:
    success: bool
    product: dict[str, Any] | None
    error: str | None = None


@dataclass
class ProductAvailabilityResponse:
    success: bool
    product_name: str
    available: bool
    error: str | None = None
    product_id: str | None = None
    stock_status: str | None = None
    price: float | None = None
    message: str | None = None
    alternatives: list[dict[str, Any]] | None = None


@dataclass
class CategoryProductsResponse:
    success: bool
    category: str
    products_found: int
    products: list[dict[str, Any]]
    error: str | None = None


@dataclass
class OrderCreationResponse:
    success: bool
    order_id: str | None
    error: str | None = None
    product_name: str | None = None
    product_id: str | None = None
    quantity: int | None = None
    unit_price: float | None = None
    total_price: float | None = None
    status: str | None = None
    message: str | None = None


@dataclass
class OrderStatusResponse:
    success: bool
    order: dict[str, Any] | None
    error: str | None = None
    message: str | None = None


@dataclass
class OrderUpdateResponse:
    success: bool
    error: str | None = None
    order_id: str | None = None
    new_status: str | None = None
    message: str | None = None


@dataclass
class OrderCancellationResponse:
    success: bool
    error: str | None = None
    order_id: str | None = None
    status: str | None = None
    reason: str | None = None
    message: str | None = None


@dataclass
class OrderValidationResponse:
    success: bool
    valid: bool
    error: str | None = None
    product: dict[str, Any] | None = None
    quantity: int | None = None
    total_price: float | None = None
    message: str | None = None


@dataclass
class AgentResponse:
    success: bool
    agent: str
    response: str
    error: str | None = None
    function_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    handoff_occurred: bool = False
    handoff_from: str | None = None
    handoff_to: str | None = None


@dataclass
class ConversationSummary:
    total_messages: int
    current_agent: str | None
    conversation_state: str
    products_mentioned: list[str]
    orders_created: list[str]
    agent_switches: int
