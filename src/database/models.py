"""Pydantic models for data validation and serialization"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class OrderStatus(str, Enum):
    """Order status enumeration"""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Product(BaseModel):
    """Product model with comprehensive validation"""

    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: str = Field(..., min_length=10, description="Product description")
    price: float = Field(..., gt=0, description="Price in USD")
    stock_status: str = Field(
        ..., description="in_stock, out_of_stock, or discontinued"
    )
    category: str = Field(..., min_length=1, description="Product category")
    specifications: Optional[Dict[str, Any]] = Field(
        None, description="Product specifications"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("stock_status")
    def validate_stock_status(cls, v: str) -> str:
        """Validate stock status values"""
        allowed_status = ["in_stock", "out_of_stock", "discontinued"]
        if v not in allowed_status:
            raise ValueError(f"Stock status must be one of: {allowed_status}")
        return v

    class Config:
        """Pydantic configuration"""

        json_encoders = {datetime: lambda v: v.isoformat()}


class OrderModel(BaseModel):
    """Order model with business logic validation"""

    order_id: str = Field(..., description="Unique order identifier")
    product_name: str = Field(..., min_length=1, description="Product name")
    product_id: Optional[str] = Field(None, description="Product identifier")
    quantity: int = Field(..., gt=0, description="Must be positive integer")
    unit_price: float = Field(..., gt=0, description="Price per unit")
    total_price: float = Field(..., gt=0, description="Total order price")
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    customer_info: Optional[Dict[str, str]] = Field(
        None, description="Customer information"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    @validator("total_price")
    def validate_total_price(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate total price equals quantity × unit_price"""
        quantity = values.get("quantity", 0)
        unit_price = values.get("unit_price", 0)
        expected_total = quantity * unit_price
        if abs(v - expected_total) > 0.01:  # Allow small floating point differences
            raise ValueError("Total price must equal quantity × unit_price")
        return v

    @validator("updated_at", always=True)
    def set_updated_at(cls, v: Optional[datetime]) -> datetime:
        """Set updated_at to current time"""
        return v or datetime.utcnow()

    class Config:
        """Pydantic configuration"""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ChatMessage(BaseModel):
    """Chat message model"""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., min_length=1, description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    function_call: Optional[Dict[str, Any]] = Field(
        None, description="Function call details"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls for OpenAI function calling"
    )

    @validator("role")
    def validate_role(cls, v: str) -> str:
        """Validate message role"""
        allowed_roles = ["user", "assistant", "system", "function", "tool"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {allowed_roles}")
        return v

    class Config:
        """Pydantic configuration"""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ProductSearchRequest(BaseModel):
    """Product search request model"""

    query: str = Field(..., min_length=1, description="Search query")
    category: Optional[str] = Field(None, description="Category filter")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results")
    price_min: Optional[float] = Field(None, ge=0, description="Minimum price")
    price_max: Optional[float] = Field(None, ge=0, description="Maximum price")

    @validator("price_max")
    def validate_price_range(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        """Validate price range"""
        price_min = values.get("price_min")
        if v is not None and price_min is not None and v < price_min:
            raise ValueError("price_max must be greater than price_min")
        return v


class OrderRequest(BaseModel):
    """Order creation request model"""

    product_name: str = Field(..., min_length=1, description="Product name")
    quantity: int = Field(..., gt=0, description="Quantity to order")
    customer_info: Optional[Dict[str, str]] = Field(
        None, description="Customer information"
    )


class VectorSearchResult(BaseModel):
    """Vector search result model"""

    product: Product
    score: float = Field(..., ge=0, le=1, description="Similarity score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        """Pydantic configuration"""

        arbitrary_types_allowed = True