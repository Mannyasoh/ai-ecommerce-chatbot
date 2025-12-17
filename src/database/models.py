from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class BaseTimestampModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class Product(BaseTimestampModel):
    product_id: str = Field(...)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)
    price: float = Field(..., gt=0)
    stock_status: str = Field(...)
    category: str = Field(..., min_length=1)
    specifications: dict[str, str | int | float | bool] | None = Field(None)

    @validator("stock_status")
    def validate_stock_status(cls, v: str) -> str:
        allowed_status = ["in_stock", "out_of_stock", "discontinued"]
        if v not in allowed_status:
            raise ValueError(f"Stock status must be one of: {allowed_status}")
        return v


class OrderModel(BaseTimestampModel):
    order_id: str = Field(...)
    product_name: str = Field(..., min_length=1)
    product_id: str | None = Field(None)
    quantity: int = Field(..., gt=0)
    unit_price: float = Field(..., gt=0)
    total_price: float = Field(..., gt=0)
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    customer_info: dict[str, str] | None = Field(None)
    updated_at: datetime | None = None

    @validator("total_price")
    def validate_total_price(cls, v: float, values: dict[str, float | int]) -> float:
        quantity = values.get("quantity", 0)
        unit_price = values.get("unit_price", 0)
        expected_total = quantity * unit_price
        if abs(v - expected_total) > 0.01:
            raise ValueError("Total price must equal quantity × unit_price")
        return v

    @validator("updated_at", always=True)
    def set_updated_at(cls, v: datetime | None) -> datetime:
        return v or datetime.utcnow()


class ChatMessage(BaseTimestampModel):
    role: str = Field(...)
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    function_call: dict[str, str | dict] | None = Field(None)
    tool_calls: list[dict[str, str | dict]] | None = Field(None)

    @validator("role")
    def validate_role(cls, v: str) -> str:
        allowed_roles = ["user", "assistant", "system", "function", "tool"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {allowed_roles}")
        return v


class ProductSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    category: str | None = Field(None)
    max_results: int = Field(default=5, ge=1, le=20)
    price_min: float | None = Field(None, ge=0)
    price_max: float | None = Field(None, ge=0)

    @validator("price_max")
    def validate_price_range(
        cls, v: float | None, values: dict[str, float | None]
    ) -> float | None:
        price_min = values.get("price_min")
        if v is not None and price_min is not None and v < price_min:
            raise ValueError("price_max must be greater than price_min")
        return v


class OrderRequest(BaseModel):
    product_name: str = Field(..., min_length=1)
    quantity: int = Field(..., gt=0)
    customer_info: dict[str, str] | None = Field(None)


class VectorSearchResult(BaseModel):
    product: Product
    score: float = Field(..., ge=0, le=1)
    metadata: dict[str, str | int | float] | None = Field(None)

    class Config:
        arbitrary_types_allowed = True
