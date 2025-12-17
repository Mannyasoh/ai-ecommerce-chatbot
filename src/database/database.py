"""Database management and operations"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings
from .models import OrderModel, OrderStatus, Product

Base = declarative_base()


class ProductTable(Base):
    """SQLAlchemy Product table"""

    __tablename__ = "products"

    product_id = Column(String(50), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    stock_status = Column(String(20), nullable=False)
    category = Column(String(100), nullable=False)
    specifications = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class OrderTable(Base):
    """SQLAlchemy Order table"""

    __tablename__ = "orders"

    order_id = Column(String(50), primary_key=True)
    product_name = Column(String(200), nullable=False)
    product_id = Column(String(50))
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    status = Column(String(20), default="pending")
    customer_info = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Database manager for CRUD operations"""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager"""
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url, echo=settings.database_echo)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self) -> None:
        """Create database tables"""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def generate_order_id(self) -> str:
        """Generate unique order ID"""
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"ORD-{timestamp}-{unique_id}"

    def create_order(self, order: OrderModel) -> str:
        """Create new order and return order_id"""
        with self.get_session() as session:
            try:
                # Generate order ID if not provided
                if not order.order_id or order.order_id == "":
                    order.order_id = self.generate_order_id()

                # Create database record
                db_order = OrderTable(
                    order_id=order.order_id,
                    product_name=order.product_name,
                    product_id=order.product_id,
                    quantity=order.quantity,
                    unit_price=order.unit_price,
                    total_price=order.total_price,
                    status=order.status.value,
                    customer_info=order.customer_info,
                )

                session.add(db_order)
                session.commit()
                session.refresh(db_order)

                return order.order_id

            except Exception as e:
                session.rollback()
                raise Exception(f"Failed to create order: {str(e)}")

    def get_order(self, order_id: str) -> Optional[OrderModel]:
        """Retrieve order by ID"""
        with self.get_session() as session:
            try:
                db_order = (
                    session.query(OrderTable).filter(OrderTable.order_id == order_id).first()
                )

                if not db_order:
                    return None

                return OrderModel(
                    order_id=db_order.order_id,
                    product_name=db_order.product_name,
                    product_id=db_order.product_id,
                    quantity=db_order.quantity,
                    unit_price=db_order.unit_price,
                    total_price=db_order.total_price,
                    status=OrderStatus(db_order.status),
                    customer_info=db_order.customer_info,
                    created_at=db_order.created_at,
                    updated_at=db_order.updated_at,
                )

            except Exception as e:
                raise Exception(f"Failed to retrieve order: {str(e)}")

    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        """Update order status"""
        with self.get_session() as session:
            try:
                db_order = (
                    session.query(OrderTable).filter(OrderTable.order_id == order_id).first()
                )

                if not db_order:
                    return False

                db_order.status = status.value
                db_order.updated_at = datetime.utcnow()
                session.commit()

                return True

            except Exception as e:
                session.rollback()
                raise Exception(f"Failed to update order status: {str(e)}")

    def get_orders_by_status(self, status: OrderStatus) -> List[OrderModel]:
        """Get orders by status"""
        with self.get_session() as session:
            try:
                db_orders = (
                    session.query(OrderTable).filter(OrderTable.status == status.value).all()
                )

                return [
                    OrderModel(
                        order_id=order.order_id,
                        product_name=order.product_name,
                        product_id=order.product_id,
                        quantity=order.quantity,
                        unit_price=order.unit_price,
                        total_price=order.total_price,
                        status=OrderStatus(order.status),
                        customer_info=order.customer_info,
                        created_at=order.created_at,
                        updated_at=order.updated_at,
                    )
                    for order in db_orders
                ]

            except Exception as e:
                raise Exception(f"Failed to retrieve orders by status: {str(e)}")

    def add_product(self, product: Product) -> bool:
        """Add product to database"""
        with self.get_session() as session:
            try:
                db_product = ProductTable(
                    product_id=product.product_id,
                    name=product.name,
                    description=product.description,
                    price=product.price,
                    stock_status=product.stock_status,
                    category=product.category,
                    specifications=product.specifications,
                )

                session.add(db_product)
                session.commit()
                return True

            except Exception as e:
                session.rollback()
                raise Exception(f"Failed to add product: {str(e)}")

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID"""
        with self.get_session() as session:
            try:
                db_product = (
                    session.query(ProductTable).filter(ProductTable.product_id == product_id).first()
                )

                if not db_product:
                    return None

                return Product(
                    product_id=db_product.product_id,
                    name=db_product.name,
                    description=db_product.description,
                    price=db_product.price,
                    stock_status=db_product.stock_status,
                    category=db_product.category,
                    specifications=db_product.specifications,
                    created_at=db_product.created_at,
                )

            except Exception as e:
                raise Exception(f"Failed to retrieve product: {str(e)}")

    def search_products(
        self, query: str, category: Optional[str] = None, limit: int = 10
    ) -> List[Product]:
        """Search products by name or description"""
        with self.get_session() as session:
            try:
                query_obj = session.query(ProductTable)

                # Text search
                search_filter = (
                    ProductTable.name.ilike(f"%{query}%") |
                    ProductTable.description.ilike(f"%{query}%")
                )
                query_obj = query_obj.filter(search_filter)

                # Category filter
                if category:
                    query_obj = query_obj.filter(ProductTable.category.ilike(f"%{category}%"))

                # Limit results
                db_products = query_obj.limit(limit).all()

                return [
                    Product(
                        product_id=product.product_id,
                        name=product.name,
                        description=product.description,
                        price=product.price,
                        stock_status=product.stock_status,
                        category=product.category,
                        specifications=product.specifications,
                        created_at=product.created_at,
                    )
                    for product in db_products
                ]

            except Exception as e:
                raise Exception(f"Failed to search products: {str(e)}")

    def get_all_products(self) -> List[Product]:
        """Get all products"""
        with self.get_session() as session:
            try:
                db_products = session.query(ProductTable).all()

                return [
                    Product(
                        product_id=product.product_id,
                        name=product.name,
                        description=product.description,
                        price=product.price,
                        stock_status=product.stock_status,
                        category=product.category,
                        specifications=product.specifications,
                        created_at=product.created_at,
                    )
                    for product in db_products
                ]

            except Exception as e:
                raise Exception(f"Failed to retrieve all products: {str(e)}")

    def close(self) -> None:
        """Close database connection"""
        self.engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()