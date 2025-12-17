import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

from loguru import logger
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeMeta,
    Session,
    registry,
    scoped_session,
    sessionmaker,
)

from ..config import settings
from .models import OrderModel, OrderStatus, Product

mapper_registry = registry()
Base: DeclarativeMeta = mapper_registry.generate_base()


class ProductTable(Base):
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


def _db_row_to_product(db_product: ProductTable) -> Product:
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


def _db_row_to_order(db_order: OrderTable) -> OrderModel:
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


class DatabaseManager:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or settings.database_url
        self.logger = logger.bind(component="database")
        self.logger.info("Initializing database connection", url=self.database_url)

        self.engine: Engine = create_engine(
            self.database_url, echo=settings.database_echo
        )
        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )
        self.create_tables()

    def create_tables(self) -> None:
        self.logger.debug("Creating database tables")
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def generate_order_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4())[:8].upper()
        return f"ORD-{timestamp}-{unique_id}"

    def create_order(self, order: OrderModel) -> str:
        with self.get_session() as session:
            if not order.order_id or order.order_id == "":
                order.order_id = self.generate_order_id()

            self.logger.info(
                "Creating new order",
                order_id=order.order_id,
                product_name=order.product_name,
                quantity=order.quantity,
                total_price=order.total_price,
            )

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
            session.flush()
            self.logger.debug("Order created successfully", order_id=order.order_id)
            return order.order_id

    def get_order(self, order_id: str) -> OrderModel | None:
        with self.get_session() as session:
            db_order = (
                session.query(OrderTable)
                .filter(OrderTable.order_id == order_id)
                .first()
            )
            return _db_row_to_order(db_order) if db_order else None

    def update_order_status(self, order_id: str, status: OrderStatus) -> bool:
        with self.get_session() as session:
            db_order = (
                session.query(OrderTable)
                .filter(OrderTable.order_id == order_id)
                .first()
            )

            if not db_order:
                return False

            db_order.status = status.value
            db_order.updated_at = datetime.utcnow()
            return True

    def get_orders_by_status(self, status: OrderStatus) -> list[OrderModel]:
        with self.get_session() as session:
            db_orders = (
                session.query(OrderTable)
                .filter(OrderTable.status == status.value)
                .all()
            )
            return [_db_row_to_order(order) for order in db_orders]

    def add_product(self, product: Product) -> bool:
        with self.get_session() as session:
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
            return True

    def get_product(self, product_id: str) -> Product | None:
        with self.get_session() as session:
            db_product = (
                session.query(ProductTable)
                .filter(ProductTable.product_id == product_id)
                .first()
            )
            return _db_row_to_product(db_product) if db_product else None

    def search_products(
        self, query: str, category: str | None = None, limit: int = 10
    ) -> list[Product]:
        with self.get_session() as session:
            query_obj = session.query(ProductTable)

            search_filter = ProductTable.name.ilike(
                f"%{query}%"
            ) | ProductTable.description.ilike(f"%{query}%")
            query_obj = query_obj.filter(search_filter)

            if category:
                query_obj = query_obj.filter(
                    ProductTable.category.ilike(f"%{category}%")
                )

            db_products = query_obj.limit(limit).all()
            return [_db_row_to_product(product) for product in db_products]

    def get_all_products(self) -> list[Product]:
        with self.get_session() as session:
            db_products = session.query(ProductTable).all()
            return [_db_row_to_product(product) for product in db_products]

    def close(self) -> None:
        self.SessionLocal.remove()
        self.engine.dispose()


db_manager = DatabaseManager()
