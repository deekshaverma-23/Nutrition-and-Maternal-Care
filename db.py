from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
import uuid

DATABASE_URL = "sqlite:///./chat.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class UserProfile(Base):
    __tablename__ = "profile"

    id = Column(String, primary_key=True, default="main")
    user_type = Column(String)
    stage = Column(String)
    diet = Column(String)
    age_months = Column(String)
    age_years = Column(String)
    child_status = Column(String)
    conditions = Column(Text)
    language = Column(String)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    title = Column(String, default="New Chat")

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"))
    role = Column(String)
    content = Column(Text)
    audio_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")



def init_db():
    Base.metadata.create_all(bind=engine)
