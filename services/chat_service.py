from db import SessionLocal, Conversation, Message
from sqlalchemy import desc

def create_conversation():
    db = SessionLocal()
    convo = Conversation(title="New Chat")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    db.close()
    return convo.id

def get_all_conversations(limit=50):
    db = SessionLocal()
    convos = (
        db.query(Conversation)
        .order_by(desc(Conversation.created_at))
        .limit(limit)
        .all()
    )
    db.close()
    return convos

def get_latest_conversation_id():
    db = SessionLocal()
    convo = (
        db.query(Conversation)
        .order_by(Conversation.created_at.desc())
        .first()
    )
    db.close()
    return convo.id if convo else None

def conversation_exists(conversation_id):
    if not conversation_id:
        return False
    db = SessionLocal()
    exists = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
        is not None
    )
    db.close()
    return exists


def get_conversation(conversation_id):
    if not conversation_id:
        return None
    db = SessionLocal()
    convo = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )
    db.close()
    return convo


def update_conversation_title(conversation_id, title):
    if not conversation_id:
        return
    db = SessionLocal()
    convo = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )
    if convo:
        convo.title = title
        db.commit()
    db.close()

def load_messages(conversation_id):
    if not conversation_id:
        return []
    db = SessionLocal()
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp)
        .all()
    )
    db.close()
    return messages

def save_message(
    conversation_id,
    role,
    content,
    audio_path=None,
):
    if not conversation_id:
        return
    db = SessionLocal()
    db.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            audio_path=audio_path,
        )
    )
    db.commit()
    db.close()

def bootstrap_conversation():
    """Create conversation only when first user message is sent."""
    return create_conversation()

def generate_title(llm, user_message):
    prompt = f"""
    Generate a short 3–6 word title for this conversation.

    User message:
    {user_message}

    Title:
    """
    result = llm.invoke(prompt)
    return str(result).strip().replace('"', "")

def delete_conversation(conversation_id):
    if not conversation_id:
        return
    db = SessionLocal()
    db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).delete()
    db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).delete()
    db.commit()
    db.close()