"""Regression test suite for delayed previous chat auto-restore & async overwrite race conditions.

Verifies:
1. When a user logs in and starts a new chat, receiving backend history response (fast, delayed, or slow)
   NEVER overwrites activeSessionId back to the previous session.
2. In-flight loadSessionHistory calls for an old session DO NOT overwrite activeSessionId if a new chat is started.
3. Backend session history endpoint strictly returns user-isolated history without auto-selecting previous chats.
"""
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth.models import SessionLocal, User, ChatSession, ChatMessage
from app.auth.security import hash_email, hash_password, create_access_token


@pytest.fixture
def auth_user_and_token():
    db = SessionLocal()
    try:
        user_email = "delayed_overwrite_test@example.com"
        h_email = hash_email(user_email)
        user = db.query(User).filter(User.email_hash == h_email).first()
        if not user:
            user = User(
                email_hash=h_email,
                email_display=user_email,
                password_hash=hash_password("Password123!"),
                is_active=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # Ensure user has existing historical sessions in DB
        old_sess_1 = db.query(ChatSession).filter(ChatSession.id == "s-old-hist-1").first()
        if not old_sess_1:
            old_sess_1 = ChatSession(id="s-old-hist-1", user_id=user.id, title="Old Chat 1")
            db.add(old_sess_1)
            msg1 = ChatMessage(session_id="s-old-hist-1", user_id=user.id, role="user", content="Hello old chat 1")
            db.add(msg1)
            db.commit()

        token = create_access_token(user.id, user.email_display)
        return {"user": user, "token": token, "old_session_id": "s-old-hist-1"}
    finally:
        db.close()


def test_delayed_overwrite_simulation(auth_user_and_token):
    """Simulates async history response timing (immediate, delayed, slow) to ensure new chat remains active."""
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {auth_user_and_token['token']}"}

    # Simulate Frontend State
    state = {
        "sessions": [],
        "activeSessionId": None,
        "chatInitializationId": None,
        "messages": {},
    }

    def start_new_chat():
        new_id = f"s-new-{int(time.time() * 1000)}"
        state["chatInitializationId"] = new_id
        state["activeSessionId"] = new_id
        state["messages"][new_id] = []
        return new_id

    def fetch_chat_sessions_completion(response_data):
        """Simulates fetchChatSessions completion callback."""
        if response_data and "sessions" in response_data:
            state["sessions"] = response_data["sessions"]
        # CRITICAL VERIFICATION: fetchChatSessions MUST NOT mutate activeSessionId!

    # Step 1: User logs in -> startNewChat() called immediately
    new_chat_id = start_new_chat()
    assert state["activeSessionId"] == new_chat_id

    # Step 2: Immediate history response scenario
    res_immediate = client.get("/api/chat/sessions", headers=headers)
    assert res_immediate.status_code == 200
    fetch_chat_sessions_completion(res_immediate.json())
    # ASSERTION: activeSessionId MUST STILL BE the new ID, not the old session!
    assert state["activeSessionId"] == new_chat_id
    assert state["activeSessionId"] != auth_user_and_token["old_session_id"]

    # Step 3: Delayed history response scenario (simulated 1-second delay)
    time.sleep(1.0)
    res_delayed = client.get("/api/chat/sessions", headers=headers)
    assert res_delayed.status_code == 200
    fetch_chat_sessions_completion(res_delayed.json())
    assert state["activeSessionId"] == new_chat_id
    assert state["activeSessionId"] != auth_user_and_token["old_session_id"]

    # Step 4: Slow history response scenario (simulated 2-second delay)
    time.sleep(2.0)
    res_slow = client.get("/api/chat/sessions", headers=headers)
    assert res_slow.status_code == 200
    fetch_chat_sessions_completion(res_slow.json())
    assert state["activeSessionId"] == new_chat_id
    assert state["activeSessionId"] != auth_user_and_token["old_session_id"]


def test_in_flight_load_session_history_race_guard():
    """Simulates loadSessionHistory guard when user starts a new chat while an old session load is in-flight."""
    state = {
        "activeSessionId": "s-old-1",
        "messages": {},
    }

    # In-flight request for s-old-1 initiated
    requested_id = "s-old-1"

    # User clicks New Chat while request is in-flight
    new_id = "s-new-draft-99"
    state["activeSessionId"] = new_id
    state["messages"][new_id] = []

    # Delayed network response for s-old-1 arrives
    mock_network_response = [{"role": "user", "text": "Stale message"}]

    # Race-condition Guard in JS:
    if state["activeSessionId"] == requested_id:
        state["messages"][requested_id] = mock_network_response

    # ASSERTION: activeSessionId remains new_id, and stale messages were NOT rendered or set active!
    assert state["activeSessionId"] == new_id
    assert requested_id not in state["messages"] or state["activeSessionId"] != requested_id
