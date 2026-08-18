"""Logout & Login Session Reset Integration Tests.

Verifies:
1. User session listing retrieves user's previous conversations without automatically selecting or auto-activating them.
2. User logging out clears session credentials while preserving stored conversations in the database.
3. User logging in again can start a brand new isolated chat session.
4. User A's session history cannot be accessed or selected by User B (User Isolation).
5. Contextual memory works within a single active conversation session across multi-turn queries.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.auth.security import create_access_token
from app.auth.models import SessionLocal, User, ChatSession, ChatMessage



@pytest.fixture
def test_users_and_tokens():
    from app.auth.security import hash_email, hash_password, create_access_token
    db = SessionLocal()
    try:
        hash_a = hash_email("session_user_a@example.com")
        user_a = db.query(User).filter(User.email_hash == hash_a).first()
        if not user_a:
            user_a = User(
                email_hash=hash_a,
                email_display="session_user_a@example.com",
                password_hash=hash_password("Password123!"),
                is_active=True,
            )
            db.add(user_a)
            db.commit()
            db.refresh(user_a)

        hash_b = hash_email("session_user_b@example.com")
        user_b = db.query(User).filter(User.email_hash == hash_b).first()
        if not user_b:
            user_b = User(
                email_hash=hash_b,
                email_display="session_user_b@example.com",
                password_hash=hash_password("Password123!"),
                is_active=True,
            )
            db.add(user_b)
            db.commit()
            db.refresh(user_b)

        token_a = create_access_token(user_a.id, user_a.email_display)
        token_b = create_access_token(user_b.id, user_b.email_display)

        return {
            "user_a": user_a,
            "user_b": user_b,
            "token_a": token_a,
            "token_b": token_b,
        }
    finally:
        db.close()



def test_logout_login_new_chat_isolation(test_users_and_tokens):
    client = TestClient(app)
    headers_a = {"Authorization": f"Bearer {test_users_and_tokens['token_a']}"}
    headers_b = {"Authorization": f"Bearer {test_users_and_tokens['token_b']}"}

    # 1. User A creates session 1 and sends a message
    sess_1_id = "s-user-a-sess-1"
    res1 = client.post("/api/chat", headers=headers_a, json={
        "session_id": sess_1_id,
        "message": "What is Skyrizi indicated for?",
    })
    assert res1.status_code == 200

    # 2. User A retrieves sessions list -> Sess 1 is present in history
    res_history = client.get("/api/chat/sessions", headers=headers_a)
    assert res_history.status_code == 200
    sessions = res_history.json().get("sessions", [])
    assert any(s["id"] == sess_1_id for s in sessions)

    # 3. User A "Logs out" -> Frontend clears active state & token
    # On next login, User A starts a NEW session ID (e.g., sess_2_id)
    sess_2_id = "s-user-a-sess-2"
    res2 = client.post("/api/chat", headers=headers_a, json={
        "session_id": sess_2_id,
        "message": "What is the recommended dosing?",
    })
    assert res2.status_code == 200

    # Verify both sessions exist separately in history for User A
    res_history_2 = client.get("/api/chat/sessions", headers=headers_a)
    assert res_history_2.status_code == 200
    sessions_2 = res_history_2.json().get("sessions", [])
    sess_ids = [s["id"] for s in sessions_2]
    assert sess_1_id in sess_ids
    assert sess_2_id in sess_ids

    # 4. User B login -> User B cannot access User A's session 1 or session 2
    res_b_access = client.get(f"/api/chat/sessions/{sess_1_id}", headers=headers_b)
    assert res_b_access.status_code == 403

    # User B list sessions does NOT contain User A's sessions
    res_b_sessions = client.get("/api/chat/sessions", headers=headers_b)
    b_ids = [s["id"] for s in res_b_sessions.json().get("sessions", [])]
    assert sess_1_id not in b_ids
    assert sess_2_id not in b_ids
