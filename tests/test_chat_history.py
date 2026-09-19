"""Conversation persistence, retention, and chat API regression tests."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from dashboard import app
from services import chat_history


def _sse_payloads(response) -> list[dict]:
    payloads = []
    for line in response.get_data(as_text=True).splitlines():
        if line.startswith("data: "):
            payloads.append(json.loads(line[6:]))
    return payloads


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            chat_history, "CHAT_DB_PATH", Path(self.temp_dir.name) / "chat.sqlite3",
        )
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_conversation_round_trip_and_delete(self):
        conversation_id = chat_history.create_conversation("今天市场怎么样？")
        chat_history.add_message(conversation_id, "user", "今天市场怎么样？")
        chat_history.add_message(conversation_id, "assistant", "市场保持震荡。")

        rows = chat_history.list_conversations()
        self.assertEqual(rows[0]["id"], conversation_id)
        self.assertEqual(rows[0]["message_count"], 2)
        self.assertEqual(rows[0]["preview"], "市场保持震荡。")
        self.assertEqual(
            [item["role"] for item in chat_history.get_messages(conversation_id)],
            ["user", "assistant"],
        )
        self.assertTrue(chat_history.delete_conversation(conversation_id))
        self.assertFalse(chat_history.conversation_exists(conversation_id))

    def test_cleanup_removes_conversations_older_than_30_days(self):
        conversation_id = chat_history.create_conversation("过期测试")
        old_time = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        with chat_history._connect() as connection:
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (old_time, conversation_id),
            )
        self.assertEqual(chat_history.cleanup_expired(), 1)
        self.assertFalse(chat_history.conversation_exists(conversation_id))

    def test_chat_api_persists_and_reuses_server_side_history(self):
        captured_history = []

        def fake_stream(question, history, web_search_payload=None):
            captured_history.append(list(history))
            yield "测试回答"

        app.config.update(TESTING=True, BYPASS_AUTH_FOR_TESTS=True)
        with patch("dashboard.is_configured", return_value=True), patch(
            "dashboard.should_search_web", return_value=False,
        ), patch("dashboard.stream_market_chat", side_effect=fake_stream):
            client = app.test_client()
            first = client.post("/api/chat", json={
                "question": "第一个问题", "web_search": "off",
            })
            first_payloads = _sse_payloads(first)
            conversation_id = first_payloads[0]["conversation_id"]
            self.assertTrue(first_payloads[-1]["done"])

            second = client.post("/api/chat", json={
                "question": "继续追问", "conversation_id": conversation_id,
                "web_search": "off",
            })
            self.assertEqual(second.status_code, 200)
            self.assertTrue(_sse_payloads(second)[-1]["done"])

            detail = client.get(f"/api/chat/conversations/{conversation_id}")
            messages = detail.get_json()["messages"]

        self.assertEqual(len(messages), 4)
        self.assertEqual(captured_history[0], [])
        self.assertEqual(
            [(item["role"], item["content"]) for item in captured_history[1]],
            [("user", "第一个问题"), ("assistant", "测试回答")],
        )


if __name__ == "__main__":
    unittest.main()
