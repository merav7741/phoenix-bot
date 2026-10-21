import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class FloatingChatTests(unittest.TestCase):
    def test_answer_runs_in_background_and_locks_only_send(self):
        answer_started = threading.Event()
        allow_answer = threading.Event()

        def fake_answer(question, history, on_progress=None):
            answer_started.set()
            if on_progress:
                on_progress("בודקת מידע")
            if not allow_answer.wait(timeout=5):
                raise TimeoutError("Mock answer was not released by the test.")
            return {"answer": "תשובת בדיקה", "sources": []}

        with patch("rag.get_answer", side_effect=fake_answer):
            app = AppTest.from_file(
                Path(__file__).resolve().parents[1] / "app.py",
                default_timeout=20,
            ).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button[0].key, "open_chat")

            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn(
                "שלום, איך אפשר לעזור?",
                "\n".join(item.value for item in app.markdown),
            )
            self.assertTrue(app.session_state["chat_notice_visible"])
            self.assertTrue(
                any(button.key == "dismiss_chat_notice" for button in app.button)
            )

            next(
                button
                for button in app.button
                if button.key == "dismiss_chat_notice"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["chat_notice_visible"])

            suggestion = next(
                button for button in app.button if button.key == "suggestion_0"
            )
            suggestion.click().run()
            self.assertTrue(answer_started.wait(timeout=2))
            self.assertTrue(
                any(
                    ".st-key-chat_panel" in item.value
                    and "scrollHeight" in item.value
                    for item in app.html
                )
            )

            send_button = next(
                button for button in app.button if button.key == "send_chat"
            )
            self.assertTrue(send_button.disabled)
            self.assertFalse(app.text_input[0].disabled)

            app.text_input[0].set_value("שאלה נוספת להמשך").run()
            self.assertEqual(
                app.session_state["chat_question"],
                "שאלה נוספת להמשך",
            )
            self.assertTrue(
                next(button for button in app.button if button.key == "send_chat").disabled
            )

            allow_answer.set()
            future = app.session_state["answer_future"]
            deadline = time.monotonic() + 2
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(future.done())

            app.run()
            self.assertFalse(app.exception)
            self.assertIsNone(app.session_state["answer_future"])
            self.assertEqual(app.session_state["messages"][-1]["content"], "תשובת בדיקה")
            self.assertFalse(
                next(button for button in app.button if button.key == "send_chat").disabled
            )

            chat_messages = list(app.session_state["messages"])
            next(
                button for button in app.button if button.key == "minimize_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["chat_open"])
            self.assertTrue(app.session_state["chat_minimized"])
            self.assertEqual(app.session_state["messages"], chat_messages)
            self.assertTrue(
                any(button.key == "restore_chat" for button in app.button)
            )

            next(
                button for button in app.button if button.key == "restore_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["chat_open"])
            self.assertFalse(app.session_state["chat_minimized"])
            self.assertEqual(app.session_state["messages"], chat_messages)
            self.assertTrue(
                any(
                    "תשובת בדיקה" in item.value
                    for item in app.markdown
                )
            )

            next(
                button for button in app.button if button.key == "close_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["chat_open"])
            self.assertFalse(app.session_state["chat_minimized"])
            self.assertEqual(app.session_state["messages"], chat_messages)


if __name__ == "__main__":
    unittest.main()
