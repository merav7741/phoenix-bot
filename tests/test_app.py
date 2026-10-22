import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from queue import Queue
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class FloatingChatTests(unittest.TestCase):
    def test_claim_guide_is_a_conversation_and_keeps_followups_in_topic(self):
        with (
            patch("rag.get_answer") as answer,
            patch(
                "rag.get_claim_guide",
                return_value=[
                    {
                        "title": "מידע רשמי",
                        "url": (
                            "https://www.fnx.co.il/claims-customer-service/"
                            "claims-car-insurance/claims-property-car-insurance/"
                        ),
                        "text": "קטע בדיקה מהמקור הרשמי",
                        "crawled_at": "2026-10-18",
                    }
                ],
            ) as get_claim_guide,
        ):
            app = AppTest.from_file(
                Path(__file__).resolve().parents[1] / "app.py",
                default_timeout=20,
            ).run()
            self.assertFalse(app.exception)
            self.assertFalse(
                any(button.key == "show_claim_guide" for button in app.button)
            )
            next(
                button for button in app.button if button.key == "open_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["chat_mode"], "שאלות")
            self.assertFalse(
                any(button.key == "show_claim_guide" for button in app.button)
            )
            app.segmented_control[0].set_value("ליווי").run()
            self.assertFalse(app.exception)
            self.assertIn(
                "היי, אני המלווה של הפניקס",
                "\n".join(item.value for item in app.markdown),
            )
            self.assertIn(
                "באיזה סוג ביטוח תרצי שאלווה אותך?",
                "\n".join(item.value for item in app.markdown),
            )

            next(
                button for button in app.button if button.key == "start_claim_guide"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["claim_guide_selected"],
                "רכב — נזק לרכב",
            )
            self.assertIn(
                "ספרי לי בקצרה מה קרה או מה תרצי לברר",
                "\n".join(item.value for item in app.info),
            )

            app.text_area[0].set_value("הייתה תאונה עם נזק לרכב").run()
            next(
                button
                for button in app.button
                if button.key == "submit_claim_guide_question"
            ).click().run()
            self.assertFalse(app.exception)

            app.text_area[0].set_value("איזה מסמכים כדאי לבדוק?").run()
            next(
                button
                for button in app.button
                if button.key == "submit_claim_guide_question"
            ).click().run()

        self.assertFalse(app.exception)
        self.assertEqual(
            get_claim_guide.call_count,
            2,
        )
        self.assertEqual(
            get_claim_guide.call_args_list[0].args[1],
            "הייתה תאונה עם נזק לרכב",
        )
        self.assertEqual(
            get_claim_guide.call_args_list[1].args[1],
            "איזה מסמכים כדאי לבדוק?",
        )
        answer.assert_not_called()
        self.assertEqual(len(app.session_state["claim_guide_messages"]), 2)
        self.assertIn(
            "קטע בדיקה מהמקור הרשמי",
            "\n".join(item.value for item in app.markdown),
        )

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
            self.assertTrue(any(button.key == "open_chat" for button in app.button))

            next(
                button for button in app.button if button.key == "open_chat"
            ).click().run()
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

            app.segmented_control[0].set_value("ליווי").run()
            self.assertFalse(app.exception)
            app.session_state["claim_guide_selected"] = "דירה"
            app.session_state["chat_error"] = "שגיאת בדיקה"
            app.session_state["pending_question"] = "שאלה ממתינה"
            pending_future = Future()
            app.session_state["answer_future"] = pending_future
            app.session_state["answer_progress_queue"] = Queue()
            next(
                button for button in app.button if button.key == "close_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state["chat_open"])
            self.assertFalse(app.session_state["chat_minimized"])
            self.assertEqual(app.session_state["messages"], [])
            self.assertEqual(app.session_state["chat_question"], "")
            self.assertIsNone(app.session_state["answer_future"])
            self.assertTrue(pending_future.cancelled())
            self.assertEqual(app.session_state["chat_mode"], "שאלות")
            self.assertNotIn("claim_guide_selected", app.session_state)
            self.assertNotIn("claim_guide_messages", app.session_state)
            self.assertNotIn("pending_question", app.session_state)
            self.assertNotIn("chat_error", app.session_state)
            self.assertTrue(app.session_state["chat_notice_visible"])

            next(
                button for button in app.button if button.key == "open_chat"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertIn(
                "שלום, איך אפשר לעזור?",
                "\n".join(item.value for item in app.markdown),
            )
            self.assertFalse(
                any("תשובת בדיקה" in item.value for item in app.markdown)
            )


if __name__ == "__main__":
    unittest.main()
