import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag import RAGError
from server.api import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_is_available_without_api_keys(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_guide_options_are_served_from_the_shared_catalog(self):
        response = self.client.get("/api/guide-options")

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            {"name": "רכב — נזק לרכב"},
            response.json(),
        )

    def test_chat_passes_conversation_context_to_rag(self):
        expected = {
            "answer": "תשובה מבוססת",
            "sources": [{"title": "מקור רשמי", "url": "https://www.fnx.co.il/"}],
        }
        with patch("server.api.get_answer", return_value=expected) as get_answer:
            response = self.client.post(
                "/api/chat",
                json={
                    "question": "מה צריך לצרף?",
                    "history": [
                        {"role": "user", "content": "איך מגישים תביעה?"},
                        {"role": "assistant", "content": "מידע כללי"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        self.assertEqual(
            get_answer.call_args.args[0],
            "מה צריך לצרף?",
        )
        self.assertEqual(
            get_answer.call_args.kwargs["history"][-1],
            {"role": "user", "content": "מה צריך לצרף?"},
        )

    def test_chat_reports_rag_errors_without_success_shaped_response(self):
        with patch(
            "server.api.get_answer",
            side_effect=RAGError("שירות Gemini לא זמין כרגע."),
        ):
            response = self.client.post(
                "/api/chat",
                json={"question": "מה כולל ביטוח רכב?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "שירות Gemini לא זמין כרגע.",
        )

    def test_unknown_guide_option_is_rejected_without_retrieval(self):
        with patch("server.api.get_claim_guide") as get_claim_guide:
            response = self.client.post(
                "/api/claim-guide",
                json={
                    "insurance_type": "https://attacker.example",
                    "question": "מה עושים?",
                },
            )

        self.assertEqual(response.status_code, 422)
        get_claim_guide.assert_not_called()

    def test_claim_guide_uses_only_the_selected_catalog_url(self):
        expected = [
            {
                "title": "תביעת דירה",
                "url": "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/",
                "text": "קטע מהמקור",
                "crawled_at": "2026-10-18",
            }
        ]
        with patch(
            "server.api.get_claim_guide",
            return_value=expected,
        ) as get_claim_guide:
            response = self.client.post(
                "/api/claim-guide",
                json={"insurance_type": "דירה", "question": "מה לצרף?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"chunks": expected})
        self.assertEqual(
            get_claim_guide.call_args.args[0],
            "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/",
        )


if __name__ == "__main__":
    unittest.main()
