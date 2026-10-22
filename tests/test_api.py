import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.api import FRONTEND_DIST, app
from server.rag import RAGError


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_is_available_without_api_keys(self):
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_production_angular_page_and_bundle_are_served(self):
        index_file = FRONTEND_DIST / "index.html"
        if not index_file.is_file():
            self.skipTest("Build the Angular frontend to test static serving.")

        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<app-root>", response.text)
        bundle_match = re.search(r'src="([^"]+\.js)"', response.text)
        self.assertIsNotNone(bundle_match)
        bundle_path = bundle_match.group(1)
        bundle = self.client.get(bundle_path)
        self.assertEqual(bundle.status_code, 200)
        self.assertIn("javascript", bundle.headers["content-type"])

    def test_missing_frontend_asset_returns_not_found(self):
        if not (FRONTEND_DIST / "index.html").is_file():
            self.skipTest("Build the Angular frontend to test static serving.")

        response = self.client.get("/missing-bundle.js")

        self.assertEqual(response.status_code, 404)

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
        with patch("server.api.get_claim_guide_answer") as get_claim_guide_answer:
            response = self.client.post(
                "/api/claim-guide",
                json={
                    "insurance_type": "https://attacker.example",
                    "question": "מה עושים?",
                },
            )

        self.assertEqual(response.status_code, 422)
        get_claim_guide_answer.assert_not_called()

    def test_claim_guide_uses_only_the_selected_catalog_url(self):
        expected = {
            "answer": "כדאי להתחיל באיסוף המסמכים שמופיעים בעמוד.",
            "sources": [
                {
                    "title": "תביעת דירה",
                    "url": "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/",
                }
            ],
        }
        with patch(
            "server.api.get_claim_guide_answer",
            return_value=expected,
        ) as get_claim_guide_answer:
            response = self.client.post(
                "/api/claim-guide",
                json={"insurance_type": "דירה", "question": "מה לצרף?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        self.assertEqual(
            get_claim_guide_answer.call_args.args[0],
            "https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/",
        )


if __name__ == "__main__":
    unittest.main()
