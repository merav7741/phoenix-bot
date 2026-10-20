import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rag
from rag import NOT_FOUND, OUT_OF_SCOPE, PERSONAL, RAGError, _parse_answer


class AnswerBoundaryTests(unittest.TestCase):
    def test_personal_question_gets_personal_boundary(self):
        result = _parse_answer('{"status":"personal"}', [])

        self.assertEqual(result, {"answer": PERSONAL, "sources": []})

    def test_out_of_scope_question_gets_relevant_boundary(self):
        result = _parse_answer('{"status":"out_of_scope"}', [])

        self.assertEqual(result, {"answer": OUT_OF_SCOPE, "sources": []})
        self.assertNotIn("פרטים האישיים", result["answer"])

    def test_unanswered_phoenix_question_gets_site_scope_boundary(self):
        result = _parse_answer('{"status":"not_found"}', [])

        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})

    def test_answer_still_requires_a_cited_retrieved_source(self):
        candidates = [
            {
                "url": "https://www.fnx.co.il/apartment-insurance/",
                "title": "ביטוח דירה",
            }
        ]
        response = json.dumps(
            {
                "status": "answer",
                "answer": "תשובה כללית",
                "source_ids": ["1"],
            },
            ensure_ascii=False,
        )

        result = _parse_answer(response, candidates)

        self.assertEqual(result["answer"], "תשובה כללית")
        self.assertEqual(result["sources"][0]["url"], candidates[0]["url"])
        self.assertEqual(result["source_ids"], [1])


class AnswerGroundingTests(unittest.TestCase):
    def setUp(self):
        self.question = "מה צריך לצרף לתביעת ביטוח סיעודי?"
        self.source = {
            "text": "יש לצרף צילום תעודת זהות וטופס ויתור סודיות רפואית.",
            "url": "https://www.fnx.co.il/claims-supportive-insurance/",
            "title": "תביעת ביטוח סיעודי",
        }
        self.gemini = Mock()

    def set_verdict(self, verdict):
        self.gemini.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps({"verdict": verdict})
        )

    def test_supported_answer_is_approved(self):
        self.set_verdict("supported")

        result = rag._verify_answer(
            self.question,
            "יש לצרף צילום תעודת זהות וטופס ויתור סודיות רפואית.",
            [self.source],
            self.gemini,
        )

        self.assertEqual(result, "supported")

    def test_unsupported_answer_is_not_approved(self):
        self.set_verdict("not_found")

        result = rag._verify_answer(
            self.question,
            "יש לצרף גם אישור מיוחד שלא מוזכר במקור.",
            [self.source],
            self.gemini,
        )

        self.assertEqual(result, "not_found")

    def test_verification_fails_closed_on_malformed_response(self):
        self.gemini.models.generate_content.return_value = SimpleNamespace(
            text="not JSON"
        )

        with self.assertRaisesRegex(RAGError, "לא ניתן לאמת"):
            rag._verify_answer(
                self.question,
                "תשובה",
                [self.source],
                self.gemini,
            )

    def test_unrecognized_verdict_is_rejected(self):
        self.set_verdict("probably_supported")

        with self.assertRaisesRegex(RAGError, "פורמט לא צפוי"):
            rag._verify_answer(
                self.question,
                "תשובה",
                [self.source],
                self.gemini,
            )

    def test_end_to_end_rejects_answer_when_claim_is_not_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            collection = Mock()
            collection.count.return_value = 1
            collection.query.return_value = {
                "documents": [[self.source["text"]]],
                "metadatas": [
                    [
                        {
                            "url": self.source["url"],
                            "page_title": self.source["title"],
                        }
                    ]
                ],
                "distances": [[0.2]],
            }
            voyage = Mock()
            voyage.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2]])
            gemini = Mock()
            gemini.models.generate_content.side_effect = [
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "status": "answer",
                            "answer": "החברה מאשרת כל תביעה תוך יום.",
                            "source_ids": [1],
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(text='{"verdict":"not_found"}'),
            ]
            persistent_client = Mock()
            persistent_client.get_collection.return_value = collection

            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(
                    rag.chromadb,
                    "PersistentClient",
                    return_value=persistent_client,
                ),
            ):
                result = rag.get_answer(self.question)

        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})
        self.assertEqual(gemini.models.generate_content.call_count, 2)


class ContextualFollowupTests(unittest.TestCase):
    def test_contextualizes_followup_from_conversation(self):
        gemini = Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "search_query": (
                        "האם ביטוח צד ג בדירה מכסה נזק רטיבות אצל השכן?"
                    ),
                    "clarification_question": None,
                },
                ensure_ascii=False,
            )
        )
        history = [
            {"role": "user", "content": "מה כולל ביטוח צד ג בדירה?"},
            {"role": "assistant", "content": "מידע כללי על הכיסוי."},
        ]

        result = rag._contextualize_question(
            "ומה לגבי זה?",
            history,
            gemini,
        )

        self.assertEqual(
            result,
            {
                "search_query": "האם ביטוח צד ג בדירה מכסה נזק רטיבות אצל השכן?"
            },
        )
        prompt = gemini.models.generate_content.call_args.kwargs["contents"]
        self.assertIn("מה כולל ביטוח צד ג בדירה?", prompt)

    def test_standalone_question_is_preserved_in_conversation(self):
        gemini = Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "search_query": "איך מגישים תביעה על נזקי מים בדירה?",
                    "clarification_question": None,
                },
                ensure_ascii=False,
            )
        )

        result = rag._contextualize_question(
            "איך מגישים תביעה על נזקי מים?",
            [{"role": "user", "content": "איך מגישים תביעה סיעודית?"}],
            gemini,
        )

        self.assertEqual(
            result["search_query"],
            "איך מגישים תביעה על נזקי מים בדירה?",
        )

    def test_ambiguous_question_returns_clarification(self):
        gemini = Mock()
        clarification = "את מתכוונת לתביעה הסיעודית או לנזילה בדירה?"
        gemini.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "search_query": "",
                    "clarification_question": clarification,
                },
                ensure_ascii=False,
            )
        )

        result = rag._contextualize_question(
            "מה לעשות עכשיו?",
            [
                {"role": "user", "content": "איך מגישים תביעה סיעודית?"},
                {"role": "assistant", "content": "כך מגישים תביעה."},
                {"role": "user", "content": "יש גם נזילה בדירה."},
            ],
            gemini,
        )

        self.assertEqual(
            result,
            {"clarification_question": clarification},
        )

    def test_invalid_contextualization_response_is_rejected(self):
        gemini = Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(
            text="not JSON"
        )

        with self.assertRaisesRegex(RAGError, "לא ניתן להבין"):
            rag._contextualize_question(
                "ומה לעשות?",
                [{"role": "user", "content": "יש נזילה בדירה."}],
                gemini,
            )

    def test_clarification_is_returned_before_search(self):
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            collection = Mock()
            collection.count.return_value = 1
            voyage = Mock()
            gemini = Mock()
            clarification = "את מתכוונת לתביעה הסיעודית או לנזילה בדירה?"
            gemini.models.generate_content.return_value = SimpleNamespace(
                text=json.dumps(
                    {
                        "search_query": "",
                        "clarification_question": clarification,
                    },
                    ensure_ascii=False,
                )
            )
            persistent_client = Mock()
            persistent_client.get_collection.return_value = collection
            history = [
                {"role": "user", "content": "איך מגישים תביעה סיעודית?"},
                {"role": "assistant", "content": "כך מגישים תביעה."},
                {"role": "user", "content": "מה לעשות עכשיו?"},
            ]

            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(
                    rag.chromadb,
                    "PersistentClient",
                    return_value=persistent_client,
                ),
            ):
                result = rag.get_answer("מה לעשות עכשיו?", history)

        self.assertEqual(
            result,
            {"answer": clarification, "sources": []},
        )
        voyage.embed.assert_not_called()

    def test_end_to_end_followup_embeds_rewritten_question(self):
        source = {
            "text": "הכיסוי נקבע לפי תנאי הפוליסה.",
            "url": "https://www.fnx.co.il/apartment-insurance/third-party/",
            "title": "ביטוח צד ג בדירה",
        }
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            collection = Mock()
            collection.count.return_value = 1
            collection.query.return_value = {
                "documents": [[source["text"]]],
                "metadatas": [
                    [
                        {
                            "url": source["url"],
                            "page_title": source["title"],
                        }
                    ]
                ],
                "distances": [[0.2]],
            }
            voyage = Mock()
            voyage.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2]])
            gemini = Mock()
            gemini.models.generate_content.side_effect = [
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "search_query": (
                                "האם ביטוח צד ג בדירה מכסה נזק לשכן?"
                            ),
                            "clarification_question": None,
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "status": "answer",
                            "answer": "הכיסוי תלוי בתנאי הפוליסה.",
                            "source_ids": [1],
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(text='{"verdict":"supported"}'),
            ]
            persistent_client = Mock()
            persistent_client.get_collection.return_value = collection

            history = [
                {"role": "user", "content": "מה כולל ביטוח צד ג בדירה?"},
                {"role": "assistant", "content": "מידע כללי על הכיסוי."},
                {"role": "user", "content": "ומה לגבי זה?"},
            ]
            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(
                    rag.chromadb,
                    "PersistentClient",
                    return_value=persistent_client,
                ),
            ):
                result = rag.get_answer("ומה לגבי זה?", history)

        self.assertEqual(result["answer"], "הכיסוי תלוי בתנאי הפוליסה.")
        self.assertEqual(
            voyage.embed.call_args.args[0],
            ["האם ביטוח צד ג בדירה מכסה נזק לשכן?"],
        )
        self.assertEqual(gemini.models.generate_content.call_count, 3)


if __name__ == "__main__":
    unittest.main()
