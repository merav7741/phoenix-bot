import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server import rag
from server.rag import NOT_FOUND, OUT_OF_SCOPE, PERSONAL, RAGError, _parse_answer


class CloudDeploymentCompatibilityTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "temporary_index"):
            self.temporary_index.cleanup()
        rag._load_collection.cache_clear()
        rag._clients.cache_clear()

    def test_cloud_secrets_are_used_when_environment_variables_are_missing(self):
        voyage = Mock()
        gemini = Mock()
        secrets = {
            "VOYAGE_API_KEY": "voyage-test-key",
            "GEMINI_API_KEY": "gemini-test-key",
        }

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(rag, "st", SimpleNamespace(secrets=secrets)),
            patch.object(rag, "load_dotenv"),
            patch.object(rag.voyageai, "Client", return_value=voyage) as voyage_client,
            patch.object(rag.genai, "Client", return_value=gemini) as gemini_client,
        ):
            result = rag._clients()

        self.assertIs(result[0], voyage)
        self.assertIsInstance(result[1], rag._GeminiFailoverClient)
        voyage_client.assert_called_once_with(api_key="voyage-test-key")
        gemini_client.assert_called_once_with(api_key="gemini-test-key")

    def test_multiple_gemini_keys_are_loaded_in_configured_order(self):
        voyage = Mock()
        gemini_clients = [Mock(), Mock(), Mock()]
        secrets = {
            "VOYAGE_API_KEY": "voyage-test-key",
            "GEMINI_API_KEY2": "gemini-secret-key-2",
        }

        with (
            patch.dict(
                "os.environ",
                {
                    "GEMINI_API_KEY": "gemini-env-key",
                    "GEMINI_API_KEY1": "gemini-env-key-1",
                },
                clear=True,
            ),
            patch.object(rag, "st", SimpleNamespace(secrets=secrets)),
            patch.object(rag, "load_dotenv"),
            patch.object(rag.voyageai, "Client", return_value=voyage),
            patch.object(
                rag.genai,
                "Client",
                side_effect=gemini_clients,
            ) as gemini_client,
        ):
            _, failover_client = rag._clients()

        self.assertIsInstance(failover_client, rag._GeminiFailoverClient)
        self.assertEqual(
            [call.kwargs["api_key"] for call in gemini_client.call_args_list],
            ["gemini-env-key", "gemini-env-key-1", "gemini-secret-key-2"],
        )

    def test_chroma_opens_a_temporary_copy_of_the_checked_in_index(self):
        with tempfile.TemporaryDirectory() as temp:
            index_path = Path(temp) / "chroma_voyage"
            index_path.mkdir()
            database = index_path / "chroma.sqlite3"
            database.write_bytes(b"original-index")
            expected_collection = Mock()
            client = Mock()
            client.get_collection.return_value = expected_collection

            with patch.object(
                rag.chromadb,
                "PersistentClient",
                return_value=client,
            ) as persistent_client:
                temporary_index, collection = rag._load_collection(str(index_path))
                self.temporary_index = temporary_index

                runtime_path = Path(persistent_client.call_args.kwargs["path"])
                runtime_database = runtime_path / "chroma.sqlite3"
                self.assertEqual(runtime_database.read_bytes(), b"original-index")
                runtime_database.write_bytes(b"runtime-write")

            self.assertIs(collection, expected_collection)
            self.assertEqual(database.read_bytes(), b"original-index")


class ClaimGuideTests(unittest.TestCase):
    def test_claim_guide_reads_only_the_selected_official_source_without_api_clients(self):
        url = "https://www.fnx.co.il/claims-customer-service/claims-health-insurance/"
        collection = Mock()
        collection.get.side_effect = [
            {
                "ids": ["guide-2", "guide-0", "other-0", "guide-1"],
                "metadatas": [
                    {"url": f"{url}surgeries/"},
                    {"url": url},
                    {"url": "https://www.fnx.co.il/claims-customer-service/claims-pets/"},
                    {"url": f"{url}surgeries/"},
                ],
            },
            {
                "ids": ["guide-2", "guide-0", "guide-1"],
                "documents": ["קטע שני", "קטע ראשון", "קטע ביניים"],
                "metadatas": [
                    {
                        "url": f"{url}surgeries/",
                        "header": "כותרת",
                        "crawled_at": "2026-10-18",
                    },
                    {"url": url, "header": "כותרת", "crawled_at": "2026-10-18"},
                    {
                        "url": f"{url}surgeries/",
                        "header": "כותרת",
                        "crawled_at": "2026-10-18",
                    },
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as temp:
            index_path = Path(temp)
            (index_path / "chroma.sqlite3").touch()
            with (
                patch.object(rag, "DB_DIR", index_path),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
                patch.object(
                    rag,
                    "_clients",
                    side_effect=AssertionError("The guide must not call external APIs."),
                ),
            ):
                guides = rag.get_claim_guide(url)

        self.assertEqual(
            [chunk["text"] for chunk in guides],
            ["קטע ראשון", "קטע ביניים", "קטע שני"],
        )
        self.assertEqual(
            [chunk["url"] for chunk in guides],
            [url, f"{url}surgeries/", f"{url}surgeries/"],
        )
        self.assertEqual(
            collection.get.call_args_list[0].kwargs,
            {"include": ["metadatas"]},
        )
        self.assertEqual(
            set(collection.get.call_args_list[1].kwargs["ids"]),
            {"guide-0", "guide-1", "guide-2"},
        )
        self.assertEqual(
            collection.get.call_args_list[1].kwargs["include"],
            ["documents", "metadatas"],
        )

    def test_claim_guide_ranks_source_chunks_by_question_terms(self):
        url = "https://www.fnx.co.il/claims-customer-service/claims-health-insurance/"
        collection = Mock()
        collection.get.side_effect = [
            {
                "ids": ["relevant-0", "unrelated-0"],
                "metadatas": [{"url": url}, {"url": url}],
            },
            {
                "ids": ["relevant-0", "unrelated-0"],
                "documents": [
                    "הגשת תביעה לאחר ניתוח בבית חולים",
                    "מידע כללי על שירותי בריאות",
                ],
                "metadatas": [
                    {"url": url, "header": "תביעה"},
                    {"url": url, "header": "שירותי בריאות"},
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as temp:
            index_path = Path(temp)
            (index_path / "chroma.sqlite3").touch()
            with (
                patch.object(rag, "DB_DIR", index_path),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                guides = rag.get_claim_guide(
                    url,
                    question="תביעה לאחר ניתוח",
                )

        self.assertEqual(len(guides), 1)
        self.assertEqual(guides[0]["text"], "הגשת תביעה לאחר ניתוח בבית חולים")

    def test_claim_guide_rejects_non_claims_urls(self):
        with patch.object(rag, "_load_collection") as load_collection:
            with self.assertRaises(ValueError):
                rag.get_claim_guide("https://example.com/claims-customer-service/")
            with self.assertRaises(ValueError):
                rag.get_claim_guide("https://www.fnx.co.il/health-insurance/")

        load_collection.assert_not_called()

    def test_claim_guide_returns_empty_when_the_selected_page_is_not_indexed(self):
        url = "https://www.fnx.co.il/claims-customer-service/claims-health-insurance/"
        collection = Mock()
        collection.get.return_value = {"ids": [], "metadatas": []}

        with tempfile.TemporaryDirectory() as temp:
            index_path = Path(temp)
            (index_path / "chroma.sqlite3").touch()
            with (
                patch.object(rag, "DB_DIR", index_path),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                self.assertEqual(rag.get_claim_guide(url), [])

        collection.get.assert_called_once_with(include=["metadatas"])


class GeminiApiFailureTests(unittest.TestCase):
    def test_invalid_key_falls_back_to_next_configured_key(self):
        first_client = Mock()
        first_client.models.generate_content.side_effect = rag.genai.errors.APIError(
            400,
            {"error": {"status": "API_KEY_INVALID", "message": "invalid key"}},
        )
        second_client = Mock()
        expected_response = object()
        second_client.models.generate_content.return_value = expected_response
        failover_client = rag._GeminiFailoverClient(
            [first_client, second_client]
        )

        result = failover_client.models.generate_content(
            model=rag.ANSWER_MODEL,
            contents="test",
        )

        self.assertIs(result, expected_response)
        first_client.models.generate_content.assert_called_once()
        second_client.models.generate_content.assert_called_once()

    def test_any_api_error_falls_back_to_next_configured_key(self):
        first_client = Mock()
        first_client.models.generate_content.side_effect = rag.genai.errors.APIError(
            404,
            {"error": {"status": "NOT_FOUND", "message": "model unavailable"}},
        )
        second_client = Mock()
        expected_response = object()
        second_client.models.generate_content.return_value = expected_response
        failover_client = rag._GeminiFailoverClient(
            [first_client, second_client]
        )

        result = failover_client.models.generate_content(
            model=rag.ANSWER_MODEL,
            contents="test",
        )

        self.assertIs(result, expected_response)
        second_client.models.generate_content.assert_called_once_with(
            model=rag.ANSWER_MODEL, contents="test"
        )

    def test_all_keys_failing_tries_fallback_model_then_raises_last_error(self):
        unavailable = rag.genai.errors.APIError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": "overloaded"}},
        )
        first_client = Mock()
        first_client.models.generate_content.side_effect = unavailable
        second_client = Mock()
        second_client.models.generate_content.side_effect = unavailable
        failover_client = rag._GeminiFailoverClient(
            [first_client, second_client]
        )

        with self.assertRaises(rag.genai.errors.APIError):
            failover_client.models.generate_content(
                model=rag.ANSWER_MODEL,
                contents="test",
            )

        models_tried = [
            call.kwargs["model"]
            for call in first_client.models.generate_content.call_args_list
        ]
        self.assertEqual(
            models_tried, [rag.ANSWER_MODEL, *rag.FALLBACK_ANSWER_MODELS]
        )
        self.assertEqual(
            second_client.models.generate_content.call_count, len(models_tried)
        )

    def test_quota_error_falls_back_to_next_configured_key(self):
        first_client = Mock()
        first_client.models.generate_content.side_effect = rag.genai.errors.APIError(
            429,
            {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}},
        )
        second_client = Mock()
        expected_response = object()
        second_client.models.generate_content.return_value = expected_response
        failover_client = rag._GeminiFailoverClient(
            [first_client, second_client]
        )

        result = failover_client.models.generate_content(
            model=rag.ANSWER_MODEL,
            contents="test",
        )

        self.assertIs(result, expected_response)
        first_client.models.generate_content.assert_called_once()
        second_client.models.generate_content.assert_called_once()

    def test_quota_error_does_not_misreport_context_as_unclear(self):
        gemini = Mock()
        gemini.models.generate_content.side_effect = rag.genai.errors.APIError(
            429,
            {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}},
        )

        with self.assertRaises(RAGError) as error:
            rag._contextualize_question(
                "ומה לגבי זה?",
                [{"role": "user", "content": "מה כולל ביטוח דירה?"}],
                gemini,
            )

        self.assertIn("מכסת Gemini", str(error.exception))
        self.assertNotIn("נסחי", str(error.exception))

    def test_invalid_key_error_points_to_secrets_not_quota(self):
        error = rag.genai.errors.APIError(
            403,
            {"error": {"status": "PERMISSION_DENIED", "message": "invalid key"}},
        )

        message = rag._gemini_api_error_message(error)

        self.assertIn("מפתח Gemini לא תקין", message)
        self.assertNotIn("מכסת", message)

    def test_other_gemini_api_errors_are_reported_as_service_unavailable(self):
        error = rag.genai.errors.APIError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": "unavailable"}},
        )

        message = rag._gemini_api_error_message(error)

        self.assertIn("שירות Gemini לא זמין", message)


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

    def test_not_found_response_is_clear_and_not_a_scope_error(self):
        self.assertIn("לא מצאתי", NOT_FOUND)
        self.assertNotIn("אני עונה רק", NOT_FOUND)

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
            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                result = rag.get_answer(self.question)

        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})
        self.assertEqual(gemini.models.generate_content.call_count, 2)


class ContextualFollowupTests(unittest.TestCase):
    def test_typos_in_first_question_are_corrected_after_weak_search(self):
        source = {
            "text": "יש לפנות למוקד תביעות נזקי רכוש בדירה.",
            "url": (
                "https://www.fnx.co.il/claims-customer-service/"
                "claims-appartment-insurance/damage-to-the-apartment/"
            ),
            "title": "תביעת נזק לדירה",
        }
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            weak_result = {
                "documents": [["תוכן לא קשור."]],
                "metadatas": [
                    [{"url": source["url"], "page_title": source["title"]}]
                ],
                "distances": [[0.9]],
            }
            relevant_result = {
                "documents": [[source["text"]]],
                "metadatas": [
                    [{"url": source["url"], "page_title": source["title"]}]
                ],
                "distances": [[0.2]],
            }
            collection = Mock()
            collection.count.return_value = 1
            collection.query.side_effect = [weak_result, relevant_result]
            voyage = Mock()
            voyage.embed.side_effect = [
                SimpleNamespace(embeddings=[[0.1, 0.2]]),
                SimpleNamespace(embeddings=[[0.3, 0.4]]),
            ]
            gemini = Mock()
            gemini.models.generate_content.side_effect = [
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "search_query": (
                                "מה עושים במקרה של נזילה בקירות הבית?"
                            ),
                            "uses_conversation_context": False,
                            "clarification_question": None,
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(
                    text=json.dumps(
                        {
                            "status": "answer",
                            "answer": "יש לפנות למוקד תביעות נזקי רכוש בדירה.",
                            "source_ids": [1],
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(text='{"verdict":"supported"}'),
            ]
            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                result = rag.get_answer("מה עושים אם יש נזילה בקירם?")

        self.assertEqual(result["answer"], "יש לפנות למוקד תביעות נזקי רכוש בדירה.")
        self.assertEqual(result["sources"][0]["url"], source["url"])
        self.assertEqual(
            voyage.embed.call_args_list[1].args[0],
            ["מה עושים במקרה של נזילה בקירות הבית?"],
        )
        contextualization_instruction = (
            gemini.models.generate_content.call_args_list[0]
            .kwargs["config"]
            .system_instruction
        )
        self.assertIn("עברית מדוברת", contextualization_instruction)
        self.assertIn("אל תשני שמות, מספרי טלפון", contextualization_instruction)
        self.assertIn("אין להפוך שאלה כללית", contextualization_instruction)
        self.assertEqual(collection.query.call_count, 2)

    def test_successful_first_search_does_not_use_extra_gemini_call(self):
        source = {
            "text": "המידע הרלוונטי נמצא בעמוד.",
            "url": "https://www.fnx.co.il/apartment-insurance/",
            "title": "ביטוח דירה",
        }
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()
            collection = Mock()
            collection.count.return_value = 1
            collection.query.return_value = {
                "documents": [[source["text"]]],
                "metadatas": [
                    [{"url": source["url"], "page_title": source["title"]}]
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
                            "answer": "המידע הרלוונטי נמצא בעמוד.",
                            "source_ids": [1],
                        },
                        ensure_ascii=False,
                    )
                ),
                SimpleNamespace(text='{"verdict":"supported"}'),
            ]
            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                result = rag.get_answer("מה כולל ביטוח דירה?")

        self.assertEqual(result["answer"], "המידע הרלוונטי נמצא בעמוד.")
        self.assertEqual(gemini.models.generate_content.call_count, 2)
        self.assertEqual(voyage.embed.call_count, 1)
        answer_instruction = (
            gemini.models.generate_content.call_args_list[0]
            .kwargs["config"]
            .system_instruction
        )
        self.assertIn("בעברית טבעית, ברורה", answer_instruction)
        self.assertIn("אל תציגי השערה כעובדה", answer_instruction)
        self.assertIn("בלי להמציא פרטי קשר", answer_instruction)

    def test_unanswered_contextual_followup_is_human_and_reuses_valid_source(self):
        previous_source = {
            "url": (
                "https://www.fnx.co.il/claims-customer-service/"
                "claims-appartment-insurance/damage-to-the-apartment/"
            ),
            "title": "תביעה בגין נזק בדירה",
        }
        result = rag._not_found_result(
            [
                {
                    "role": "assistant",
                    "content": "מידע על הגשת תביעה.",
                    "sources": [previous_source],
                }
            ],
            uses_context=True,
        )

        self.assertIn("לא מצאתי באתר הנחיות מאומתות", result["answer"])
        self.assertIn("לא רוצה לנחש", result["answer"])
        self.assertEqual(result["sources"], [previous_source])

    def test_unrelated_question_does_not_reuse_previous_source(self):
        previous_source = {
            "url": "https://www.fnx.co.il/apartment-insurance/",
            "title": "ביטוח דירה",
        }
        result = rag._not_found_result(
            [
                {
                    "role": "assistant",
                    "content": "מידע על ביטוח דירה.",
                    "sources": [previous_source],
                }
            ],
            uses_context=False,
        )

        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})

    def test_invalid_previous_source_is_not_reused(self):
        result = rag._not_found_result(
            [
                {
                    "role": "assistant",
                    "content": "תשובה",
                    "sources": [
                        {
                            "url": "https://example.com/",
                            "title": "מקור חיצוני",
                        }
                    ],
                }
            ],
            uses_context=True,
        )

        self.assertEqual(result, {"answer": NOT_FOUND, "sources": []})

    def test_contextualizes_followup_from_conversation(self):
        gemini = Mock()
        gemini.models.generate_content.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "search_query": (
                        "האם ביטוח צד ג בדירה מכסה נזק רטיבות אצל השכן?"
                    ),
                    "uses_conversation_context": True,
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
                "search_query": "האם ביטוח צד ג בדירה מכסה נזק רטיבות אצל השכן?",
                "uses_conversation_context": True,
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
                    "uses_conversation_context": False,
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
        clarification = (
            "את מתכוונת למה לעשות כשהנזילה עדיין פעילה, או בזמן "
            "שממתינים לטיפול בתביעה?"
        )
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
            "מה עושים בזמן הזה??",
            [
                {
                    "role": "user",
                    "content": "יש לי נזילה בבית, איך מגישים תביעה?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "פני למרכז התביעות, מלאי טופס, צרפי תמונות וחשבונית "
                        "או הצעת מחיר ושלחי את המסמכים."
                    ),
                },
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

    def test_leak_claim_followup_clarifies_before_search(self):
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            collection = Mock()
            collection.count.return_value = 1
            voyage = Mock()
            gemini = Mock()
            clarification = (
                "את מתכוונת למה לעשות כשהנזילה עדיין פעילה, או בזמן "
                "שממתינים לטיפול בתביעה?"
            )
            gemini.models.generate_content.return_value = SimpleNamespace(
                text=json.dumps(
                    {
                        "search_query": "",
                        "clarification_question": clarification,
                    },
                    ensure_ascii=False,
                )
            )
            history = [
                {
                    "role": "user",
                    "content": "יש לי נזילה בבית, איך מגישים תביעה?",
                },
                {
                    "role": "assistant",
                    "content": (
                        "פני למרכז התביעות, מלאי טופס, צרפי תמונות וחשבונית "
                        "או הצעת מחיר ושלחי את המסמכים."
                    ),
                },
                {"role": "user", "content": "מה עושים בזמן הזה??"},
            ]

            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                result = rag.get_answer("מה עושים בזמן הזה??", history)

        self.assertEqual(
            result,
            {"answer": clarification, "sources": []},
        )
        voyage.embed.assert_not_called()

    def test_unretrieved_leak_followup_returns_contextual_response_and_source(self):
        prior_source = {
            "url": (
                "https://www.fnx.co.il/claims-customer-service/"
                "claims-appartment-insurance/damage-to-the-apartment/"
            ),
            "title": "תביעה בגין נזק בדירה",
        }
        with tempfile.TemporaryDirectory() as temp:
            db_dir = Path(temp)
            (db_dir / "chroma.sqlite3").touch()

            collection = Mock()
            collection.count.return_value = 1
            collection.query.return_value = {
                "documents": [["קטע לא קשור מתוך האתר."]],
                "metadatas": [
                    [{"url": prior_source["url"], "page_title": prior_source["title"]}]
                ],
                "distances": [[0.9]],
            }
            voyage = Mock()
            voyage.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2]])
            gemini = Mock()
            gemini.models.generate_content.return_value = SimpleNamespace(
                text=json.dumps(
                    {
                        "search_query": "מה לעשות כשהמים נוזלים בקירות?",
                        "uses_conversation_context": True,
                        "clarification_question": None,
                    },
                    ensure_ascii=False,
                )
            )
            history = [
                {
                    "role": "user",
                    "content": "יש לי נזילה בבית, איך מגישים תביעה?",
                },
                {
                    "role": "assistant",
                    "content": "כך מגישים תביעה.",
                    "sources": [prior_source],
                },
                {"role": "user", "content": "מה עושים בזמן הזה?"},
                {
                    "role": "assistant",
                    "content": "האם הכוונה בזמן הנזילה או בהמתנה לתביעה?",
                },
                {"role": "user", "content": "בזמן שנוזל לי מים בקירות"},
            ]

            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
            ):
                result = rag.get_answer(
                    "בזמן שנוזל לי מים בקירות",
                    history,
                )

        self.assertIn("לא מצאתי באתר הנחיות מאומתות", result["answer"])
        self.assertEqual(result["sources"], [prior_source])
        self.assertEqual(gemini.models.generate_content.call_count, 1)

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
                            "uses_conversation_context": True,
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
            history = [
                {"role": "user", "content": "מה כולל ביטוח צד ג בדירה?"},
                {"role": "assistant", "content": "מידע כללי על הכיסוי."},
                {"role": "user", "content": "ומה לגבי זה?"},
            ]
            with (
                patch.object(rag, "DB_DIR", db_dir),
                patch.object(rag, "_clients", return_value=(voyage, gemini)),
                patch.object(rag, "_load_collection", return_value=(None, collection)),
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
