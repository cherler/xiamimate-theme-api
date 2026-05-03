from __future__ import annotations

import contextlib
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from data_platform.api.product_theme_api import (
    CandidateRecord,
    CategoryBenchmarkRequest,
    CandidatePoolRequest,
    CandidateExpansionJobRequest,
    CandidateExpansionJobStatusRequest,
    OpportunityDiscoveryRequest,
    ProductForecastExplainRequest,
    ProductThemeService,
    ResolveCandidatesRequest,
    _build_query_variants,
    _build_candidate_pool_quality,
    _candidate_field_matches_required_terms,
    _category_distribution,
    _category_level_name,
    _fine_category_name,
    _leaf_category_name,
    _score_candidate,
    _score_category_match,
)
from data_platform.api.seller_scope import evaluate_seller_scope


def make_record(
    *,
    title: str,
    category: str = "",
    category_path: str = "",
    search_term: str = "",
    keywords: list[str] | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        asin="B000000001",
        domain=1,
        marketplace="US",
        product_title=title,
        brand="TestBrand",
        category=category,
        category_path=category_path,
        search_term=search_term,
        business_priority=0,
        business_tier="",
        is_active=True,
        current_price=None,
        current_rating=None,
        current_review_count=None,
        current_bsr=None,
        current_offer_count=None,
        history_rows_30d=0,
        has_sales_signal_30d=False,
        has_price_data_30d=False,
        latest_history_date=None,
        keywords=keywords or [],
    )


def score_for_query(query: str, record: CandidateRecord) -> tuple[float, list[str], dict]:
    phrases, tokens, _ = _build_query_variants(query, [], [])
    return _score_candidate(record, phrases, tokens)


class CandidateSemanticRecallTests(unittest.TestCase):
    def test_multi_token_query_requires_product_core_term(self) -> None:
        fan = make_record(
            title="Portable rechargeable desk fan",
            category="Fans",
            category_path="Home & Kitchen > Heating, Cooling & Air Quality > Fans",
            search_term="portable fan",
            keywords=["portable fan", "desk fan"],
        )

        score, reasons, breakdown = score_for_query("portable blender", fan)

        self.assertEqual(score, 0.0)
        self.assertNotIn("required_product_term_match", reasons)
        self.assertEqual(breakdown["required_product_terms"], ["blender"])
        self.assertEqual(breakdown["matched_required_product_terms"], [])

    def test_multi_token_query_keeps_candidate_with_core_term(self) -> None:
        blender = make_record(
            title="Portable blender for shakes and smoothies",
            category="Countertop Blenders",
            category_path="Home & Kitchen > Kitchen & Dining > Small Appliances > Blenders",
            search_term="portable blender",
            keywords=["portable blender", "smoothie blender"],
        )

        score, reasons, breakdown = score_for_query("portable blender", blender)

        self.assertGreater(score, 0.0)
        self.assertIn("required_product_term_match", reasons)
        self.assertEqual(breakdown["required_product_terms"], ["blender"])
        self.assertEqual(breakdown["matched_required_product_terms"], ["blender"])

    def test_multi_token_query_rejects_title_only_ambiguous_core_word(self) -> None:
        makeup_set = make_record(
            title="Makeup Brushes Set with Blender Sponges and Portable Makeup Bag",
            category="",
            category_path="Beauty & Personal Care > Tools & Accessories > Makeup Brushes & Tools > Brush Sets",
            search_term="makeup brushes",
            keywords=["makeup brushes", "foundation concealer blush bronzer"],
        )

        score, reasons, breakdown = score_for_query("portable blender", makeup_set)

        self.assertEqual(score, 0.0)
        self.assertIn("required_product_term_match", reasons)
        self.assertNotIn("required_product_term_semantic_field_match", reasons)
        self.assertFalse(breakdown["compound_phrase_match"])

    def test_leaf_category_is_available_when_l3_is_too_coarse(self) -> None:
        category_path = "Health & Household > Health Care > Alternative Medicine > Aromatherapy > Humidifiers"
        humidifier = make_record(
            title="Cool mist humidifier",
            category="Alternative Medicine",
            category_path=category_path,
            search_term="humidifier",
            keywords=["humidifier"],
        )

        score, _, breakdown = score_for_query("humidifier", humidifier)

        self.assertGreater(score, 0.0)
        self.assertEqual(_category_level_name(category_path, 2), "Alternative Medicine")
        self.assertEqual(_leaf_category_name(category_path), "Humidifiers")
        self.assertEqual(_fine_category_name(category_path), "Humidifiers")
        self.assertEqual(breakdown["matched_required_product_terms"], ["humidifier"])

    def test_fine_category_anchor_ignores_keyword_only_adjacent_leaf(self) -> None:
        diffuser_item = {
            "leaf_category_name": "Diffusers",
            "fine_category_name": "Diffusers",
            "keywords": ["cool mist humidifier"],
        }
        humidifier_item = {
            "leaf_category_name": "Humidifiers",
            "fine_category_name": "Humidifiers",
            "keywords": [],
        }

        self.assertFalse(
            _candidate_field_matches_required_terms(
                diffuser_item,
                ["leaf_category_name", "fine_category_name"],
                ["humidifier"],
            )
        )
        self.assertTrue(
            _candidate_field_matches_required_terms(
                humidifier_item,
                ["leaf_category_name", "fine_category_name"],
                ["humidifier"],
            )
        )

    def test_pool_quality_flags_small_pure_pool_for_expansion(self) -> None:
        items = [
            {"leaf_category_name": "Humidifiers", "fine_category_name": "Humidifiers", "root_category_name": "Home & Kitchen"},
            {"leaf_category_name": "Humidifiers", "fine_category_name": "Humidifiers", "root_category_name": "Home & Kitchen"},
        ]

        quality = _build_candidate_pool_quality(
            items,
            candidate_total_before_semantic_gate=35,
            candidate_total_before_category_anchor=35,
            leaf_distribution=_category_distribution(items, "leaf_category_name"),
            fine_distribution=_category_distribution(items, "fine_category_name"),
            root_distribution=_category_distribution(items, "root_category_name"),
        )

        self.assertFalse(quality["is_sufficient_for_analysis"])
        self.assertTrue(quality["should_expand_pool"])
        self.assertEqual(quality["insufficient_coverage_reason"], "pure_candidate_count_below_min_pool_size")
        self.assertEqual(quality["dominant_leaf_category"], "Humidifiers")
        self.assertEqual(quality["dominant_leaf_share"], 1.0)

    def test_pool_quality_accepts_sufficient_pure_pool_but_can_still_expand_to_target(self) -> None:
        items = [
            {"leaf_category_name": "Humidifiers", "fine_category_name": "Humidifiers", "root_category_name": "Home & Kitchen"}
            for _ in range(8)
        ]

        quality = _build_candidate_pool_quality(
            items,
            candidate_total_before_semantic_gate=12,
            candidate_total_before_category_anchor=10,
            leaf_distribution=_category_distribution(items, "leaf_category_name"),
            fine_distribution=_category_distribution(items, "fine_category_name"),
            root_distribution=_category_distribution(items, "root_category_name"),
        )

        self.assertTrue(quality["is_sufficient_for_analysis"])
        self.assertTrue(quality["should_expand_pool"])
        self.assertIsNone(quality["insufficient_coverage_reason"])
        self.assertEqual(quality["category_anchor_confidence"], "medium")

    def test_pool_quality_rejects_scattered_category_pool(self) -> None:
        items = [
            {"leaf_category_name": f"Leaf{i}", "fine_category_name": f"Leaf{i}", "root_category_name": "Home & Kitchen"}
            for i in range(10)
        ]

        quality = _build_candidate_pool_quality(
            items,
            candidate_total_before_semantic_gate=20,
            candidate_total_before_category_anchor=10,
            leaf_distribution=_category_distribution(items, "leaf_category_name"),
            fine_distribution=_category_distribution(items, "fine_category_name"),
            root_distribution=_category_distribution(items, "root_category_name"),
        )

        self.assertFalse(quality["is_sufficient_for_analysis"])
        self.assertTrue(quality["should_expand_pool"])
        self.assertEqual(quality["insufficient_coverage_reason"], "dominant_category_share_below_threshold")
        self.assertLess(quality["dominant_leaf_share"], 0.7)

    def test_category_match_prefers_exact_full_path(self) -> None:
        score = _score_category_match(
            category_query="humidifier",
            category_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
            candidate_name="Humidifiers",
            candidate_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
        )

        self.assertEqual(score, 0.98)

    def test_category_match_accepts_path_without_query(self) -> None:
        score = _score_category_match(
            category_query=None,
            category_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
            candidate_name="Humidifiers",
            candidate_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
        )

        self.assertEqual(score, 0.98)

    def test_category_match_accepts_leaf_query(self) -> None:
        score = _score_category_match(
            category_query="Humidifiers",
            category_path=None,
            candidate_name="Humidifiers",
            candidate_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
        )

        self.assertEqual(score, 0.92)

    def test_category_match_rejects_unrelated_category(self) -> None:
        score = _score_category_match(
            category_query="humidifier",
            category_path="Home & Kitchen > Heating, Cooling & Air Quality > Humidifiers",
            candidate_name="Diffusers",
            candidate_path="Health & Household > Alternative Medicine > Aromatherapy > Diffusers",
        )

        self.assertEqual(score, 0.0)

    def test_resolve_candidates_request_accepts_category_recall_controls(self) -> None:
        request = ResolveCandidatesRequest(
            product_query="humidifier",
            recall_mode="HYBRID",
            category_id=12345,
            category_path="  Home & Kitchen > Humidifiers  ",
            include_descendants=True,
            min_pool_size=6,
            target_pool_size=18,
            expand_if_small=True,
        )

        self.assertEqual(request.recall_mode, "hybrid")
        self.assertEqual(request.category_id, 12345)
        self.assertEqual(request.category_path, "Home & Kitchen > Humidifiers")
        self.assertEqual(request.min_pool_size, 6)
        self.assertEqual(request.target_pool_size, 18)
        self.assertTrue(request.expand_if_small)

    def test_resolve_candidates_request_rejects_unknown_recall_mode(self) -> None:
        with self.assertRaises(ValueError):
            ResolveCandidatesRequest(product_query="humidifier", recall_mode="unknown")

    def test_category_benchmark_request_accepts_explicit_anchor(self) -> None:
        request = CategoryBenchmarkRequest(
            candidate_asins=["B000000001"],
            benchmark_category_id=12345,
            benchmark_category_path="  Home & Kitchen > Humidifiers  ",
            benchmark_level="L3",
            include_descendants=False,
        )

        self.assertEqual(request.benchmark_category_id, 12345)
        self.assertEqual(request.benchmark_category_path, "Home & Kitchen > Humidifiers")
        self.assertEqual(request.benchmark_level, "l3")
        self.assertFalse(request.include_descendants)

    def test_category_benchmark_request_rejects_unknown_level(self) -> None:
        with self.assertRaises(ValueError):
            CategoryBenchmarkRequest(candidate_asins=["B000000001"], benchmark_level="department")

    def test_candidate_pool_request_accepts_candidate_pool_id_without_asins(self) -> None:
        request = CandidatePoolRequest(
            candidate_pool_id="11111111-1111-4111-8111-111111111111",
            marketplace="US",
        )

        self.assertEqual(request.candidate_asins, [])
        self.assertEqual(request.candidate_pool_id, "11111111-1111-4111-8111-111111111111")

    def test_candidate_pool_lineage_records_query_category_and_filters(self) -> None:
        service = ProductThemeService()
        request = ResolveCandidatesRequest(
            product_query="humidifier",
            recall_mode="hybrid",
            category_id=12345,
            category_path="Home & Kitchen > Humidifiers",
            max_candidates=20,
        )

        lineage = service._build_candidate_pool_lineage(
            request=request,
            marketplace="US",
            domain=1,
            normalized_query="humidifier",
            normalized_phrases=["humidifier"],
            tokens=["humidifier"],
            query_expansions=[],
        )

        self.assertEqual(lineage["source"], "resolve_candidates")
        self.assertEqual(lineage["ranking_version"], "semantic_recall_v2")
        self.assertEqual(lineage["query"]["normalized_query"], "humidifier")
        self.assertEqual(lineage["category"]["category_id"], 12345)
        self.assertEqual(lineage["filters"]["recall_mode"], "hybrid")

    def test_candidate_expansion_job_request_normalizes_controls(self) -> None:
        request = CandidateExpansionJobRequest(
            product_query="humidifier",
            recall_mode="CATEGORY",
            category_id=12345,
            category_path="  Home & Kitchen > Humidifiers  ",
            target_asin_count=20,
            priority="INTERACTIVE_HIGH",
            idempotency_key="  req-1  ",
        )

        self.assertEqual(request.recall_mode, "category")
        self.assertEqual(request.priority, "interactive_high")
        self.assertEqual(request.category_path, "Home & Kitchen > Humidifiers")
        self.assertEqual(request.idempotency_key, "req-1")

    def test_candidate_expansion_rejects_category_path_without_id(self) -> None:
        service = ProductThemeService()
        request = CandidateExpansionJobRequest(
            product_query="humidifier",
            recall_mode="hybrid",
            category_path="Home & Kitchen > Humidifiers",
        )

        with self.assertRaises(HTTPException) as context:
            service.create_candidate_expansion_job(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("category_id", str(context.exception.detail))

    def test_candidate_expansion_rejects_category_recall_without_id(self) -> None:
        service = ProductThemeService()
        request = CandidateExpansionJobRequest(product_query="humidifier", recall_mode="category")

        with self.assertRaises(HTTPException) as context:
            service.create_candidate_expansion_job(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("category_id", str(context.exception.detail))

    def test_candidate_expansion_category_lock_key_is_stable(self) -> None:
        service = ProductThemeService()

        lock_key = service._candidate_expansion_category_lock_key(
            domain=1,
            category_id=12345,
            include_descendants=True,
        )

        self.assertEqual(lock_key, "candidate-expansion:domain:1:category:12345:desc:1")

    def test_candidate_expansion_reuses_active_category_job(self) -> None:
        service = ProductThemeService()
        request = CandidateExpansionJobRequest(product_query="humidifier", category_id=12345)
        queries: list[str] = []

        existing_row = {
            "job_id": "kexp_existing",
            "marketplace": "US",
            "domain": 1,
            "source": "agent_interactive",
            "priority": "interactive_normal",
            "product_query": "humidifier",
            "recall_mode": "hybrid",
            "category_id": 12345,
            "category_path": None,
            "include_descendants": True,
            "target_asin_count": 20,
            "min_pool_size": 8,
            "status": "hydrating",
            "status_reason": "existing active job",
            "requested_by_session_id": None,
            "requested_by_user_id": None,
            "tokens_estimated": 90,
            "tokens_reserved": 0,
            "tokens_consumed": 0,
            "token_wait_until": None,
            "result_candidate_asins": [],
            "result_new_asin_count": 0,
            "error_message": None,
            "created_at": None,
            "updated_at": None,
            "started_at": None,
            "finished_at": None,
            "meta_json": {},
        }

        def fake_query(_conn: object, sql: str, _params: list[object] | None = None) -> list[dict]:
            queries.append(sql)
            if "pg_advisory_lock" in sql or "pg_advisory_unlock" in sql:
                return [{}]
            if "status = ANY" in sql:
                return [existing_row]
            return []

        with patch("data_platform.api.product_theme_api._postgres_conn", return_value=contextlib.nullcontext(object())):
            with patch("data_platform.api.product_theme_api._run_pg_dict_query", side_effect=fake_query):
                result = service.create_candidate_expansion_job(request)

        self.assertFalse(result["created"])
        self.assertEqual(result["job"]["job_id"], "kexp_existing")
        self.assertFalse(any("INSERT INTO sync.keepa_candidate_expansion_jobs" in sql for sql in queries))

    def test_candidate_expansion_status_request_accepts_csv_statuses(self) -> None:
        request = CandidateExpansionJobStatusRequest(statuses="queued, waiting_token", limit=5)

        self.assertEqual(request.statuses, ["queued", "waiting_token"])
        self.assertEqual(request.limit, 5)

    def test_candidate_expansion_readiness_distinguishes_serving_pending(self) -> None:
        service = ProductThemeService()
        job = {
            "result_candidate_asins": ["B000000001", "B000000002", "B000000003"],
            "min_pool_size": 8,
        }

        readiness = service._build_candidate_expansion_data_readiness(
            job,
            {
                "registry_hit_count": 3,
                "snapshot_hit_count": 3,
                "history_hit_count": 3,
                "history_row_count": 270,
                "serving_base_hit_count": 0,
                "serving_base_row_count": 0,
            },
        )

        self.assertTrue(readiness["registry_ready"])
        self.assertTrue(readiness["hydration_ready"])
        self.assertFalse(readiness["analysis_ready"])
        self.assertEqual(readiness["readiness_status"], "serving_sync_pending")

    def test_candidate_expansion_readiness_marks_analysis_ready_after_serving_sync(self) -> None:
        service = ProductThemeService()
        job = {
            "result_candidate_asins": ["B000000001", "B000000002", "B000000003"],
            "min_pool_size": 8,
        }

        readiness = service._build_candidate_expansion_data_readiness(
            job,
            {
                "registry_hit_count": 3,
                "snapshot_hit_count": 3,
                "history_hit_count": 3,
                "history_row_count": 270,
                "serving_base_hit_count": 3,
                "serving_base_row_count": 90,
            },
        )

        self.assertTrue(readiness["serving_ready"])
        self.assertTrue(readiness["analysis_ready"])
        self.assertEqual(readiness["readiness_status"], "analysis_ready")

    def test_opportunity_discovery_request_normalizes_controls(self) -> None:
        request = OpportunityDiscoveryRequest(
            marketplace="us",
            platform="amazon",
            query="  humidifier  ",
            category_path="  Home & Kitchen > Humidifiers  ",
            min_data_confidence="MEDIUM",
            limit=5,
        )

        self.assertEqual(request.marketplace, "us")
        self.assertEqual(request.platform, "Amazon")
        self.assertEqual(request.query, "humidifier")
        self.assertEqual(request.category_path, "Home & Kitchen > Humidifiers")
        self.assertEqual(request.min_data_confidence, "medium")

    def test_opportunity_discovery_request_rejects_non_amazon_platform(self) -> None:
        with self.assertRaises(ValueError):
            OpportunityDiscoveryRequest(platform="TikTok")

    def test_seller_scope_blocks_digital_media_categories(self) -> None:
        service = ProductThemeService()
        rows = [
            {
                "category_id": 1,
                "category_path": "Digital Software > Antivirus & Security > Antivirus",
                "category_name": "Antivirus",
            },
            {
                "category_id": 2,
                "category_path": "Movies & TV > Movies",
                "category_name": "Movies",
            },
            {
                "category_id": 3,
                "category_path": "Sports & Outdoors > Golf > Golf Balls",
                "category_name": "Golf Balls",
            },
        ]

        kept, summary = service._filter_category_opportunity_rows_by_seller_scope(rows)

        self.assertEqual([row["category_id"] for row in kept], [3])
        self.assertEqual(summary["filtered_count"], 2)
        self.assertEqual(summary["reason_counts"]["digital_or_licensed_goods"], 1)
        self.assertEqual(summary["reason_counts"]["copyright_media"], 1)

    def test_seller_scope_allows_physical_opportunity_categories(self) -> None:
        for category_path in [
            "Clothing, Shoes & Jewelry > Men > Shirts",
            "Sports & Outdoors > Golf > Golf Balls",
            "Electronics > Headphones, Earbuds & Accessories > Earbud Headphones",
        ]:
            self.assertTrue(evaluate_seller_scope(category_path=category_path).allowed)

        self.assertFalse(evaluate_seller_scope(query="杀毒软件").allowed)
        self.assertFalse(evaluate_seller_scope(query="电影").allowed)

    def test_unclassified_opportunity_rows_do_not_become_unknown_cards(self) -> None:
        service = ProductThemeService()
        rows = [
            {
                "category_id": None,
                "category_path": "UNKNOWN",
                "category_name": "UNKNOWN",
                "candidate_count": 310,
                "row_count": 3006,
            },
            {
                "category_id": 3408951,
                "category_path": "Sports & Outdoors > Hunting & Fishing > Fishing",
                "category_name": "Fishing",
                "candidate_count": 34,
                "row_count": 334,
            },
            {
                "category_id": None,
                "category_path": "Home & Kitchen > Storage",
                "category_name": "Storage",
                "candidate_count": 12,
                "row_count": 120,
            },
        ]

        kept, summary = service._filter_unclassified_category_opportunity_rows(rows)

        self.assertEqual(len(kept), 2)
        self.assertEqual(summary["filtered_count"], 1)
        self.assertEqual(summary["reason"], "missing_category_id_path_and_name")
        self.assertEqual(kept[0]["category_name"], "Fishing")

    def test_candidate_expansion_rejects_out_of_scope_query(self) -> None:
        service = ProductThemeService()
        request = CandidateExpansionJobRequest(product_query="antivirus software license", recall_mode="keyword")

        with self.assertRaises(HTTPException) as context:
            service.create_candidate_expansion_job(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("seller_scope", context.exception.detail)

    def test_product_forecast_explain_request_defaults_to_top_10(self) -> None:
        request = ProductForecastExplainRequest(candidate_asins=["B001", "B002"], marketplace="US")

        self.assertEqual(request.top_n, 10)
        self.assertEqual(request.candidate_asins, ["B001", "B002"])

    def test_product_forecast_explain_request_rejects_large_top_n(self) -> None:
        with self.assertRaises(ValueError):
            ProductForecastExplainRequest(candidate_asins=["B001"], top_n=30)

    def test_opportunity_score_uses_design_weights(self) -> None:
        service = ProductThemeService()
        score = service._build_opportunity_score(
            {
                "demand_score": 100,
                "trend_score": 80,
                "competition_headroom_score": 60,
                "price_fit_score": 100,
                "forecast_growth_score": 40,
                "coverage_gap_score": 50,
                "evidence_quality_score": 80,
            }
        )

        self.assertEqual(score, 75.0)

    def test_category_opportunity_card_explains_metrics(self) -> None:
        service = ProductThemeService()
        card = service._build_category_opportunity_card(
            row={
                "category_id": 12345,
                "category_path": "Sports & Outdoors > Sports > Golf > Golf Balls",
                "category_name": "Golf Balls",
                "candidate_count": 12,
                "row_count": 120,
                "trend_rows": 60,
                "sales_window_sum": 3000.0,
                "sales_mean_7": 14.0,
                "sales_mean_prev": 10.0,
                "trend_mean_7": 30.0,
                "trend_mean_prev": 20.0,
                "price_p50": 39.99,
                "review_count_median": 100,
                "offer_count_avg": 0.61,
                "max_date": "2026-05-02",
            },
            marketplace="US",
            window_days=30,
            max_sales_window_sum=3000.0,
            include_expandable=True,
        )

        self.assertEqual(card["evidence_summary"]["candidate_count"], 12)
        self.assertEqual(card["evidence_summary"]["row_count"], 120)
        self.assertEqual(card["metric_explanations"]["sales_window_sum"]["candidate_count"], 12)
        self.assertEqual(card["metric_explanations"]["sales_window_sum"]["row_count"], 120)
        self.assertIn("不要默认加美元符号", card["metric_explanations"]["sales_window_sum"]["display_guidance"])
        self.assertEqual(
            card["metric_explanations"]["opportunity_score"]["weights"]["demand_score"],
            0.20,
        )
        self.assertIn("不是供应商数量", card["metric_explanations"]["offer_count_avg"]["plain_language"])

    def test_opportunity_metric_definitions_include_offer_meaning(self) -> None:
        service = ProductThemeService()
        definitions = service._opportunity_metric_definitions()

        self.assertIn("20%需求", definitions["opportunity_score"]["formula"])
        self.assertIn("candidate_count", definitions["sales_window_sum"]["sample_scope_fields"])
        self.assertIn("不要默认加美元符号", definitions["sales_window_sum"]["display_guidance"])
        self.assertIn("不是供应商数量", definitions["offer_count_avg"]["meaning"])

    def test_opportunity_llm_presentation_includes_table_and_explanations(self) -> None:
        service = ProductThemeService()
        card = service._build_category_opportunity_card(
            row={
                "category_id": 12345,
                "category_path": "Sports & Outdoors > Sports > Golf > Golf Balls",
                "category_name": "Golf Balls",
                "candidate_count": 12,
                "row_count": 120,
                "trend_rows": 60,
                "sales_window_sum": 3000.0,
                "sales_mean_7": 14.0,
                "sales_mean_prev": 10.0,
                "trend_mean_7": 30.0,
                "trend_mean_prev": 20.0,
                "price_p50": 39.99,
                "review_count_median": 100,
                "offer_count_avg": 0.61,
                "max_date": "2026-05-02",
            },
            marketplace="US",
            window_days=30,
            max_sales_window_sum=3000.0,
            include_expandable=True,
        )
        result = service._with_opportunity_llm_presentation(
            {
                "marketplace": "US",
                "platform": "Amazon",
                "opportunity_count": 1,
                "opportunities": [card],
                "metric_definitions": service._opportunity_metric_definitions(),
            }
        )

        text = result["opportunity_cards_text"]
        self.assertIn("| 排名 | 机会主题 | 得分 | 类目路径 | 窗口销量估算 | 样本ASIN数 | 日数据行数 |", text)
        self.assertIn("字段解释", text)
        self.assertIn("样本ASIN数=12", text)
        self.assertIn("不要把窗口销量估算渲染成美元金额", text)
        self.assertEqual(result["opportunities_for_llm"][0]["candidate_count"], 12)
        self.assertEqual(result["opportunities_for_llm"][0]["row_count"], 120)


if __name__ == "__main__":
    unittest.main()