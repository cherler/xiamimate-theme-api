from __future__ import annotations

import unittest

from data_platform.api.product_theme_api import (
    CandidateRecord,
    _build_query_variants,
    _candidate_field_matches_required_terms,
    _category_level_name,
    _fine_category_name,
    _leaf_category_name,
    _score_candidate,
)


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


if __name__ == "__main__":
    unittest.main()