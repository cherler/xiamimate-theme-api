from __future__ import annotations

import unittest

from data_platform.api.product_theme import workspace_contract


class WorkspaceContractTests(unittest.TestCase):
    def test_build_payload_extracts_theme_and_title(self) -> None:
        payload = workspace_contract.build_workspace_payload(
            {"product_theme": "Portable Blender", "marketplace": "US"}
        )
        self.assertEqual(payload["theme_key"], "portable blender")
        self.assertEqual(payload["title"], "Portable Blender")
        self.assertEqual(payload["brief"]["marketplace"], "US")
        self.assertEqual(payload["schema_version"], workspace_contract.WORKSPACE_CONTRACT_SCHEMA_VERSION)

    def test_build_payload_safe_on_non_dict(self) -> None:
        payload = workspace_contract.build_workspace_payload(None)  # type: ignore[arg-type]
        self.assertEqual(payload["theme_key"], "unknown")
        self.assertIn("brief", payload)
        self.assertIn("evidence", payload)

    def test_maybe_attach_disabled_returns_unchanged(self) -> None:
        data = {"product_theme": "x"}
        out = workspace_contract.maybe_attach_workspace_payload(data, enabled=False)
        self.assertNotIn("workspace_payload", out)

    def test_maybe_attach_enabled_adds_payload_without_mutating_input(self) -> None:
        data = {"product_theme": "x"}
        out = workspace_contract.maybe_attach_workspace_payload(data, enabled=True)
        self.assertIn("workspace_payload", out)
        self.assertNotIn("workspace_payload", data)  # 原对象不被改写

    def test_maybe_attach_idempotent(self) -> None:
        data = {"product_theme": "x", "workspace_payload": {"existing": True}}
        out = workspace_contract.maybe_attach_workspace_payload(data, enabled=True)
        self.assertEqual(out["workspace_payload"], {"existing": True})


if __name__ == "__main__":
    unittest.main()
