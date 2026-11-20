#!/usr/bin/env python3
"""colab_tools 統合テストランナー。"""

import argparse
import os
import sys
import unittest

from colab_mcp.config import ColabConfig


def _filter_suite(suite: unittest.TestSuite, skip_live: bool) -> unittest.TestSuite:
    """必要に応じてライブネットワークテスト（test_live_mcp）を除外したスイートを構築する。"""
    if not skip_live:
        return suite
    filtered = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            filtered.addTest(_filter_suite(item, skip_live))
        elif "test_live_mcp" not in item.id():
            filtered.addTest(item)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="colab_tools テストスイートを実行する。")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="外部ネットワーク通信を行う test_live_mcp を除外し、単体テストのみ実行する",
    )
    args = parser.parse_args()
    skip_live = args.offline or (
        os.environ.get("COLAB_SKIP_LIVE_TESTS", "").strip().lower()
        in ("1", "true", "yes")
    )

    print("=" * 65)
    print(" colab_tools (API / MCP / CLI) Test Suite")
    print("=" * 65)

    is_configured, msg = ColabConfig.verify()
    print(f"Auth Status: {'[CONFIGURED]' if is_configured else '[NO TOKEN SET]'}")
    print(f"Details:     {msg}")
    if skip_live:
        print("Mode:        [OFFLINE] ライブ MCP 契約テストをスキップします")
    print("-" * 65)

    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = _filter_suite(loader.discover(start_dir, pattern="test_*.py"), skip_live)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 65)
    if result.wasSuccessful():
        print("ALL TESTS PASSED! colab_tools is fully functional.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
