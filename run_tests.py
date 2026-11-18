#!/usr/bin/env python3
"""Unified Test Runner for colab_with_mcp.
"""

import os
import sys
import unittest

from colab_mcp.config import ColabConfig


def main():
    print("=" * 65)
    print(" colab_with_mcp (API / MCP / CLI) Test Suite")
    print("=" * 65)

    is_configured, msg = ColabConfig.verify()
    print(f"Auth Status: {'[CONFIGURED]' if is_configured else '[NO TOKEN SET]'}")
    print(f"Details:     {msg}")
    print("-" * 65)

    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 65)
    if result.wasSuccessful():
        print("ALL TESTS PASSED! colab_with_mcp is fully functional.")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
