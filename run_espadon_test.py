#!/usr/bin/env python
"""
Run the Espadon Cross-Verification Test

This script runs only the TestCase15_EspadonSpheres test case
to compare results with espadon's sp.similarity.from.mesh function.

Usage:
    python run_espadon_test.py
"""

import sys
import os
import unittest

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up Django settings if needed
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'spatialmetrics.settings')

try:
    import django
    django.setup()
except:
    pass

from app.tests.test_metrics_mathematical_correctness import TestCase15_EspadonSpheres


def run_espadon_test():
    """Run the Espadon cross-verification test"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCase15_EspadonSpheres)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("ESPADON TEST SUMMARY")
    print("="*70)
    print(f"Total tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n" + "="*70)
        print("FAILURES")
        print("="*70)
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\n" + "="*70)
        print("ERRORS")
        print("="*70)
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_espadon_test())
