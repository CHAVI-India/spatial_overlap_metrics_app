#!/usr/bin/env python
"""
Test Runner for Spatial Overlap Metrics

This script runs the mathematical correctness tests and provides a detailed report.

Usage:
    python run_metric_tests.py
    
Or with Django:
    python manage.py test app.tests.test_metrics_mathematical_correctness
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

from app.tests.test_metrics_mathematical_correctness import (
    TestCase1_IdenticalCubes,
    TestCase2_NoOverlap,
    TestCase3_PartialOverlap_50Percent,
    TestCase4_ConcentricCubes_Undercontouring,
    TestCase5_ConcentricCubes_Overcontouring,
    TestCase6_EmptyVolumes,
    TestCase7_SingleVoxel,
    TestCase8_DifferentIntensities,
    TestCase9_VariationOfInformation,
    TestCase10_SurfaceDSC,
    TestCase11_IdenticalSpheres,
    TestCase12_ConcentricSpheres,
    TestCase13_OffsetSpheres,
    TestCase14_SeparatedSpheres,
    TestCase15_EspadonSpheres,
)


def run_tests():
    """Run all mathematical correctness tests"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        TestCase1_IdenticalCubes,
        TestCase2_NoOverlap,
        TestCase3_PartialOverlap_50Percent,
        TestCase4_ConcentricCubes_Undercontouring,
        TestCase5_ConcentricCubes_Overcontouring,
        TestCase6_EmptyVolumes,
        TestCase7_SingleVoxel,
        TestCase8_DifferentIntensities,
        TestCase9_VariationOfInformation,
        TestCase10_SurfaceDSC,
        TestCase11_IdenticalSpheres,
        TestCase12_ConcentricSpheres,
        TestCase13_OffsetSpheres,
        TestCase14_SeparatedSpheres,
        TestCase15_EspadonSpheres,
    ]
    
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
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
    sys.exit(run_tests())
