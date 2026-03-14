"""
Mathematical Correctness Tests for Spatial Overlap Metrics

This test suite contains pre-calculated expected values for all metrics
to verify mathematical correctness of the implementations.

Each test case has:
1. Clearly defined input volumes (geometric shapes)
2. Pre-calculated expected metric values
3. Tolerance for floating-point comparison
"""

import numpy as np
import unittest
from app.utils.spatial_overlap_metrics import (
    dice_similarity,
    jaccard_similarity,
    hausdorff_distance_95,
    mean_surface_distance,
    added_path_length,
    overcontouring_mean_distance_to_conformity,
    undercontouring_mean_distance_to_conformity,
    mean_distance_to_conformity,
    volume_overlap_error,
    variation_of_information,
    cosine_similarity,
    surface_dsc,
)


class TestCase1_IdenticalCubes(unittest.TestCase):
    """
    Test Case 1: Two identical 4x4x4 cubes
    
    Expected Results:
    - DSC: 1.0 (perfect overlap)
    - Jaccard: 1.0 (perfect overlap)
    - HD95: 0.0 (identical surfaces)
    - MSD: 0.0 (identical surfaces)
    - APL: 0.0 (no missing contour)
    - OMDC: 0.0 (no overcontouring)
    - UMDC: 0.0 (no undercontouring)
    - MDC: 0.0 (perfect conformity)
    - VOE: 0.0 (no error)
    - Cosine: 1.0 (identical vectors)
    - Surface DSC: 1.0 (perfect surface agreement)
    """
    
    def setUp(self):
        # Create two identical 4x4x4 cubes in a 10x10x10 volume
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol1[3:7, 3:7, 3:7] = 255
        self.vol2 = self.vol1.copy()
        self.spacing = (1.0, 1.0, 1.0)
        
        # Pre-calculated expected values
        self.expected = {
            'DSC': 1.0,
            'Jaccard': 1.0,
            'HD95': 0.0,
            'MSD': 0.0,
            'APL': 0.0,
            'OMDC': 0.0,
            'UMDC': 0.0,
            'MDC': 0.0,
            'VOE': 0.0,
            'Cosine': 1.0,
            'SurfaceDSC': 1.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']}, got {result}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']}, got {result}")
    
    def test_hd95(self):
        result = hausdorff_distance_95(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['HD95'], places=6,
                              msg=f"HD95: expected {self.expected['HD95']}, got {result}")
    
    def test_msd(self):
        result = mean_surface_distance(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['MSD'], places=6,
                              msg=f"MSD: expected {self.expected['MSD']}, got {result}")
    
    def test_apl(self):
        result = added_path_length(self.vol1, self.vol2, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['APL'], places=6,
                              msg=f"APL: expected {self.expected['APL']}, got {result}")
    
    def test_omdc(self):
        result = overcontouring_mean_distance_to_conformity(self.vol1, self.vol2, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['OMDC'], places=6,
                              msg=f"OMDC: expected {self.expected['OMDC']}, got {result}")
    
    def test_umdc(self):
        result = undercontouring_mean_distance_to_conformity(self.vol1, self.vol2, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['UMDC'], places=6,
                              msg=f"UMDC: expected {self.expected['UMDC']}, got {result}")
    
    def test_mdc(self):
        result = mean_distance_to_conformity(self.vol1, self.vol2, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['MDC'], places=6,
                              msg=f"MDC: expected {self.expected['MDC']}, got {result}")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']}, got {result}")
    
    def test_cosine(self):
        result = cosine_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6,
                              msg=f"Cosine: expected {self.expected['Cosine']}, got {result}")
    
    def test_surface_dsc(self):
        result = surface_dsc(self.vol1, self.vol2, tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['SurfaceDSC'], places=6,
                              msg=f"Surface DSC: expected {self.expected['SurfaceDSC']}, got {result}")


class TestCase2_NoOverlap(unittest.TestCase):
    """
    Test Case 2: Two non-overlapping 2x2x2 cubes
    
    Cube 1: [1:3, 1:3, 1:3] - 8 voxels
    Cube 2: [7:9, 7:9, 7:9] - 8 voxels
    
    Expected Results:
    - DSC: 0.0 (no overlap)
    - Jaccard: 0.0 (no overlap)
    - HD95: ~10.39 (diagonal distance between cubes)
    - MSD: ~10.39 (mean distance between surfaces)
    - APL: > 0 (all reference contour is missing)
    - OMDC: 0.0 (no overcontouring - test doesn't extend into reference)
    - UMDC: 0.0 (no undercontouring - reference doesn't extend into test)
    - MDC: 0.0 (average of OMDC and UMDC)
    - VOE: 1.0 (complete error)
    - Cosine: 0.0 (orthogonal - no overlap)
    - Surface DSC: 0.0 (no surface agreement)
    """
    
    def setUp(self):
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol1[1:3, 1:3, 1:3] = 255  # 2x2x2 cube
        
        self.vol2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol2[7:9, 7:9, 7:9] = 255  # 2x2x2 cube, separated
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Pre-calculated expected values
        self.expected = {
            'DSC': 0.0,
            'Jaccard': 0.0,
            'VOE': 1.0,
            'Cosine': 0.0,
            'SurfaceDSC': 0.0,
            'OMDC': 0.0,
            'UMDC': 0.0,
            'MDC': 0.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']}, got {result}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']}, got {result}")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']}, got {result}")
    
    def test_cosine(self):
        result = cosine_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6,
                              msg=f"Cosine: expected {self.expected['Cosine']}, got {result}")
    
    def test_surface_dsc(self):
        result = surface_dsc(self.vol1, self.vol2, tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['SurfaceDSC'], places=6,
                              msg=f"Surface DSC: expected {self.expected['SurfaceDSC']}, got {result}")
    
    def test_hd95_positive(self):
        """HD95 should be positive for non-overlapping volumes"""
        result = hausdorff_distance_95(self.vol1, self.vol2)
        self.assertGreater(result, 5.0, msg=f"HD95 should be > 5.0 for separated cubes, got {result}")
    
    def test_msd_positive(self):
        """MSD should be positive for non-overlapping volumes"""
        result = mean_surface_distance(self.vol1, self.vol2)
        self.assertGreater(result, 5.0, msg=f"MSD should be > 5.0 for separated cubes, got {result}")


class TestCase3_PartialOverlap_50Percent(unittest.TestCase):
    """
    Test Case 3: Two cubes with exactly 50% overlap
    
    Cube 1: [2:6, 2:6, 2:6] - 4x4x4 = 64 voxels
    Cube 2: [4:8, 4:8, 4:8] - 4x4x4 = 64 voxels
    Intersection: [4:6, 4:6, 4:6] - 2x2x2 = 8 voxels
    Union: 64 + 64 - 8 = 120 voxels
    
    Expected Results:
    - DSC: 2*8 / (64+64) = 16/128 = 0.125
    - Jaccard: 8/120 = 0.0666...
    - VOE: 1 - 0.0666... = 0.9333...
    - Cosine: dot(v1,v2) / (||v1|| * ||v2||) = 8 / (8 * 8) = 8/64 = 0.125
    """
    
    def setUp(self):
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol1[2:6, 2:6, 2:6] = 255  # 4x4x4 = 64 voxels
        
        self.vol2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol2[4:8, 4:8, 4:8] = 255  # 4x4x4 = 64 voxels
        
        # Intersection: [4:6, 4:6, 4:6] = 2x2x2 = 8 voxels
        self.spacing = (1.0, 1.0, 1.0)
        
        # Pre-calculated expected values
        intersection = 8
        size1 = 64
        size2 = 64
        union = size1 + size2 - intersection  # 120
        
        self.expected = {
            'DSC': (2.0 * intersection) / (size1 + size2),  # 16/128 = 0.125
            'Jaccard': intersection / union,  # 8/120 = 0.0666...
            'VOE': 1.0 - (intersection / union),  # 0.9333...
            'Cosine': intersection / np.sqrt(size1 * size2),  # 8/64 = 0.125
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']:.6f}, got {result:.6f}")
    
    def test_cosine(self):
        result = cosine_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6,
                              msg=f"Cosine: expected {self.expected['Cosine']:.6f}, got {result:.6f}")
    
    def test_jaccard_dsc_relationship(self):
        """Verify mathematical relationship: Jaccard = DSC / (2 - DSC)"""
        dsc = dice_similarity(self.vol1, self.vol2)
        jaccard = jaccard_similarity(self.vol1, self.vol2)
        expected_jaccard = dsc / (2.0 - dsc)
        self.assertAlmostEqual(jaccard, expected_jaccard, places=6,
                              msg=f"Jaccard-DSC relationship failed: {jaccard:.6f} != {expected_jaccard:.6f}")
    
    def test_voe_jaccard_relationship(self):
        """Verify relationship: VOE = 1 - Jaccard"""
        jaccard = jaccard_similarity(self.vol1, self.vol2)
        voe = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(voe, 1.0 - jaccard, places=6,
                              msg=f"VOE-Jaccard relationship failed: {voe:.6f} != {1.0 - jaccard:.6f}")


class TestCase4_ConcentricCubes_Undercontouring(unittest.TestCase):
    """
    Test Case 4: Concentric cubes - smaller cube inside larger cube
    
    Reference (larger): [2:8, 2:8, 2:8] - 6x6x6 = 216 voxels
    Test (smaller): [3:7, 3:7, 3:7] - 4x4x4 = 64 voxels
    Intersection: [3:7, 3:7, 3:7] - 4x4x4 = 64 voxels (all of test)
    Union: 216 voxels (all of reference)
    
    Expected Results:
    - DSC: 2*64 / (216+64) = 128/280 = 0.457142...
    - Jaccard: 64/216 = 0.296296...
    - VOE: 1 - 0.296296... = 0.703703...
    - OMDC: 0.0 (test doesn't extend beyond reference)
    - UMDC: > 0 (reference extends beyond test by 1 voxel on each side)
    """
    
    def setUp(self):
        self.vol_reference = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_reference[2:8, 2:8, 2:8] = 255  # 6x6x6 = 216 voxels
        
        self.vol_test = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_test[3:7, 3:7, 3:7] = 255  # 4x4x4 = 64 voxels
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Pre-calculated expected values
        intersection = 64
        size_ref = 216
        size_test = 64
        union = size_ref  # Test is completely inside reference
        
        self.expected = {
            'DSC': (2.0 * intersection) / (size_ref + size_test),  # 128/280
            'Jaccard': intersection / union,  # 64/216
            'VOE': 1.0 - (intersection / union),  # 152/216
            'OMDC': 0.0,  # No overcontouring
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']:.6f}, got {result:.6f}")
    
    def test_omdc(self):
        """Test has no overcontouring (completely inside reference)"""
        result = overcontouring_mean_distance_to_conformity(
            self.vol_reference, self.vol_test, spacing=self.spacing
        )
        self.assertAlmostEqual(result, self.expected['OMDC'], places=6,
                              msg=f"OMDC: expected {self.expected['OMDC']:.6f}, got {result:.6f}")
    
    def test_umdc_positive(self):
        """Reference extends beyond test, so UMDC should be positive"""
        result = undercontouring_mean_distance_to_conformity(
            self.vol_reference, self.vol_test, spacing=self.spacing
        )
        # The undercontouring region is 1 voxel away from test boundary
        # Expected UMDC should be 1.0 (axis-aligned distance)
        self.assertAlmostEqual(result, 1.0, places=6,
                              msg=f"UMDC: expected 1.0, got {result:.6f}")


class TestCase5_ConcentricCubes_Overcontouring(unittest.TestCase):
    """
    Test Case 5: Concentric cubes - larger cube around smaller cube
    
    Reference (smaller): [3:7, 3:7, 3:7] - 4x4x4 = 64 voxels
    Test (larger): [2:8, 2:8, 2:8] - 6x6x6 = 216 voxels
    Intersection: [3:7, 3:7, 3:7] - 4x4x4 = 64 voxels (all of reference)
    Union: 216 voxels (all of test)
    
    Expected Results:
    - DSC: 2*64 / (64+216) = 128/280 = 0.457142...
    - Jaccard: 64/216 = 0.296296...
    - VOE: 1 - 0.296296... = 0.703703...
    - OMDC: > 0 (test extends beyond reference by 1 voxel on each side)
    - UMDC: 0.0 (reference doesn't extend beyond test)
    """
    
    def setUp(self):
        self.vol_reference = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_reference[3:7, 3:7, 3:7] = 255  # 4x4x4 = 64 voxels
        
        self.vol_test = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_test[2:8, 2:8, 2:8] = 255  # 6x6x6 = 216 voxels
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Pre-calculated expected values
        intersection = 64
        size_ref = 64
        size_test = 216
        union = size_test  # Reference is completely inside test
        
        self.expected = {
            'DSC': (2.0 * intersection) / (size_ref + size_test),  # 128/280
            'Jaccard': intersection / union,  # 64/216
            'VOE': 1.0 - (intersection / union),  # 152/216
            'UMDC': 0.0,  # No undercontouring
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol_reference, self.vol_test)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']:.6f}, got {result:.6f}")
    
    def test_umdc(self):
        """Reference doesn't extend beyond test, so UMDC should be 0"""
        result = undercontouring_mean_distance_to_conformity(
            self.vol_reference, self.vol_test, spacing=self.spacing
        )
        self.assertAlmostEqual(result, self.expected['UMDC'], places=6,
                              msg=f"UMDC: expected {self.expected['UMDC']:.6f}, got {result:.6f}")
    
    def test_omdc_positive(self):
        """Test extends beyond reference, so OMDC should be positive"""
        result = overcontouring_mean_distance_to_conformity(
            self.vol_reference, self.vol_test, spacing=self.spacing
        )
        # The overcontouring region is 1 voxel away from reference boundary
        # Expected OMDC should be 1.0 (axis-aligned distance)
        self.assertAlmostEqual(result, 1.0, places=6,
                              msg=f"OMDC: expected 1.0, got {result:.6f}")


class TestCase6_EmptyVolumes(unittest.TestCase):
    """
    Test Case 6: Both volumes are empty
    
    Expected Results (by convention):
    - DSC: 1.0 (perfect agreement on emptiness)
    - Jaccard: 1.0 (perfect agreement)
    - VOE: 0.0 (no error)
    - HD95: inf (no surfaces to compare)
    - MSD: inf (no surfaces to compare)
    - APL: 0.0 (no contours)
    - OMDC: 0.0 (no voxels to measure)
    - UMDC: 0.0 (no voxels to measure)
    - MDC: 0.0 (no voxels to measure)
    """
    
    def setUp(self):
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.spacing = (1.0, 1.0, 1.0)
        
        self.expected = {
            'DSC': 1.0,
            'Jaccard': 1.0,
            'VOE': 0.0,
            'APL': 0.0,
            'OMDC': 0.0,
            'UMDC': 0.0,
            'MDC': 0.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6)
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6)
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6)
    
    def test_hd95_is_inf(self):
        result = hausdorff_distance_95(self.vol1, self.vol2)
        self.assertTrue(np.isinf(result), msg=f"HD95 should be inf for empty volumes, got {result}")
    
    def test_msd_is_inf(self):
        result = mean_surface_distance(self.vol1, self.vol2)
        self.assertTrue(np.isinf(result), msg=f"MSD should be inf for empty volumes, got {result}")


class TestCase7_SingleVoxel(unittest.TestCase):
    """
    Test Case 7: Single voxel volumes
    
    Vol1: Single voxel at [5,5,5]
    Vol2: Single voxel at [5,5,5] (same location)
    
    Expected Results:
    - DSC: 1.0 (perfect overlap)
    - Jaccard: 1.0 (perfect overlap)
    - VOE: 0.0 (no error)
    - Cosine: 1.0 (identical)
    """
    
    def setUp(self):
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol1[5, 5, 5] = 255
        
        self.vol2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol2[5, 5, 5] = 255
        
        self.spacing = (1.0, 1.0, 1.0)
        
        self.expected = {
            'DSC': 1.0,
            'Jaccard': 1.0,
            'VOE': 0.0,
            'Cosine': 1.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6)
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6)
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6)
    
    def test_cosine(self):
        result = cosine_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6)


class TestCase8_DifferentIntensities(unittest.TestCase):
    """
    Test Case 8: Same region but different intensity values
    
    Vol1: Cube with intensity 255
    Vol2: Same cube with intensity 128
    
    All metrics should treat both as binary (>0), so results should be identical
    
    Expected Results:
    - DSC: 1.0
    - Jaccard: 1.0
    - VOE: 0.0
    """
    
    def setUp(self):
        self.vol1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol1[3:7, 3:7, 3:7] = 255
        
        self.vol2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol2[3:7, 3:7, 3:7] = 128  # Different intensity
        
        self.expected = {
            'DSC': 1.0,
            'Jaccard': 1.0,
            'VOE': 0.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg="DSC should be 1.0 regardless of intensity values")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg="Jaccard should be 1.0 regardless of intensity values")
    
    def test_voe(self):
        result = volume_overlap_error(self.vol1, self.vol2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg="VOE should be 0.0 regardless of intensity values")


class TestCase9_VariationOfInformation(unittest.TestCase):
    """
    Test Case 9: Variation of Information (VI) with known values
    
    VI = H(X) + H(Y) - 2*MI(X,Y)
    where H is entropy and MI is mutual information
    
    For identical binary volumes: VI = 0.0
    For completely different volumes: VI > 0.0
    """
    
    def setUp(self):
        # Identical volumes
        self.vol_identical_1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_identical_1[3:7, 3:7, 3:7] = 255
        self.vol_identical_2 = self.vol_identical_1.copy()
        
        # Completely different volumes (no overlap)
        self.vol_diff_1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_diff_1[1:3, 1:3, 1:3] = 255
        self.vol_diff_2 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_diff_2[7:9, 7:9, 7:9] = 255
        
        # Empty volumes
        self.vol_empty_1 = np.zeros((10, 10, 10), dtype=np.uint8)
        self.vol_empty_2 = np.zeros((10, 10, 10), dtype=np.uint8)
    
    def test_vi_identical_volumes(self):
        """VI should be 0.0 for identical volumes"""
        result = variation_of_information(self.vol_identical_1, self.vol_identical_2)
        self.assertAlmostEqual(result, 0.0, places=6,
                              msg=f"VI for identical volumes should be 0.0, got {result}")
    
    def test_vi_different_volumes(self):
        """VI should be > 0.0 for different volumes"""
        result = variation_of_information(self.vol_diff_1, self.vol_diff_2)
        self.assertGreaterEqual(result, 0.0,
                               msg=f"VI should be >= 0.0, got {result}")
    
    def test_vi_empty_volumes(self):
        """VI should be 0.0 for empty volumes"""
        result = variation_of_information(self.vol_empty_1, self.vol_empty_2)
        self.assertAlmostEqual(result, 0.0, places=6,
                              msg=f"VI for empty volumes should be 0.0, got {result}")
    
    def test_vi_symmetry(self):
        """VI should be symmetric: VI(A,B) = VI(B,A)"""
        vi_1 = variation_of_information(self.vol_diff_1, self.vol_diff_2)
        vi_2 = variation_of_information(self.vol_diff_2, self.vol_diff_1)
        self.assertAlmostEqual(vi_1, vi_2, places=6,
                              msg=f"VI should be symmetric: {vi_1} != {vi_2}")
    
    def test_vi_non_negative(self):
        """VI should always be non-negative"""
        # Test with random volumes
        vol1 = np.random.randint(0, 2, (10, 10, 10), dtype=np.uint8) * 255
        vol2 = np.random.randint(0, 2, (10, 10, 10), dtype=np.uint8) * 255
        result = variation_of_information(vol1, vol2)
        self.assertGreaterEqual(result, 0.0,
                               msg=f"VI should be non-negative, got {result}")


class TestCase10_SurfaceDSC(unittest.TestCase):
    """
    Test Case 10: Surface DSC with known tolerance values
    
    Surface DSC measures surface agreement within tolerance τ (tau)
    
    For identical volumes with τ=3mm: Surface DSC = 1.0
    For separated volumes with τ=3mm: Surface DSC = 0.0
    For slightly offset volumes within τ: Surface DSC > 0.0
    """
    
    def setUp(self):
        self.spacing = (1.0, 1.0, 1.0)
        
        # Identical volumes
        self.vol_identical_1 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_identical_1[5:15, 5:15, 5:15] = 255
        self.vol_identical_2 = self.vol_identical_1.copy()
        
        # Slightly offset volumes (1 voxel shift in x)
        self.vol_offset_1 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_offset_1[5:15, 5:15, 5:15] = 255
        self.vol_offset_2 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_offset_2[6:16, 5:15, 5:15] = 255  # Shifted by 1 voxel
        
        # Separated volumes
        self.vol_separated_1 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_separated_1[2:5, 2:5, 2:5] = 255
        self.vol_separated_2 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_separated_2[15:18, 15:18, 15:18] = 255
        
        # Empty volumes
        self.vol_empty_1 = np.zeros((20, 20, 20), dtype=np.uint8)
        self.vol_empty_2 = np.zeros((20, 20, 20), dtype=np.uint8)
    
    def test_surface_dsc_identical(self):
        """Surface DSC should be 1.0 for identical volumes"""
        result = surface_dsc(self.vol_identical_1, self.vol_identical_2, 
                           tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, 1.0, places=6,
                              msg=f"Surface DSC for identical volumes should be 1.0, got {result}")
    
    def test_surface_dsc_offset_within_tolerance(self):
        """Surface DSC should be > 0 for volumes offset within tolerance"""
        # 1 voxel offset = 1mm distance, which is < 3mm tolerance
        result = surface_dsc(self.vol_offset_1, self.vol_offset_2, 
                           tau=3.0, spacing=self.spacing)
        self.assertGreater(result, 0.0,
                          msg=f"Surface DSC should be > 0 for offset within tolerance, got {result}")
        self.assertLessEqual(result, 1.0,
                            msg=f"Surface DSC should be <= 1.0, got {result}")
    
    def test_surface_dsc_separated(self):
        """Surface DSC should be 0.0 for well-separated volumes"""
        result = surface_dsc(self.vol_separated_1, self.vol_separated_2, 
                           tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, 0.0, places=6,
                              msg=f"Surface DSC for separated volumes should be 0.0, got {result}")
    
    def test_surface_dsc_empty_both(self):
        """Surface DSC should be 1.0 for both empty volumes"""
        result = surface_dsc(self.vol_empty_1, self.vol_empty_2, 
                           tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, 1.0, places=6,
                              msg=f"Surface DSC for empty volumes should be 1.0, got {result}")
    
    def test_surface_dsc_one_empty(self):
        """Surface DSC should be 0.0 when one volume is empty"""
        result = surface_dsc(self.vol_identical_1, self.vol_empty_1, 
                           tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, 0.0, places=6,
                              msg=f"Surface DSC with one empty volume should be 0.0, got {result}")
    
    def test_surface_dsc_bounds(self):
        """Surface DSC should always be in [0, 1]"""
        result = surface_dsc(self.vol_offset_1, self.vol_offset_2, 
                           tau=3.0, spacing=self.spacing)
        self.assertGreaterEqual(result, 0.0,
                               msg=f"Surface DSC should be >= 0.0, got {result}")
        self.assertLessEqual(result, 1.0,
                            msg=f"Surface DSC should be <= 1.0, got {result}")
    
    def test_surface_dsc_tolerance_effect(self):
        """Larger tolerance should give higher Surface DSC for offset volumes"""
        # With small tolerance (0.5mm), offset of 1mm should give low score
        result_small_tau = surface_dsc(self.vol_offset_1, self.vol_offset_2, 
                                      tau=0.5, spacing=self.spacing)
        
        # With large tolerance (5mm), offset of 1mm should give high score
        result_large_tau = surface_dsc(self.vol_offset_1, self.vol_offset_2, 
                                      tau=5.0, spacing=self.spacing)
        
        self.assertGreater(result_large_tau, result_small_tau,
                          msg=f"Larger tolerance should give higher Surface DSC: "
                              f"{result_large_tau} should be > {result_small_tau}")


class TestCase11_IdenticalSpheres(unittest.TestCase):
    """
    Test Case 11: Two identical spheres
    
    Creates spherical shapes using distance from center
    Sphere radius = 5 voxels, centered in 20x20x20 volume
    
    Expected Results:
    - DSC: 1.0 (perfect overlap)
    - Jaccard: 1.0 (perfect overlap)
    - VOE: 0.0 (no error)
    - HD95: 0.0 (identical surfaces)
    - MSD: 0.0 (identical surfaces)
    - Cosine: 1.0 (identical)
    - Surface DSC: 1.0 (perfect surface agreement)
    """
    
    def setUp(self):
        # Create sphere using distance from center
        size = 20
        center = size // 2
        radius = 5
        
        # Create coordinate grids
        x, y, z = np.ogrid[:size, :size, :size]
        
        # Calculate distance from center
        distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
        
        # Create sphere (distance <= radius)
        self.sphere1 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere1[distance <= radius] = 255
        
        self.sphere2 = self.sphere1.copy()
        self.spacing = (1.0, 1.0, 1.0)
        
        self.expected = {
            'DSC': 1.0,
            'Jaccard': 1.0,
            'VOE': 0.0,
            'HD95': 0.0,
            'MSD': 0.0,
            'Cosine': 1.0,
            'SurfaceDSC': 1.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']}, got {result}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']}, got {result}")
    
    def test_voe(self):
        result = volume_overlap_error(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']}, got {result}")
    
    def test_hd95(self):
        result = hausdorff_distance_95(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['HD95'], places=6,
                              msg=f"HD95: expected {self.expected['HD95']}, got {result}")
    
    def test_msd(self):
        result = mean_surface_distance(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['MSD'], places=6,
                              msg=f"MSD: expected {self.expected['MSD']}, got {result}")
    
    def test_cosine(self):
        result = cosine_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6,
                              msg=f"Cosine: expected {self.expected['Cosine']}, got {result}")
    
    def test_surface_dsc(self):
        result = surface_dsc(self.sphere1, self.sphere2, tau=3.0, spacing=self.spacing)
        self.assertAlmostEqual(result, self.expected['SurfaceDSC'], places=6,
                              msg=f"Surface DSC: expected {self.expected['SurfaceDSC']}, got {result}")


class TestCase12_ConcentricSpheres(unittest.TestCase):
    """
    Test Case 12: Concentric spheres with different radii
    
    Inner sphere: radius = 3 voxels → volume ≈ 113 voxels (4/3 * π * 3³)
    Outer sphere: radius = 5 voxels → volume ≈ 523 voxels (4/3 * π * 5³)
    
    Inner sphere completely inside outer sphere
    Intersection = inner sphere volume
    
    Expected Results:
    - DSC: 2*113/(113+523) = 226/636 ≈ 0.355
    - Jaccard: 113/523 ≈ 0.216
    - VOE: 1 - 0.216 ≈ 0.784
    - OMDC: 0.0 (inner doesn't extend beyond outer)
    - UMDC: ~2.0 (outer extends ~2 voxels beyond inner on average)
    """
    
    def setUp(self):
        size = 20
        center = size // 2
        
        # Create coordinate grids
        x, y, z = np.ogrid[:size, :size, :size]
        distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
        
        # Inner sphere (radius = 3)
        self.sphere_inner = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere_inner[distance <= 3] = 255
        
        # Outer sphere (radius = 5)
        self.sphere_outer = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere_outer[distance <= 5] = 255
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Calculate actual volumes
        inner_volume = np.sum(self.sphere_inner > 0)
        outer_volume = np.sum(self.sphere_outer > 0)
        intersection = inner_volume  # Inner is completely inside outer
        
        self.expected = {
            'DSC': (2.0 * intersection) / (inner_volume + outer_volume),
            'Jaccard': intersection / outer_volume,
            'VOE': 1.0 - (intersection / outer_volume),
            'OMDC': 0.0,  # Inner doesn't extend beyond outer
        }
    
    def test_dsc(self):
        result = dice_similarity(self.sphere_inner, self.sphere_outer)
        self.assertAlmostEqual(result, self.expected['DSC'], places=3,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.sphere_inner, self.sphere_outer)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=3,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_voe(self):
        result = volume_overlap_error(self.sphere_inner, self.sphere_outer)
        self.assertAlmostEqual(result, self.expected['VOE'], places=3,
                              msg=f"VOE: expected {self.expected['VOE']:.6f}, got {result:.6f}")
    
    def test_omdc(self):
        """Inner sphere doesn't extend beyond outer, so OMDC should be 0"""
        result = overcontouring_mean_distance_to_conformity(
            self.sphere_outer, self.sphere_inner, spacing=self.spacing
        )
        self.assertAlmostEqual(result, self.expected['OMDC'], places=6,
                              msg=f"OMDC: expected {self.expected['OMDC']}, got {result}")
    
    def test_umdc_positive(self):
        """Outer sphere extends beyond inner, so UMDC should be positive"""
        result = undercontouring_mean_distance_to_conformity(
            self.sphere_outer, self.sphere_inner, spacing=self.spacing
        )
        # Should be approximately 2.0 (difference in radii)
        self.assertGreater(result, 0.0,
                          msg=f"UMDC should be > 0 for concentric spheres, got {result}")
        self.assertLess(result, 3.0,
                       msg=f"UMDC should be < 3.0 for these spheres, got {result}")


class TestCase13_OffsetSpheres(unittest.TestCase):
    """
    Test Case 13: Two spheres offset by known distance
    
    Sphere 1: radius = 4, center at (10, 10, 10)
    Sphere 2: radius = 4, center at (14, 10, 10) - offset by 4 voxels in x
    
    Distance between centers = 4 voxels
    Spheres touch but have minimal overlap
    
    Expected Results:
    - DSC: < 1.0 (partial overlap)
    - Jaccard: < 1.0 (partial overlap)
    - HD95: should reflect the offset distance
    - MSD: should be positive
    """
    
    def setUp(self):
        size = 25
        radius = 4
        
        # Sphere 1 centered at (10, 10, 10)
        x1, y1, z1 = np.ogrid[:size, :size, :size]
        distance1 = np.sqrt((x1 - 10)**2 + (y1 - 10)**2 + (z1 - 10)**2)
        self.sphere1 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere1[distance1 <= radius] = 255
        
        # Sphere 2 centered at (14, 10, 10) - offset by 4 in x
        x2, y2, z2 = np.ogrid[:size, :size, :size]
        distance2 = np.sqrt((x2 - 14)**2 + (y2 - 10)**2 + (z2 - 10)**2)
        self.sphere2 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere2[distance2 <= radius] = 255
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Calculate actual overlap
        intersection = np.sum((self.sphere1 > 0) & (self.sphere2 > 0))
        vol1_size = np.sum(self.sphere1 > 0)
        vol2_size = np.sum(self.sphere2 > 0)
        union = vol1_size + vol2_size - intersection
        
        self.expected = {
            'DSC': (2.0 * intersection) / (vol1_size + vol2_size),
            'Jaccard': intersection / union if union > 0 else 0.0,
        }
    
    def test_dsc_partial_overlap(self):
        """DSC should be between 0 and 1 for offset spheres"""
        result = dice_similarity(self.sphere1, self.sphere2)
        self.assertGreater(result, 0.0,
                          msg=f"DSC should be > 0 for touching spheres, got {result}")
        self.assertLess(result, 1.0,
                       msg=f"DSC should be < 1 for offset spheres, got {result}")
        # Verify it matches expected calculation
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard_partial_overlap(self):
        """Jaccard should be between 0 and 1 for offset spheres"""
        result = jaccard_similarity(self.sphere1, self.sphere2)
        self.assertGreater(result, 0.0,
                          msg=f"Jaccard should be > 0 for touching spheres, got {result}")
        self.assertLess(result, 1.0,
                       msg=f"Jaccard should be < 1 for offset spheres, got {result}")
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_hd95_positive(self):
        """HD95 should be positive for offset spheres"""
        result = hausdorff_distance_95(self.sphere1, self.sphere2)
        self.assertGreater(result, 0.0,
                          msg=f"HD95 should be > 0 for offset spheres, got {result}")
    
    def test_msd_positive(self):
        """MSD should be positive for offset spheres"""
        result = mean_surface_distance(self.sphere1, self.sphere2)
        self.assertGreater(result, 0.0,
                          msg=f"MSD should be > 0 for offset spheres, got {result}")
    
    def test_surface_dsc_with_tolerance(self):
        """Surface DSC should be high with large tolerance for slightly offset spheres"""
        # With large tolerance (10mm), offset of 4mm should give high score
        result = surface_dsc(self.sphere1, self.sphere2, tau=10.0, spacing=self.spacing)
        self.assertGreater(result, 0.5,
                          msg=f"Surface DSC should be > 0.5 with large tolerance, got {result}")


class TestCase14_SeparatedSpheres(unittest.TestCase):
    """
    Test Case 14: Two spheres completely separated
    
    Sphere 1: radius = 3, center at (7, 7, 7)
    Sphere 2: radius = 3, center at (17, 17, 17)
    
    Distance between centers ≈ 17.3 voxels (√(10² + 10² + 10²))
    No overlap
    
    Expected Results:
    - DSC: 0.0 (no overlap)
    - Jaccard: 0.0 (no overlap)
    - VOE: 1.0 (complete error)
    - Cosine: 0.0 (no overlap)
    - HD95: > 10.0 (large distance)
    - MSD: > 10.0 (large distance)
    """
    
    def setUp(self):
        size = 25
        radius = 3
        
        # Sphere 1 centered at (7, 7, 7)
        x1, y1, z1 = np.ogrid[:size, :size, :size]
        distance1 = np.sqrt((x1 - 7)**2 + (y1 - 7)**2 + (z1 - 7)**2)
        self.sphere1 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere1[distance1 <= radius] = 255
        
        # Sphere 2 centered at (17, 17, 17)
        x2, y2, z2 = np.ogrid[:size, :size, :size]
        distance2 = np.sqrt((x2 - 17)**2 + (y2 - 17)**2 + (z2 - 17)**2)
        self.sphere2 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere2[distance2 <= radius] = 255
        
        self.spacing = (1.0, 1.0, 1.0)
        
        self.expected = {
            'DSC': 0.0,
            'Jaccard': 0.0,
            'VOE': 1.0,
            'Cosine': 0.0,
        }
    
    def test_dsc(self):
        result = dice_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']}, got {result}")
    
    def test_jaccard(self):
        result = jaccard_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']}, got {result}")
    
    def test_voe(self):
        result = volume_overlap_error(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']}, got {result}")
    
    def test_cosine(self):
        result = cosine_similarity(self.sphere1, self.sphere2)
        self.assertAlmostEqual(result, self.expected['Cosine'], places=6,
                              msg=f"Cosine: expected {self.expected['Cosine']}, got {result}")
    
    def test_hd95_large(self):
        """HD95 should be large for separated spheres"""
        result = hausdorff_distance_95(self.sphere1, self.sphere2)
        self.assertGreater(result, 10.0,
                          msg=f"HD95 should be > 10.0 for separated spheres, got {result}")
    
    def test_msd_large(self):
        """MSD should be large for separated spheres"""
        result = mean_surface_distance(self.sphere1, self.sphere2)
        self.assertGreater(result, 10.0,
                          msg=f"MSD should be > 10.0 for separated spheres, got {result}")


class TestCase15_EspadonSpheres(unittest.TestCase):
    """
    Test Case 15: Espadon cross-verification test case
    
    This test case replicates the espadon R package test setup:
    - Sphere 1: radius = 10 units, center at (25, 25, 25)
    - Sphere 2: radius = 11 units, center at (28, 25, 25) - offset by dR=3 in x
    
    The espadon test uses:
    ```R
    R1 <- 10
    R2 <- 11
    dR <- 3
    sp.similarity.from.mesh(mesh1, mesh2,
                           hausdorff.quantile = seq(0, 1, 0.05),
                           surface.tol = seq(0, dR + abs(R2-R1), 0.5))
    ```
    
    Geometric analysis:
    - Distance between centers: 3 units
    - Sum of radii: 10 + 11 = 21 units
    - Since center distance (3) < sum of radii (21), spheres overlap
    - Overlap region can be calculated using sphere intersection formulas
    
    This test allows cross-verification with espadon's results for:
    - DSC (Dice Similarity Coefficient)
    - Jaccard Index
    - Hausdorff Distance (95th percentile)
    - Mean Surface Distance
    - Surface DSC with various tolerance values
    
    Note: Use this test to compare results with espadon's sp.similarity.from.mesh output
    """
    
    def setUp(self):
        # Use a larger volume to accommodate both spheres
        # Volume size: 60x60x60 to have enough space
        size = 60
        
        # Sphere 1: radius = 10, centered at (25, 25, 25)
        R1 = 10
        center1 = (25, 25, 25)
        
        x1, y1, z1 = np.ogrid[:size, :size, :size]
        distance1 = np.sqrt((x1 - center1[0])**2 + (y1 - center1[1])**2 + (z1 - center1[2])**2)
        self.sphere1 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere1[distance1 <= R1] = 255
        
        # Sphere 2: radius = 11, centered at (28, 25, 25) - offset by dR=3 in x
        R2 = 11
        dR = 3
        center2 = (25 + dR, 25, 25)  # (28, 25, 25)
        
        x2, y2, z2 = np.ogrid[:size, :size, :size]
        distance2 = np.sqrt((x2 - center2[0])**2 + (y2 - center2[1])**2 + (z2 - center2[2])**2)
        self.sphere2 = np.zeros((size, size, size), dtype=np.uint8)
        self.sphere2[distance2 <= R2] = 255
        
        self.spacing = (1.0, 1.0, 1.0)
        
        # Calculate actual volumes and overlap
        self.vol1_size = np.sum(self.sphere1 > 0)
        self.vol2_size = np.sum(self.sphere2 > 0)
        self.intersection = np.sum((self.sphere1 > 0) & (self.sphere2 > 0))
        self.union = self.vol1_size + self.vol2_size - self.intersection
        
        # Store parameters for reference
        self.R1 = R1
        self.R2 = R2
        self.dR = dR
        self.center1 = center1
        self.center2 = center2
        
        # Expected values (calculated from actual voxel data)
        self.expected = {
            'DSC': (2.0 * self.intersection) / (self.vol1_size + self.vol2_size),
            'Jaccard': self.intersection / self.union if self.union > 0 else 0.0,
            'VOE': 1.0 - (self.intersection / self.union) if self.union > 0 else 0.0,
        }
    
    def test_sphere_volumes(self):
        """Verify sphere volumes are approximately correct"""
        # Theoretical volumes: V = (4/3) * π * r³
        theoretical_vol1 = (4.0/3.0) * np.pi * (self.R1 ** 3)
        theoretical_vol2 = (4.0/3.0) * np.pi * (self.R2 ** 3)
        
        print(f"\nSphere Volume Verification:")
        print(f"  Sphere 1 (R={self.R1}): {self.vol1_size} voxels (theoretical: {theoretical_vol1:.1f})")
        print(f"  Sphere 2 (R={self.R2}): {self.vol2_size} voxels (theoretical: {theoretical_vol2:.1f})")
        
        # Allow some tolerance due to discretization
        tolerance = 0.15  # 15% tolerance for voxelization
        self.assertAlmostEqual(self.vol1_size, theoretical_vol1, delta=theoretical_vol1 * tolerance,
                              msg=f"Sphere 1 volume: expected ~{theoretical_vol1:.1f}, got {self.vol1_size}")
        self.assertAlmostEqual(self.vol2_size, theoretical_vol2, delta=theoretical_vol2 * tolerance,
                              msg=f"Sphere 2 volume: expected ~{theoretical_vol2:.1f}, got {self.vol2_size}")
    
    def test_spheres_overlap(self):
        """Verify that spheres do overlap (sanity check)"""
        self.assertGreater(self.intersection, 0,
                          msg=f"Spheres should overlap: R1={self.R1}, R2={self.R2}, dR={self.dR}")
        # Center distance < sum of radii means overlap
        center_distance = self.dR
        sum_of_radii = self.R1 + self.R2
        self.assertLess(center_distance, sum_of_radii,
                       msg=f"Center distance ({center_distance}) should be < sum of radii ({sum_of_radii})")
    
    def test_dsc_espadon(self):
        """Calculate DSC for espadon cross-verification"""
        result = dice_similarity(self.sphere1, self.sphere2)
        # Print for cross-verification with espadon (before assertion)
        print(f"\nEspadon Test - DSC: {result:.6f}")
        print(f"  Expected: {self.expected['DSC']:.6f}")
        print(f"  Sphere 1 volume: {self.vol1_size} voxels (R={self.R1})")
        print(f"  Sphere 2 volume: {self.vol2_size} voxels (R={self.R2})")
        print(f"  Intersection: {self.intersection} voxels")
        print(f"  Separation: dR={self.dR}")
        self.assertAlmostEqual(result, self.expected['DSC'], places=6,
                              msg=f"DSC: expected {self.expected['DSC']:.6f}, got {result:.6f}")
    
    def test_jaccard_espadon(self):
        """Calculate Jaccard for espadon cross-verification"""
        result = jaccard_similarity(self.sphere1, self.sphere2)
        print(f"\nEspadon Test - Jaccard: {result:.6f}")
        print(f"  Expected: {self.expected['Jaccard']:.6f}")
        self.assertAlmostEqual(result, self.expected['Jaccard'], places=6,
                              msg=f"Jaccard: expected {self.expected['Jaccard']:.6f}, got {result:.6f}")
    
    def test_voe_espadon(self):
        """Calculate VOE for espadon cross-verification"""
        result = volume_overlap_error(self.sphere1, self.sphere2)
        print(f"\nEspadon Test - VOE: {result:.6f}")
        print(f"  Expected: {self.expected['VOE']:.6f}")
        self.assertAlmostEqual(result, self.expected['VOE'], places=6,
                              msg=f"VOE: expected {self.expected['VOE']:.6f}, got {result:.6f}")
    
    def test_hausdorff_espadon(self):
        """Calculate HD95 for espadon cross-verification"""
        result = hausdorff_distance_95(self.sphere1, self.sphere2)
        print(f"\nEspadon Test - HD95: {result:.6f}")
        # HD95 should be positive but reasonable for overlapping spheres
        self.assertGreater(result, 0.0,
                          msg=f"HD95 should be > 0 for offset spheres, got {result:.6f}")
        # Maximum possible distance should be less than diameter of larger sphere + separation
        max_expected = 2 * self.R2 + self.dR
        self.assertLess(result, max_expected,
                       msg=f"HD95 should be < {max_expected}, got {result}")
    
    def test_msd_espadon(self):
        """Calculate MSD for espadon cross-verification"""
        result = mean_surface_distance(self.sphere1, self.sphere2)
        print(f"\nEspadon Test - MSD: {result:.6f}")
        self.assertGreater(result, 0.0,
                          msg=f"MSD should be > 0 for offset spheres, got {result:.6f}")
    
    def test_apl_espadon(self):
        """Calculate Added Path Length for espadon cross-verification"""
        result = added_path_length(self.sphere1, self.sphere2, spacing=self.spacing)
        print(f"\nEspadon Test - APL (Added Path Length): {result:.6f}")
        self.assertGreater(result, 0.0,
                          msg=f"APL should be > 0 for offset spheres, got {result:.6f}")
    
    def test_surface_dsc_espadon_multiple_tolerances(self):
        """Calculate Surface DSC with multiple tolerance values for espadon cross-verification
        
        Espadon uses: surface.tol = seq(0, dR + abs(R2-R1), 0.5)
        Which is: seq(0, 3 + abs(11-10), 0.5) = seq(0, 4, 0.5)
        Tolerance values: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
        """
        # Generate tolerance values as in espadon
        max_tol = self.dR + abs(self.R2 - self.R1)  # 3 + 1 = 4
        tolerances = np.arange(0.0, max_tol + 0.5, 0.5)
        
        print(f"\nEspadon Test - Surface DSC with varying tolerance:")
        print(f"  Tolerance range: 0 to {max_tol} (step 0.5)")
        
        surface_dsc_results = []
        for tau in tolerances:
            if tau == 0.0:
                # Skip tau=0 as it may cause issues
                continue
            result = surface_dsc(self.sphere1, self.sphere2, tau=tau, spacing=self.spacing)
            surface_dsc_results.append((tau, result))
            print(f"  tau={tau:.1f}: Surface DSC = {result:.6f}")
            
            # Surface DSC should be in [0, 1]
            self.assertGreaterEqual(result, 0.0,
                                   msg=f"Surface DSC should be >= 0, got {result} for tau={tau}")
            self.assertLessEqual(result, 1.0,
                                msg=f"Surface DSC should be <= 1, got {result} for tau={tau}")
        
        # Verify that Surface DSC increases with tolerance
        for i in range(len(surface_dsc_results) - 1):
            tau1, sdsc1 = surface_dsc_results[i]
            tau2, sdsc2 = surface_dsc_results[i + 1]
            self.assertLessEqual(sdsc1, sdsc2 + 0.01,  # Allow small numerical tolerance
                                msg=f"Surface DSC should increase with tolerance: "
                                    f"tau={tau1:.1f} gave {sdsc1:.6f}, tau={tau2:.1f} gave {sdsc2:.6f}")
    
    def test_cosine_similarity_espadon(self):
        """Calculate Cosine similarity for espadon cross-verification"""
        result = cosine_similarity(self.sphere1, self.sphere2)
        print(f"\nEspadon Test - Cosine Similarity: {result:.6f}")
        # Should be between 0 and 1 for overlapping volumes
        self.assertGreater(result, 0.0,
                          msg=f"Cosine should be > 0 for overlapping spheres, got {result:.6f}")
        self.assertLessEqual(result, 1.0,
                            msg=f"Cosine should be <= 1, got {result:.6f}")
    
    def test_print_summary_for_espadon_comparison(self):
        """Print comprehensive summary for easy comparison with espadon results"""
        print("\n" + "="*70)
        print("ESPADON CROSS-VERIFICATION TEST SUMMARY")
        print("="*70)
        print(f"\nTest Configuration:")
        print(f"  Sphere 1: radius = {self.R1} units, center = {self.center1}")
        print(f"  Sphere 2: radius = {self.R2} units, center = {self.center2}")
        print(f"  Separation: dR = {self.dR} units")
        print(f"  Voxel spacing: {self.spacing}")
        
        print(f"\nVolume Information:")
        print(f"  Sphere 1 volume: {self.vol1_size} voxels")
        print(f"  Sphere 2 volume: {self.vol2_size} voxels")
        print(f"  Intersection: {self.intersection} voxels")
        print(f"  Union: {self.union} voxels")
        
        print(f"\nMetric Results:")
        dsc = dice_similarity(self.sphere1, self.sphere2)
        jaccard = jaccard_similarity(self.sphere1, self.sphere2)
        voe = volume_overlap_error(self.sphere1, self.sphere2)
        hd95 = hausdorff_distance_95(self.sphere1, self.sphere2)
        msd = mean_surface_distance(self.sphere1, self.sphere2)
        apl = added_path_length(self.sphere1, self.sphere2, spacing=self.spacing)
        cosine = cosine_similarity(self.sphere1, self.sphere2)
        
        print(f"  DSC (Dice):              {dsc:.6f}")
        print(f"  Jaccard Index:           {jaccard:.6f}")
        print(f"  VOE:                     {voe:.6f}")
        print(f"  HD95:                    {hd95:.6f}")
        print(f"  MSD:                     {msd:.6f}")
        print(f"  APL:                     {apl:.6f}")
        print(f"  Cosine Similarity:       {cosine:.6f}")
        
        print(f"\nSurface DSC (at key tolerance values):")
        for tau in [0.5, 1.0, 2.0, 3.0, 4.0]:
            sdsc = surface_dsc(self.sphere1, self.sphere2, tau=tau, spacing=self.spacing)
            print(f"  tau = {tau:.1f}:  {sdsc:.6f}")
        
        print("\n" + "="*70)
        print("Compare these results with espadon's sp.similarity.from.mesh output")
        print("="*70 + "\n")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
