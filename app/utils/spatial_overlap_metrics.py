"""
Spatial Overlap Metrics Module

This module calculates spatial overlap metrics between two binary masks from NIfTI files.
Users can select ROI pairs for comparison, and metrics are computed per image series (CT/MR/PET).
Supports STAPLE contours and regular ROIs.

Available metrics:
1. Dice Similarity Coefficient (DSC)
2. Jaccard Similarity Coefficient (Jaccard)
3. 95% Hausdorff Distance (HD95)
4. Added Path Length (APL)
5. Mean Surface Distance (MSD)
6. Overcontouring Mean Distance to Conformity (OMDC)
7. Undercontouring Mean Distance to Conformity (UMDC)
8. Mean Distance to Conformity (MDC)
9. Volume Overlap Error (VOE)
10. Variation of Information (VI)
11. Cosine Similarity (Cosine)
12. Surface Dice Similarity Coefficient (SurfaceDSC)

Reference: https://github.com/CHAVI-India/draw-client-2.0/blob/main/spatial_overlap/utils/metrics.py
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from django.conf import settings
from django.db import transaction
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.metrics import mutual_info_score

logger = logging.getLogger(__name__)


def dice_similarity(volume1, volume2):
    """
    Calculate Dice Similarity Coefficient between two binary volumes.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Dice Similarity Coefficient.
    """
    intersection = np.sum((volume1 > 0) & (volume2 > 0))
    size1 = np.sum(volume1 > 0)
    size2 = np.sum(volume2 > 0)
    
    if size1 + size2 == 0:
        return 1.0
    return (2. * intersection) / (size1 + size2)


def jaccard_similarity(volume1, volume2):
    """
    Calculate Jaccard Similarity Coefficient between two binary volumes.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Jaccard Similarity Coefficient.
    """
    intersection = np.sum((volume1 > 0) & (volume2 > 0))
    union = np.sum((volume1 > 0) | (volume2 > 0))

    if union == 0:
        return 1.0
    return intersection / union


def volume_overlap_error(volume1, volume2):
    """
    Calculate Volume Overlap Error between two binary volumes.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Volume Overlap Error.
    """
    intersection = np.sum((volume1 > 0) & (volume2 > 0))
    union = np.sum((volume1 > 0) | (volume2 > 0))

    if union == 0:
        return 0.0
    return 1 - (intersection / union)


def variation_of_information(volume1, volume2):
    """
    Calculate Variation of Information between two binary volumes.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Variation of Information.
    """
    volume1_flat = volume1.flatten()
    volume2_flat = volume2.flatten()
    
    h1 = mutual_info_score(volume1_flat, volume1_flat)
    h2 = mutual_info_score(volume2_flat, volume2_flat)
    mi = mutual_info_score(volume1_flat, volume2_flat)

    return h1 + h2 - 2 * mi


def cosine_similarity(volume1, volume2):
    """
    Calculate Cosine Similarity between two binary volumes.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Cosine Similarity.
    """
    volume1_flat = volume1.flatten().reshape(1, -1)
    volume2_flat = volume2.flatten().reshape(1, -1)

    return sklearn_cosine_similarity(volume1_flat, volume2_flat)[0][0]


def compute_volume(volume, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate the volume of a binary mask in cubic centimeters (cm³).

    Args:
        volume (numpy.ndarray): Binary volume.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions in mm.

    Returns:
        float: Volume in cubic centimeters (cm³).
    """
    # Count non-zero voxels
    num_voxels = np.sum(volume > 0)
    
    # Calculate voxel volume in mm³
    voxel_volume_mm3 = np.prod(spacing)
    
    # Total volume in mm³
    total_volume_mm3 = num_voxels * voxel_volume_mm3
    
    # Convert to cm³ (1 cm³ = 1000 mm³)
    volume_cm3 = total_volume_mm3 / 1000.0
    
    return volume_cm3


def surface_dsc(volume1, volume2, tau=3.0, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Surface Dice Similarity Coefficient between two binary volumes.
    
    From: Nikolov S et al. Clinically Applicable Segmentation of Head and Neck Anatomy for
    Radiotherapy: Deep Learning Algorithm Development and Validation Study J Med Internet Res
    2021;23(7):e26151, DOI: 10.2196/26151

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.
        tau (float): Accepted deviation between contours in mm. Default is 3.0 mm.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions in mm.

    Returns:
        float: Surface Dice Similarity Coefficient (0 to 1, where 1 is perfect agreement).
    """
    vol1_binary = (volume1 > 0).astype(np.uint8)
    vol2_binary = (volume2 > 0).astype(np.uint8)

    if not np.any(vol1_binary) and not np.any(vol2_binary):
        return 1.0
    if not np.any(vol1_binary) or not np.any(vol2_binary):
        return 0.0

    label_a = sitk.GetImageFromArray(vol1_binary)
    label_a.SetSpacing(spacing)

    label_b = sitk.GetImageFromArray(vol2_binary)
    label_b.SetSpacing(spacing)

    binary_contour_filter = sitk.BinaryContourImageFilter()
    binary_contour_filter.FullyConnectedOn()
    a_contour = binary_contour_filter.Execute(label_a)
    b_contour = binary_contour_filter.Execute(label_b)

    dist_to_a = sitk.SignedMaurerDistanceMap(
        a_contour, useImageSpacing=True, squaredDistance=False
    )

    dist_to_b = sitk.SignedMaurerDistanceMap(
        b_contour, useImageSpacing=True, squaredDistance=False
    )

    b_intersection = sitk.GetArrayFromImage(b_contour * (dist_to_a <= tau)).sum()
    a_intersection = sitk.GetArrayFromImage(a_contour * (dist_to_b <= tau)).sum()

    surface_sum = (
        sitk.GetArrayFromImage(a_contour).sum()
        + sitk.GetArrayFromImage(b_contour).sum()
    )

    if surface_sum == 0:
        return 0.0

    return (b_intersection + a_intersection) / surface_sum


def mean_distance_to_conformity(volume1, volume2, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Mean Distance to Conformity between two binary volumes.
    
    MDC measures the average distance from voxels in the symmetric difference
    (XOR) of two volumes to the nearest boundary of the other volume.

    Args:
        volume1 (numpy.ndarray): Reference binary volume.
        volume2 (numpy.ndarray): Test binary volume.
        spacing (tuple): Voxel spacing in mm.

    Returns:
        float: Mean Distance to Conformity in mm.
    """
    vol1_binary = (volume1 > 0).astype(np.uint8)
    vol2_binary = (volume2 > 0).astype(np.uint8)

    under_region = vol1_binary & (~vol2_binary)
    under_coords = np.argwhere(under_region)

    over_region = vol2_binary & (~vol1_binary)
    over_coords = np.argwhere(over_region)

    under_distances = np.array([])
    over_distances = np.array([])

    if len(under_coords) > 0:
        under_distances = _calculate_axis_aligned_distance(under_coords, vol2_binary, spacing)

    if len(over_coords) > 0:
        over_distances = _calculate_axis_aligned_distance(over_coords, vol1_binary, spacing)

    valid_under = under_distances[~np.isnan(under_distances)] if len(under_distances) > 0 else np.array([])
    valid_over = over_distances[~np.isnan(over_distances)] if len(over_distances) > 0 else np.array([])

    under_mdc = np.mean(valid_under) if len(valid_under) > 0 else 0.0
    over_mdc = np.mean(valid_over) if len(valid_over) > 0 else 0.0

    if len(valid_under) == 0 and len(valid_over) == 0:
        return 0.0
    
    return (under_mdc + over_mdc) / 2.0


def hausdorff_distance_95(volume1, volume2, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Hausdorff Distance 95th percentile between two binary volumes.
    Uses SimpleITK methods matching PlatiPy's implementation exactly.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions.

    Returns:
        float: Hausdorff Distance 95th percentile.
    """
    # Convert numpy arrays to SimpleITK images
    label_a = sitk.GetImageFromArray((volume1 > 0).astype(np.uint8))
    label_a.SetSpacing(spacing)
    
    label_b = sitk.GetImageFromArray((volume2 > 0).astype(np.uint8))
    label_b.SetSpacing(spacing)
    
    # Check for empty volumes
    if sitk.GetArrayViewFromImage(label_a).sum() == 0 or sitk.GetArrayViewFromImage(label_b).sum() == 0:
        return np.inf
    
    # Use PlatiPy's approach: compute max distance per direction
    max_sd_list = []
    
    for la, lb in ((label_a, label_b), (label_b, label_a)):
        label_intensity_stat = sitk.LabelIntensityStatisticsImageFilter()
        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                la, squaredDistance=False, useImageSpacing=True
            )
        )
        moving_label_contour = sitk.LabelContour(lb)
        label_intensity_stat.Execute(moving_label_contour, reference_distance_map)
        
        max_sd_list.append(label_intensity_stat.GetMaximum(1))
    
    # HD95 is the 95th percentile of the two max distances
    hd_95 = np.percentile(max_sd_list, 95)
    
    return float(hd_95)


def mean_surface_distance(volume1, volume2, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Mean Surface Distance between two binary volumes.
    Uses SimpleITK methods matching PlatiPy's implementation exactly.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions.

    Returns:
        float: Mean Surface Distance.
    """
    # Convert numpy arrays to SimpleITK images
    label_a = sitk.GetImageFromArray((volume1 > 0).astype(np.uint8))
    label_a.SetSpacing(spacing)
    
    label_b = sitk.GetImageFromArray((volume2 > 0).astype(np.uint8))
    label_b.SetSpacing(spacing)
    
    # Check for empty volumes
    if sitk.GetArrayViewFromImage(label_a).sum() == 0 or sitk.GetArrayViewFromImage(label_b).sum() == 0:
        return np.inf
    
    # Use PlatiPy's approach: weighted average based on contour points
    mean_sd_list = []
    num_points = []
    
    for la, lb in ((label_a, label_b), (label_b, label_a)):
        label_intensity_stat = sitk.LabelIntensityStatisticsImageFilter()
        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                la, squaredDistance=False, useImageSpacing=True
            )
        )
        moving_label_contour = sitk.LabelContour(lb)
        label_intensity_stat.Execute(moving_label_contour, reference_distance_map)
        
        mean_sd_list.append(label_intensity_stat.GetMean(1))
        num_points.append(label_intensity_stat.GetNumberOfPixels(1))
    
    # Weighted average based on number of surface points
    mean_surf_dist = np.dot(mean_sd_list, num_points) / np.sum(num_points)
    
    return float(mean_surf_dist)


def added_path_length(volume1, volume2, distance_threshold_mm=3, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Added Path Length between two binary volumes.
    Measures the total contour length in the reference that is missing in the test segmentation.

    Args:
        volume1 (numpy.ndarray): Reference binary volume.
        volume2 (numpy.ndarray): Test binary volume.
        distance_threshold_mm (float): Distance threshold in mm. Default is 3mm.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions in mm.

    Returns:
        float: Total added path length in mm.
    """
    vol1_binary = (volume1 > 0).astype(np.uint8)
    vol2_binary = (volume2 > 0).astype(np.uint8)
    
    label_ref = sitk.GetImageFromArray(vol1_binary)
    label_ref.SetSpacing(spacing)
    
    label_test = sitk.GetImageFromArray(vol2_binary)
    label_test.SetSpacing(spacing)
    
    n_slices = label_ref.GetSize()[2]
    distance_voxels = int(np.ceil(distance_threshold_mm / np.mean(spacing[:2])))
    
    added_path_length_list = []
    
    for i in range(n_slices):
        ref_slice_array = sitk.GetArrayViewFromImage(label_ref)[i]
        test_slice_array = sitk.GetArrayViewFromImage(label_test)[i]
        
        if ref_slice_array.sum() + test_slice_array.sum() == 0:
            continue
        
        label_ref_slice = label_ref[:, :, i]
        label_test_slice = label_test[:, :, i]
        
        label_ref_contour = sitk.LabelContour(label_ref_slice)
        label_test_contour = sitk.LabelContour(label_test_slice)
        
        if distance_threshold_mm > 0:
            kernel = [int(distance_voxels) for k in range(2)]
            label_test_contour = sitk.BinaryDilate(label_test_contour, kernel)
        
        added_path = sitk.MaskNegated(label_ref_contour, label_test_contour)
        added_path_length = sitk.GetArrayViewFromImage(added_path).sum()
        added_path_length_list.append(added_path_length)
    
    total_apl_mm = np.sum(added_path_length_list) * np.mean(spacing[:2])
    
    return total_apl_mm


def _calculate_axis_aligned_distance(test_coords, ref_volume, spacing):
    """
    Calculate minimum axis-aligned distance from test voxels to reference volume boundary.
    
    Args:
        test_coords (numpy.ndarray): (N, 3) array of test voxel coordinates.
        ref_volume (numpy.ndarray): 3D binary reference volume.
        spacing (tuple): Voxel spacing in (x, y, z) dimensions.
    
    Returns:
        numpy.ndarray: Array of minimum distances for each test voxel.
    """
    if len(test_coords) == 0:
        return np.array([])
    
    distances = []
    shape = ref_volume.shape
    
    dist_max = np.sqrt(
        (shape[0] * spacing[0])**2 + 
        (shape[1] * spacing[1])**2 + 
        (shape[2] * spacing[2])**2
    )
    
    for i, j, k in test_coords:
        min_dist = dist_max
        
        for idx in range(i + 1, shape[0]):
            di = abs(idx - i) * spacing[0]
            if di > min_dist:
                break
            if ref_volume[idx, j, k] > 0:
                min_dist = di
                break
        
        for idx in range(i - 1, -1, -1):
            di = abs(idx - i) * spacing[0]
            if di > min_dist:
                break
            if ref_volume[idx, j, k] > 0:
                min_dist = di
                break
        
        for idx in range(j + 1, shape[1]):
            dj = abs(idx - j) * spacing[1]
            if dj > min_dist:
                break
            if ref_volume[i, idx, k] > 0:
                min_dist = dj
                break
        
        for idx in range(j - 1, -1, -1):
            dj = abs(idx - j) * spacing[1]
            if dj > min_dist:
                break
            if ref_volume[i, idx, k] > 0:
                min_dist = dj
                break
        
        for idx in range(k + 1, shape[2]):
            dk = abs(idx - k) * spacing[2]
            if dk > min_dist:
                break
            if ref_volume[i, j, idx] > 0:
                min_dist = dk
                break
        
        for idx in range(k - 1, -1, -1):
            dk = abs(idx - k) * spacing[2]
            if dk > min_dist:
                break
            if ref_volume[i, j, idx] > 0:
                min_dist = dk
                break
        
        distances.append(min_dist if min_dist != dist_max else np.nan)
    
    return np.array(distances)


def undercontouring_mean_distance_to_conformity(volume1, volume2, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Undercontouring Mean Distance to Conformity (UMDC).
    
    Args:
        volume1 (numpy.ndarray): Reference binary volume.
        volume2 (numpy.ndarray): Test binary volume.
        spacing (tuple): Voxel spacing in mm.

    Returns:
        float: UMDC in mm.
    """
    vol1_binary = (volume1 > 0).astype(np.uint8)
    vol2_binary = (volume2 > 0).astype(np.uint8)
    
    under_region = vol1_binary & (~vol2_binary)
    under_coords = np.argwhere(under_region)
    
    if len(under_coords) == 0:
        return 0.0
    
    under_distances = _calculate_axis_aligned_distance(under_coords, vol2_binary, spacing)
    valid_under = under_distances[~np.isnan(under_distances)]
    
    return np.mean(valid_under) if len(valid_under) > 0 else 0.0


def overcontouring_mean_distance_to_conformity(volume1, volume2, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Overcontouring Mean Distance to Conformity (OMDC).
    
    Args:
        volume1 (numpy.ndarray): Reference binary volume.
        volume2 (numpy.ndarray): Test binary volume.
        spacing (tuple): Voxel spacing in mm.

    Returns:
        float: OMDC in mm.
    """
    vol1_binary = (volume1 > 0).astype(np.uint8)
    vol2_binary = (volume2 > 0).astype(np.uint8)
    
    over_region = vol2_binary & (~vol1_binary)
    over_coords = np.argwhere(over_region)
    
    if len(over_coords) == 0:
        return 0.0
    
    over_distances = _calculate_axis_aligned_distance(over_coords, vol1_binary, spacing)
    valid_over = over_distances[~np.isnan(over_distances)]
    
    return np.mean(valid_over) if len(valid_over) > 0 else 0.0


def load_nifti_volume(nifti_path: Path) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Load a NIfTI file and return the volume array and spacing.
    
    Args:
        nifti_path: Path to the NIfTI file.
    
    Returns:
        Tuple of (volume array, spacing tuple) or (None, None) if error.
    """
    try:
        image = sitk.ReadImage(str(nifti_path))
        volume = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        return volume, spacing
    except Exception as e:
        logger.error(f"Error loading NIfTI file {nifti_path}: {e}")
        return None, None


def get_roi_nifti_path(roi) -> Optional[Path]:
    """
    Get the NIfTI file path for an ROI (RTStructROI or StapleROI).
    
    Args:
        roi: RTStructROI model instance.
    
    Returns:
        Path to the NIfTI file or None if not found.
    """
    from app.models import RTStructROI
    from app.utils.dcm_to_nifti_converter import sanitize_for_path
    
    try:
        if roi.instance:
            series = roi.instance.series
            if not series.nifti_file_path:
                return None
            
            nifti_dir = Path(settings.MEDIA_ROOT) / series.nifti_file_path
            roi_filename = sanitize_for_path(roi.roi_name)
            nifti_path = nifti_dir / f"{roi_filename}.nii.gz"
            
            if nifti_path.exists():
                return nifti_path
        
        elif roi.staple_roi:
            # STAPLE ROIs have the file path stored in the database
            if not roi.staple_roi.staple_roi_file_path:
                return None
            
            nifti_path = Path(settings.MEDIA_ROOT) / roi.staple_roi.staple_roi_file_path
            
            if nifti_path.exists():
                return nifti_path
        
        return None
    except Exception as e:
        logger.error(f"Error getting NIfTI path for ROI {roi.id}: {e}")
        return None


def get_rois_for_series(series_instance_uid: str) -> List:
    """
    Get all ROIs (including STAPLE) that are linked to a specific image series.
    
    Args:
        series_instance_uid: The series instance UID of the image series.
    
    Returns:
        List of RTStructROI instances with available NIfTI files.
    """
    from app.models import RTStructROI, DICOMSeries, DICOMInstance
    
    try:
        image_series = DICOMSeries.objects.get(series_instance_uid=series_instance_uid)
        
        rtstruct_instances = DICOMInstance.objects.filter(
            referenced_series_instance_uid=image_series,
            series__modality='RTSTRUCT'
        )
        
        rois = RTStructROI.objects.filter(
            instance__in=rtstruct_instances
        )
        
        # Get STAPLE ROIs for this image series
        # StapleROI.instance points to the image series instance
        # RTStructROI with staple_roi set and instance=None are STAPLE results
        image_instances = DICOMInstance.objects.filter(series=image_series)
        logger.info(f"  Found {image_instances.count()} image instances for series {series_instance_uid[-8:]}")
        
        staple_rois = RTStructROI.objects.filter(
            staple_roi__instance__in=image_instances,
            instance__isnull=True  # STAPLE results have no instance, only staple_roi
        )
        logger.info(f"  Found {staple_rois.count()} STAPLE ROIs for series {series_instance_uid[-8:]}")
        for sr in staple_rois:
            logger.info(f"    STAPLE ROI: {sr.roi_name} (ID: {sr.id})")
        
        all_rois = list(rois) + list(staple_rois)
        logger.info(f"  Total ROIs before NIfTI filter: {len(all_rois)} (regular: {len(rois)}, STAPLE: {len(staple_rois)})")
        
        available_rois = [roi for roi in all_rois if get_roi_nifti_path(roi) is not None]
        
        return available_rois
    
    except Exception as e:
        logger.error(f"Error getting ROIs for series {series_instance_uid}: {e}")
        return []


def compute_spatial_overlap_metrics(
    reference_roi_id: int,
    target_roi_id: int,
    save_to_db: bool = True
) -> Dict[str, float]:
    """
    Compute spatial overlap metrics between two ROIs and optionally save to database.
    
    Args:
        reference_roi_id: ID of the reference RTStructROI.
        target_roi_id: ID of the target RTStructROI.
        save_to_db: Whether to save results to StructureROIPair model.
    
    Returns:
        Dictionary with metric names and values.
    """
    from app.models import RTStructROI, StructureROIPair
    
    results = {
        'DSC': None,
        'Jaccard': None,
        'HD95': None,
        'APL': None,
        'MSD': None,
        'OMDC': None,
        'UMDC': None,
        'MDC': None,
        'VOE': None,
        'VI': None,
        'Cosine': None,
        'SurfaceDSC': None,
        'Volume_Ref': None,
        'Volume_Target': None,
        'error': None
    }
    
    try:
        reference_roi = RTStructROI.objects.get(id=reference_roi_id)
        target_roi = RTStructROI.objects.get(id=target_roi_id)
        
        ref_nifti_path = get_roi_nifti_path(reference_roi)
        target_nifti_path = get_roi_nifti_path(target_roi)
        
        if not ref_nifti_path or not target_nifti_path:
            results['error'] = "NIfTI files not found for one or both ROIs"
            return results
        
        ref_volume, ref_spacing = load_nifti_volume(ref_nifti_path)
        target_volume, target_spacing = load_nifti_volume(target_nifti_path)
        
        if ref_volume is None or target_volume is None:
            results['error'] = "Failed to load NIfTI volumes"
            return results
        
        if ref_volume.shape != target_volume.shape:
            results['error'] = f"Volume shapes do not match: {ref_volume.shape} vs {target_volume.shape}"
            return results
        
        spacing = ref_spacing
        
        logger.info(f"Computing metrics for ROI pair: {reference_roi.roi_name} vs {target_roi.roi_name}")
        logger.info(f"  Reference volume shape: {ref_volume.shape}, spacing: {ref_spacing}, unique values: {np.unique(ref_volume)}, non-zero voxels: {np.sum(ref_volume > 0)}")
        logger.info(f"  Target volume shape: {target_volume.shape}, spacing: {target_spacing}, unique values: {np.unique(target_volume)}, non-zero voxels: {np.sum(target_volume > 0)}")
        logger.info(f"  Reference NIfTI path: {ref_nifti_path}")
        logger.info(f"  Target NIfTI path: {target_nifti_path}")
        
        # Compute volumes for each ROI
        results['Volume_Ref'] = compute_volume(ref_volume, spacing=spacing)
        results['Volume_Target'] = compute_volume(target_volume, spacing=spacing)
        
        # Compute overlap metrics
        results['DSC'] = dice_similarity(ref_volume, target_volume)
        results['Jaccard'] = jaccard_similarity(ref_volume, target_volume)
        results['HD95'] = hausdorff_distance_95(ref_volume, target_volume, spacing=spacing)
        results['APL'] = added_path_length(ref_volume, target_volume, spacing=spacing)
        results['MSD'] = mean_surface_distance(ref_volume, target_volume, spacing=spacing)
        results['OMDC'] = overcontouring_mean_distance_to_conformity(ref_volume, target_volume, spacing=spacing)
        results['UMDC'] = undercontouring_mean_distance_to_conformity(ref_volume, target_volume, spacing=spacing)
        results['MDC'] = mean_distance_to_conformity(ref_volume, target_volume, spacing=spacing)
        results['VOE'] = volume_overlap_error(ref_volume, target_volume)
        results['VI'] = variation_of_information(ref_volume, target_volume)
        results['Cosine'] = cosine_similarity(ref_volume, target_volume)
        results['SurfaceDSC'] = surface_dsc(ref_volume, target_volume, tau=3.0, spacing=spacing)
        
        if save_to_db:
            with transaction.atomic():
                for metric_name, metric_value in results.items():
                    if metric_name != 'error' and metric_value is not None:
                        # Handle inf values for database storage
                        db_value = float(metric_value) if not np.isinf(metric_value) else None
                        if db_value is not None:
                            StructureROIPair.objects.create(
                                reference_rt_structure_roi=reference_roi,
                                target_rt_structure_roi=target_roi,
                                metric_calculated=metric_name,
                                metric_value=db_value
                            )
        
        # Convert numpy types to JSON-serializable Python types for return
        for key in ['DSC', 'Jaccard', 'HD95', 'APL', 'MSD', 'OMDC', 'UMDC', 'MDC', 'VOE', 'VI', 'Cosine', 'SurfaceDSC', 'Volume_Ref', 'Volume_Target']:
            if results[key] is not None:
                val = float(results[key])
                # Convert inf/nan to None for proper JSON serialization
                if np.isinf(val) or np.isnan(val):
                    results[key] = None
                else:
                    results[key] = val
        
        logger.info(f"Metrics computed successfully: {results}")
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        results['error'] = str(e)
    
    return results


def compute_metrics_for_multiple_pairs(
    roi_pairs: List[Tuple[int, int]],
    series_instance_uid: Optional[str] = None
) -> List[Dict]:
    """
    Compute spatial overlap metrics for multiple ROI pairs.
    
    Args:
        roi_pairs: List of tuples (reference_roi_id, target_roi_id).
        series_instance_uid: Optional series UID to validate ROIs belong to same series.
    
    Returns:
        List of result dictionaries for each pair.
    """
    results = []
    
    for ref_id, target_id in roi_pairs:
        logger.info(f"Processing ROI pair: {ref_id} vs {target_id}")
        result = compute_spatial_overlap_metrics(ref_id, target_id, save_to_db=True)
        results.append({
            'reference_roi_id': ref_id,
            'target_roi_id': target_id,
            'metrics': result
        })
    
    return results
