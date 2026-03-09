"""
Spatial Overlap Metrics Module

This module calculates spatial overlap metrics between two binary masks from NIfTI files.
Users can select ROI pairs for comparison, and metrics are computed per image series (CT/MR/PET).
Supports STAPLE contours and regular ROIs.

Required metrics:
1. Dice similarity coefficient (DSC)
2. 95% hausdorff distance (HD95)
3. Added path length (APL)
4. Mean surface distance (MSD)
5. Overcontouring MDC (OMDC) 
6. Undercontouring MDC (UMDC)

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


def hausdorff_distance_95(volume1, volume2):
    """
    Calculate Hausdorff Distance 95th percentile between two binary volumes.
    Uses distance transform for efficient computation.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Hausdorff Distance 95th percentile.
    """
    vol1_binary = volume1 > 0
    vol2_binary = volume2 > 0
    
    if not np.any(vol1_binary) or not np.any(vol2_binary):
        return np.inf
    
    dist1 = distance_transform_edt(~vol1_binary)
    dist2 = distance_transform_edt(~vol2_binary)
    
    surface_distances_1_to_2 = dist2[vol1_binary]
    surface_distances_2_to_1 = dist1[vol2_binary]
    
    hd_95_1_to_2 = np.percentile(surface_distances_1_to_2, 95)
    hd_95_2_to_1 = np.percentile(surface_distances_2_to_1, 95)
    
    return max(hd_95_1_to_2, hd_95_2_to_1)


def mean_surface_distance(volume1, volume2):
    """
    Calculate Mean Surface Distance between two binary volumes.
    Uses distance transform for efficient computation.

    Args:
        volume1 (numpy.ndarray): First binary volume.
        volume2 (numpy.ndarray): Second binary volume.

    Returns:
        float: Mean Surface Distance.
    """
    vol1_binary = volume1 > 0
    vol2_binary = volume2 > 0
    
    if not np.any(vol1_binary) or not np.any(vol2_binary):
        return np.inf
    
    dist1 = distance_transform_edt(~vol1_binary)
    dist2 = distance_transform_edt(~vol2_binary)
    
    surface_distances_1_to_2 = dist2[vol1_binary]
    surface_distances_2_to_1 = dist1[vol2_binary]
    
    msd1 = np.mean(surface_distances_1_to_2)
    msd2 = np.mean(surface_distances_2_to_1)
    
    return (msd1 + msd2) / 2


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
        'HD95': None,
        'APL': None,
        'MSD': None,
        'OMDC': None,
        'UMDC': None,
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
        
        results['DSC'] = dice_similarity(ref_volume, target_volume)
        results['HD95'] = hausdorff_distance_95(ref_volume, target_volume)
        results['APL'] = added_path_length(ref_volume, target_volume, spacing=spacing)
        results['MSD'] = mean_surface_distance(ref_volume, target_volume)
        results['OMDC'] = overcontouring_mean_distance_to_conformity(ref_volume, target_volume, spacing=spacing)
        results['UMDC'] = undercontouring_mean_distance_to_conformity(ref_volume, target_volume, spacing=spacing)
        
        if save_to_db:
            with transaction.atomic():
                for metric_name, metric_value in results.items():
                    if metric_name != 'error' and metric_value is not None:
                        StructureROIPair.objects.create(
                            reference_rt_structure_roi=reference_roi,
                            target_rt_structure_roi=target_roi,
                            metric_calculated=metric_name,
                            metric_value=float(metric_value) if not np.isinf(metric_value) else None
                        )
        
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
