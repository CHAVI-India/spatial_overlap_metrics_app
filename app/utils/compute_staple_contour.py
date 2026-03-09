"""
STAPLE Contour Computation Module

This module computes STAPLE (Simultaneous Truth and Performance Level Estimation) contours
for a given structure from multiple raters (structure sets). For a given image series,
multiple structure sets are combined to generate the STAPLE contour.

The STAPLE contour is stored in the same location as NIfTI files with filename:
staple_<structurename>.nii.gz

Database relationship flow:
Patient -> DICOMStudy -> DICOMSeries -> StapleROI -> RTStructROI
(One STAPLE ROI linked to multiple RTStructROI entries from multiple structure sets)
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import SimpleITK as sitk
import numpy as np
from django.conf import settings
from django.db import transaction

logger = logging.getLogger(__name__)


def sanitize_for_path(name: str) -> str:
    """Sanitize a string to be safe for use in file paths."""
    import re
    sanitized = re.sub(r'[^\w\-]', '_', str(name))
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized


def compute_staple_contour(
    image_series_id: int,
    structure_name: str,
    rtstruct_series_ids: List[int],
    foreground_value: int = 1,
    threshold: float = 0.95,
    progress_callback=None
) -> Tuple[Optional[str], List[str]]:
    """
    Compute STAPLE contour from multiple structure sets for a given structure.
    
    Args:
        image_series_id: ID of the reference image series
        structure_name: Name of the structure to compute STAPLE for
        rtstruct_series_ids: List of RTStruct series IDs containing the structure
        foreground_value: Value representing foreground in segmentations (default: 1)
        threshold: Probability threshold for STAPLE result (default: 0.95)
        progress_callback: Optional callback function(progress_pct, message)
        
    Returns:
        Tuple of (staple_nifti_path, errors)
    """
    from app.models import DICOMSeries, StapleROI, RTStructROI, DICOMInstance
    
    errors = []
    
    try:
        if progress_callback:
            progress_callback(5, f"Loading image series and structure masks for {structure_name}...")
        
        # Get the image series
        try:
            image_series = DICOMSeries.objects.get(id=image_series_id)
        except DICOMSeries.DoesNotExist:
            errors.append(f"Image series {image_series_id} not found")
            return None, errors
        
        # Sanitize structure name for filename
        safe_structure_name = sanitize_for_path(structure_name)
        
        # Collect all segmentation masks for this structure
        segmentations = []
        rtstruct_roi_ids = []
        
        for idx, rtstruct_id in enumerate(rtstruct_series_ids):
            if progress_callback:
                pct = 10 + int((idx / len(rtstruct_series_ids)) * 40)
                progress_callback(pct, f"Loading mask {idx+1}/{len(rtstruct_series_ids)}...")
            
            try:
                rtstruct_series = DICOMSeries.objects.get(id=rtstruct_id)
            except DICOMSeries.DoesNotExist:
                logger.warning(f"RTStruct series {rtstruct_id} not found")
                continue
            
            # Get the NIfTI directory for this RTStruct
            if not rtstruct_series.nifti_file_path:
                logger.warning(f"RTStruct series {rtstruct_id} has no NIfTI files")
                continue
            
            nifti_dir = Path(settings.MEDIA_ROOT) / rtstruct_series.nifti_file_path
            
            # Find the mask file for this structure
            mask_filename = nifti_dir / f"{safe_structure_name}.nii.gz"
            
            if not mask_filename.exists():
                logger.warning(f"Mask file not found: {mask_filename}")
                continue
            
            # Load the mask
            try:
                mask = sitk.ReadImage(str(mask_filename))
                segmentations.append(mask)
                
                # Get the RTStructROI entry for tracking
                roi_instance = DICOMInstance.objects.filter(series=rtstruct_series).first()
                if roi_instance:
                    roi_entry = RTStructROI.objects.filter(
                        instance=roi_instance,
                        roi_name=structure_name
                    ).first()
                    if roi_entry:
                        rtstruct_roi_ids.append(roi_entry.id)
                
                logger.info(f"Loaded mask from {mask_filename.name}")
            except Exception as e:
                logger.error(f"Failed to load mask {mask_filename}: {e}")
                errors.append(f"Failed to load mask from series {rtstruct_id}: {str(e)}")
                continue
        
        if len(segmentations) < 2:
            errors.append(f"Need at least 2 segmentations for STAPLE, found {len(segmentations)}")
            return None, errors
        
        if progress_callback:
            progress_callback(50, f"Computing STAPLE from {len(segmentations)} segmentations...")
        
        # Compute STAPLE
        logger.info(f"Computing STAPLE for {structure_name} from {len(segmentations)} segmentations")
        
        try:
            # Use STAPLE algorithm to get probability map
            staple_probabilities = sitk.STAPLE(segmentations, foreground_value)
            
            # Threshold to get binary segmentation
            staple_segmentation = staple_probabilities > threshold
            
            # Cast to unsigned char for consistency
            staple_segmentation = sitk.Cast(staple_segmentation, sitk.sitkUInt8)
            
            # Set intensity values
            staple_segmentation = staple_segmentation * 255
            
        except Exception as e:
            logger.error(f"STAPLE computation failed: {e}")
            errors.append(f"STAPLE computation failed: {str(e)}")
            return None, errors
        
        if progress_callback:
            progress_callback(70, "Saving STAPLE contour...")
        
        # Determine output path (same location as image series NIfTI)
        patient_id = sanitize_for_path(image_series.study.patient.patient_id)
        study_uid = sanitize_for_path(image_series.study.study_instance_uid)
        series_uid = sanitize_for_path(image_series.series_instance_uid)
        
        staple_dir = Path(settings.MEDIA_ROOT) / "nifti_files" / patient_id / study_uid / series_uid / "staple"
        staple_dir.mkdir(parents=True, exist_ok=True)
        
        # Save STAPLE contour
        output_filename = f"staple_{safe_structure_name}.nii.gz"
        output_path = staple_dir / output_filename
        
        sitk.WriteImage(staple_segmentation, str(output_path))
        logger.info(f"Saved STAPLE contour to {output_path}")
        
        if progress_callback:
            progress_callback(85, "Updating database...")
        
        # Update database
        with transaction.atomic():
            # Create a StapleROI entry
            # We need to link it to a DICOMInstance - use the image series
            image_instance = DICOMInstance.objects.filter(series=image_series).first()
            
            if not image_instance:
                errors.append("No DICOM instance found for image series")
                return None, errors
            
            # Get relative path for database storage
            relative_path = output_path.relative_to(settings.MEDIA_ROOT)
            
            staple_roi = StapleROI.objects.create(
                instance=image_instance,
                staple_roi_file_path=str(relative_path)
            )
            
            # Create RTStructROI entry for the STAPLE result
            staple_rtstruct_roi = RTStructROI.objects.create(
                staple_roi=staple_roi,
                roi_name=f"STAPLE_{structure_name}",
                roi_number=None
            )
            
            # Link existing RTStructROI entries to this STAPLE ROI
            for roi_id in rtstruct_roi_ids:
                try:
                    roi = RTStructROI.objects.get(id=roi_id)
                    roi.staple_roi = staple_roi
                    roi.save(update_fields=['staple_roi'])
                except RTStructROI.DoesNotExist:
                    logger.warning(f"RTStructROI {roi_id} not found")
        
        if progress_callback:
            progress_callback(100, "STAPLE computation complete")
        
        # Return relative path
        relative_path = output_path.relative_to(settings.MEDIA_ROOT)
        return str(relative_path), errors
        
    except Exception as e:
        error_msg = f"Error computing STAPLE for {structure_name}: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        return None, errors
