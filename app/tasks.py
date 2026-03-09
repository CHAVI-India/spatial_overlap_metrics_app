"""
Celery tasks for the app.
"""

import logging
from celery import shared_task
from celery_progress.backend import ProgressRecorder

from app.models import DICOMSeries
from app.utils.dcm_to_nifti_converter import convert_series_with_rtstructs

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def convert_series_to_nifti(self, series_ids):
    """
    Celery task to convert multiple DICOM series and their associated RTStructs to NIfTI.
    
    Args:
        series_ids: List of DICOMSeries IDs to convert
        
    Returns:
        Dictionary with conversion results for all series
    """
    progress_recorder = ProgressRecorder(self)
    total_series = len(series_ids)
    results = {
        'success': True,
        'total_series': total_series,
        'processed_series': 0,
        'failed_series': 0,
        'series_results': [],
        'errors': []
    }
    
    def progress_callback(pct, message):
        """Update progress for Celery."""
        overall_pct = int((results['processed_series'] / total_series) * 100 + (pct / total_series))
        progress_recorder.set_progress(overall_pct, 100, description=message)
    
    for idx, series_id in enumerate(series_ids, 1):
        try:
            series = DICOMSeries.objects.get(id=series_id)
            base_message = f"Converting series {idx}/{total_series}: {series.series_instance_uid[:20]}..."
            progress_recorder.set_progress(
                int((idx - 1) / total_series * 100),
                100,
                description=base_message
            )
            
            # Convert the series and its associated RTStructs
            result = convert_series_with_rtstructs(series_id, progress_callback)
            
            if result['success']:
                results['processed_series'] += 1
            else:
                results['failed_series'] += 1
                results['errors'].extend(result.get('errors', []))
            
            results['series_results'].append({
                'series_id': series_id,
                'series_uid': series.series_instance_uid,
                'success': result['success'],
                'image_nifti': result.get('image_nifti'),
                'rtstruct_count': len(result.get('rtstruct_niftis', [])),
                'errors': result.get('errors', [])
            })
            
        except DICOMSeries.DoesNotExist:
            error_msg = f"Series {series_id} not found"
            logger.error(error_msg)
            results['failed_series'] += 1
            results['errors'].append(error_msg)
            results['series_results'].append({
                'series_id': series_id,
                'success': False,
                'errors': [error_msg]
            })
        except Exception as e:
            error_msg = f"Error processing series {series_id}: {str(e)}"
            logger.error(error_msg)
            results['failed_series'] += 1
            results['errors'].append(error_msg)
            results['series_results'].append({
                'series_id': series_id,
                'success': False,
                'errors': [error_msg]
            })
    
    # Final progress update
    progress_recorder.set_progress(100, 100, description="NIfTI conversion complete!")
    
    # Set overall success flag
    if results['failed_series'] > 0:
        results['success'] = results['processed_series'] > 0
    
    return results


@shared_task(bind=True)
def compute_staple_task(self, image_series_id, structure_name, rtstruct_series_ids, threshold=0.95):
    """
    Celery task to compute STAPLE contour from multiple structure sets.
    
    Args:
        image_series_id: ID of the reference image series
        structure_name: Name of the structure to compute STAPLE for
        rtstruct_series_ids: List of RTStruct series IDs containing the structure
        threshold: Probability threshold for STAPLE result (default: 0.95)
        
    Returns:
        Dictionary with STAPLE computation results
    """
    from app.utils.compute_staple_contour import compute_staple_contour
    
    progress_recorder = ProgressRecorder(self)
    
    def progress_callback(pct, message):
        """Update progress for Celery."""
        progress_recorder.set_progress(pct, 100, description=message)
    
    # Compute STAPLE
    staple_path, errors = compute_staple_contour(
        image_series_id=image_series_id,
        structure_name=structure_name,
        rtstruct_series_ids=rtstruct_series_ids,
        threshold=threshold,
        progress_callback=progress_callback
    )
    
    result = {
        'success': staple_path is not None,
        'staple_path': staple_path,
        'structure_name': structure_name,
        'num_segmentations': len(rtstruct_series_ids),
        'errors': errors
    }
    
    return result


@shared_task(bind=True)
def compute_batch_staple_task(self, staple_requests):
    """
    Celery task to compute STAPLE contours for multiple ROIs across multiple patients.
    
    Args:
        staple_requests: List of dicts with keys:
            - image_series_id: ID of the reference image series
            - structure_name: Name of the structure
            - rtstruct_series_ids: List of RTStruct series IDs
            - threshold: Probability threshold (default: 0.95)
        
    Returns:
        Dictionary with batch STAPLE computation results
    """
    from app.utils.compute_staple_contour import compute_staple_contour
    
    progress_recorder = ProgressRecorder(self)
    total_requests = len(staple_requests)
    
    results = {
        'success': True,
        'total_requests': total_requests,
        'completed': 0,
        'failed': 0,
        'staple_results': [],
        'errors': []
    }
    
    for idx, req in enumerate(staple_requests, 1):
        try:
            image_series_id = req['image_series_id']
            structure_name = req['structure_name']
            rtstruct_series_ids = req['rtstruct_series_ids']
            threshold = req.get('threshold', 0.95)
            
            # Update progress
            base_pct = int((idx - 1) / total_requests * 100)
            progress_recorder.set_progress(
                base_pct,
                100,
                description=f"Computing STAPLE {idx}/{total_requests}: {structure_name}"
            )
            
            def progress_callback(pct, message):
                """Update progress for this specific STAPLE computation."""
                overall_pct = base_pct + int(pct / total_requests)
                progress_recorder.set_progress(overall_pct, 100, description=message)
            
            # Compute STAPLE
            staple_path, errors = compute_staple_contour(
                image_series_id=image_series_id,
                structure_name=structure_name,
                rtstruct_series_ids=rtstruct_series_ids,
                threshold=threshold,
                progress_callback=progress_callback
            )
            
            if staple_path:
                results['completed'] += 1
                results['staple_results'].append({
                    'success': True,
                    'structure_name': structure_name,
                    'image_series_id': image_series_id,
                    'staple_path': staple_path,
                    'num_segmentations': len(rtstruct_series_ids),
                    'errors': errors
                })
            else:
                results['failed'] += 1
                results['errors'].extend(errors)
                results['staple_results'].append({
                    'success': False,
                    'structure_name': structure_name,
                    'image_series_id': image_series_id,
                    'errors': errors
                })
                
        except Exception as e:
            error_msg = f"Error processing STAPLE request {idx}: {str(e)}"
            logger.error(error_msg)
            results['failed'] += 1
            results['errors'].append(error_msg)
            results['staple_results'].append({
                'success': False,
                'structure_name': req.get('structure_name', 'Unknown'),
                'errors': [error_msg]
            })
    
    # Final progress update
    progress_recorder.set_progress(
        100, 100,
        description=f"Batch STAPLE complete: {results['completed']}/{total_requests} successful"
    )
    
    results['success'] = results['completed'] > 0
    return results


__all__ = ['convert_series_to_nifti', 'compute_staple_task', 'compute_batch_staple_task']
