"""
Celery tasks for the app.
"""

import logging
import time
from celery import shared_task, chain, group, chord
from celery.exceptions import SoftTimeLimitExceeded
from celery_progress.backend import ProgressRecorder

from app.models import DICOMSeries
from app.utils.dcm_to_nifti_converter import convert_series_with_rtstructs
from app.utils.spatial_overlap_metrics import compute_spatial_overlap_metrics

logger = logging.getLogger(__name__)


@shared_task(bind=True, time_limit=60*60*3, soft_time_limit=60*60*2.5)
def convert_series_to_nifti_chunked(self, series_ids, start_index=0, accumulated_results=None):
    """
    Celery task to convert DICOM series to NIfTI with chunking support.
    Can handle timeouts by spawning continuation tasks.
    
    Args:
        series_ids: List of DICOMSeries IDs to convert
        start_index: Index to start processing from (for continuation)
        accumulated_results: Results from previous chunks (for continuation)
        
    Returns:
        Dictionary with conversion results for all series
    """
    progress_recorder = ProgressRecorder(self)
    total_series = len(series_ids)
    start_time = time.time()
    
    # Initialize or merge results
    if accumulated_results is None:
        results = {
            'success': True,
            'total_series': total_series,
            'processed_series': 0,
            'failed_series': 0,
            'series_results': [],
            'errors': []
        }
    else:
        results = accumulated_results
        logger.info(f"Continuing from index {start_index}, already processed {results['processed_series']} series")
    
    def progress_callback(pct, message):
        """Update progress for Celery."""
        overall_pct = int((results['processed_series'] / total_series) * 100 + (pct / total_series))
        progress_recorder.set_progress(overall_pct, 100, description=message)
    
    # Process series starting from start_index
    remaining_series = series_ids[start_index:]
    
    try:
        for local_idx, series_id in enumerate(remaining_series):
            # Check if we're approaching time limit (leave 5 minutes buffer)
            elapsed_time = time.time() - start_time
            if elapsed_time > (60 * 60 * 2):  # 2 hours elapsed
                logger.warning(f"Approaching time limit at {elapsed_time}s, spawning continuation task")
                # Spawn continuation task for remaining series
                remaining_ids = series_ids[start_index + local_idx:]
                if remaining_ids:
                    logger.info(f"Spawning continuation task for {len(remaining_ids)} remaining series")
                    convert_series_to_nifti_chunked.apply_async(
                        args=[remaining_ids, 0, results],
                        countdown=5
                    )
                return results
            
            try:
                idx = start_index + local_idx + 1
                series = DICOMSeries.objects.get(id=series_id)
                base_message = f"Converting series {idx}/{total_series}: {series.series_instance_uid[:20]}..."
                progress_recorder.set_progress(
                    int((results['processed_series'] / total_series) * 100),
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
    
    except SoftTimeLimitExceeded:
        logger.warning("Soft time limit exceeded, spawning continuation task")
        # Spawn continuation task for remaining series
        current_idx = start_index + local_idx
        remaining_ids = series_ids[current_idx:]
        if remaining_ids:
            logger.info(f"Spawning continuation task for {len(remaining_ids)} remaining series")
            convert_series_to_nifti_chunked.apply_async(
                args=[remaining_ids, 0, results],
                countdown=5
            )
        return results
    
    # Final progress update
    progress_recorder.set_progress(100, 100, description="NIfTI conversion complete!")
    
    # Set overall success flag
    if results['failed_series'] > 0:
        results['success'] = results['processed_series'] > 0
    
    return results


@shared_task(bind=True)
def convert_series_to_nifti(self, series_ids):
    """
    Wrapper task that calls the chunked version.
    Maintains backward compatibility with existing code.
    
    Args:
        series_ids: List of DICOMSeries IDs to convert
        
    Returns:
        Dictionary with conversion results for all series
    """
    return convert_series_to_nifti_chunked(series_ids, start_index=0, accumulated_results=None)


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
def compute_single_spatial_overlap(self, pair_data):
    """
    Celery task to compute spatial overlap metrics for a single ROI pair.
    This task is designed to run in parallel with other instances.
    
    Args:
        pair_data: Dict with keys:
            - reference_roi_id: ID of reference ROI
            - target_roi_id: ID of target ROI
            - reference_roi_name: Name of reference ROI (for display)
            - target_roi_name: Name of target ROI (for display)
            - pair_index: Index of this pair (for logging)
            - total_pairs: Total number of pairs (for logging)
        
    Returns:
        Dictionary with computation result for this pair
    """
    reference_roi_id = int(pair_data['reference_roi_id'])
    target_roi_id = int(pair_data['target_roi_id'])
    reference_roi_name = pair_data.get('reference_roi_name', 'Unknown')
    target_roi_name = pair_data.get('target_roi_name', 'Unknown')
    pair_index = pair_data.get('pair_index', 0)
    total_pairs = pair_data.get('total_pairs', 1)
    
    logger.info(f"Computing pair {pair_index}/{total_pairs}: ROI {reference_roi_id} vs {target_roi_id}")
    
    try:
        # Compute metrics
        metrics = compute_spatial_overlap_metrics(
            reference_roi_id=reference_roi_id,
            target_roi_id=target_roi_id,
            save_to_db=True
        )
        
        result = {
            'reference_roi_id': reference_roi_id,
            'reference_roi_name': reference_roi_name,
            'target_roi_id': target_roi_id,
            'target_roi_name': target_roi_name,
            'metrics': metrics,
            'success': metrics.get('error') is None,
            'pair_index': pair_index
        }
        
        if metrics.get('error'):
            logger.error(f"Pair {pair_index}/{total_pairs} failed: {metrics['error']}")
        else:
            logger.info(f"Completed pair {pair_index}/{total_pairs}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error computing pair {pair_index}/{total_pairs}: {str(e)}"
        logger.error(error_msg)
        return {
            'reference_roi_id': reference_roi_id,
            'reference_roi_name': reference_roi_name,
            'target_roi_id': target_roi_id,
            'target_roi_name': target_roi_name,
            'metrics': {
                'error': str(e),
                'DSC': None,
                'HD95': None,
                'APL': None,
                'MSD': None,
                'OMDC': None,
                'UMDC': None
            },
            'success': False,
            'pair_index': pair_index
        }


@shared_task
def collect_spatial_overlap_results(results):
    """
    Callback task to collect and aggregate results from parallel spatial overlap computations.
    Used as the callback in a chord.
    
    Args:
        results: List of result dictionaries from individual pair computations
        
    Returns:
        Dictionary with aggregated computation results
    """
    total_pairs = len(results)
    completed = 0
    failed = 0
    errors = []
    
    for res in results:
        if res['success']:
            completed += 1
        else:
            failed += 1
            errors.append(f"Pair {res['pair_index']}: {res['metrics'].get('error', 'Unknown error')}")
    
    # Sort results by pair_index to maintain order
    results.sort(key=lambda x: x.get('pair_index', 0))
    
    logger.info(f"Parallel spatial overlap computation complete: {completed} successful, {failed} failed")
    
    return {
        'success': True,
        'total_pairs': total_pairs,
        'completed': completed,
        'failed': failed,
        'pair_results': results,
        'errors': errors
    }


@shared_task(bind=True)
def compute_spatial_overlap_task_parallel(self, roi_pairs, batch_size=4):
    """
    Celery task to compute spatial overlap metrics for multiple ROI pairs in parallel.
    Uses chord to execute pairs in parallel and collect results asynchronously.
    
    Args:
        roi_pairs: List of dicts with keys:
            - reference_roi_id: ID of reference ROI
            - target_roi_id: ID of target ROI
            - reference_roi_name: Name of reference ROI (for display)
            - target_roi_name: Name of target ROI (for display)
        batch_size: Number of pairs to process in parallel (default: 4)
        
    Returns:
        Dictionary with computation results for all pairs
    """
    progress_recorder = ProgressRecorder(self)
    total_pairs = len(roi_pairs)
    
    logger.info(f"Starting parallel spatial overlap computation for {total_pairs} ROI pairs (batch_size={batch_size})")
    
    # Prepare pair data with indices
    pairs_with_indices = []
    for idx, pair in enumerate(roi_pairs, 1):
        pair_data = pair.copy()
        pair_data['pair_index'] = idx
        pair_data['total_pairs'] = total_pairs
        pairs_with_indices.append(pair_data)
    
    # Update initial progress
    progress_recorder.set_progress(
        0,
        100,
        description=f"Starting parallel computation for {total_pairs} pairs..."
    )
    
    # Create chord: group of parallel tasks with a callback to collect results
    # This avoids calling .get() within a task
    job = chord(
        (compute_single_spatial_overlap.s(pair_data) for pair_data in pairs_with_indices),
        collect_spatial_overlap_results.s()
    )
    
    # Apply async and return the chord result
    # The chord will execute all pairs in parallel (up to worker concurrency)
    # and then call the callback to aggregate results
    result = job.apply_async()
    
    # Since we can't wait for results in a task, we return a reference
    # The frontend will poll for completion using the task_id
    logger.info(f"Chord created for {total_pairs} pairs, task will complete asynchronously")
    
    # Return immediately - the chord will handle completion
    return {
        'success': True,
        'total_pairs': total_pairs,
        'message': f'Processing {total_pairs} pairs in parallel',
        'chord_id': result.id
    }


@shared_task(bind=True)
def compute_spatial_overlap_task(self, roi_pairs):
    """
    Celery task to compute spatial overlap metrics for multiple ROI pairs.
    Uses sequential processing (kept for backward compatibility).
    For better performance, use compute_spatial_overlap_task_parallel instead.
    
    Args:
        roi_pairs: List of dicts with keys:
            - reference_roi_id: ID of reference ROI
            - target_roi_id: ID of target ROI
            - reference_roi_name: Name of reference ROI (for display)
            - target_roi_name: Name of target ROI (for display)
        
    Returns:
        Dictionary with computation results for all pairs
    """
    progress_recorder = ProgressRecorder(self)
    total_pairs = len(roi_pairs)
    
    results = {
        'success': True,
        'total_pairs': total_pairs,
        'completed': 0,
        'failed': 0,
        'pair_results': [],
        'errors': []
    }
    
    logger.info(f"Starting spatial overlap computation for {total_pairs} ROI pairs")
    
    for idx, pair in enumerate(roi_pairs, 1):
        try:
            reference_roi_id = int(pair['reference_roi_id'])
            target_roi_id = int(pair['target_roi_id'])
            reference_roi_name = pair.get('reference_roi_name', 'Unknown')
            target_roi_name = pair.get('target_roi_name', 'Unknown')
            
            # Update progress
            progress_pct = int((idx - 1) / total_pairs * 100)
            progress_recorder.set_progress(
                progress_pct,
                100,
                description=f"Computing pair {idx}/{total_pairs}: {reference_roi_name} vs {target_roi_name}"
            )
            
            logger.info(f"Computing pair {idx}/{total_pairs}: ROI {reference_roi_id} vs {target_roi_id}")
            
            # Compute metrics
            metrics = compute_spatial_overlap_metrics(
                reference_roi_id=reference_roi_id,
                target_roi_id=target_roi_id,
                save_to_db=True
            )
            
            results['pair_results'].append({
                'reference_roi_id': reference_roi_id,
                'reference_roi_name': reference_roi_name,
                'target_roi_id': target_roi_id,
                'target_roi_name': target_roi_name,
                'metrics': metrics,
                'success': metrics.get('error') is None
            })
            
            if metrics.get('error'):
                results['failed'] += 1
                results['errors'].append(f"Pair {idx}: {metrics['error']}")
            else:
                results['completed'] += 1
            
            logger.info(f"Completed pair {idx}/{total_pairs}")
            
        except Exception as e:
            error_msg = f"Error computing pair {idx}/{total_pairs}: {str(e)}"
            logger.error(error_msg)
            results['failed'] += 1
            results['errors'].append(error_msg)
            results['pair_results'].append({
                'reference_roi_id': pair.get('reference_roi_id'),
                'reference_roi_name': pair.get('reference_roi_name', 'Unknown'),
                'target_roi_id': pair.get('target_roi_id'),
                'target_roi_name': pair.get('target_roi_name', 'Unknown'),
                'metrics': {
                    'error': str(e),
                    'DSC': None,
                    'HD95': None,
                    'APL': None,
                    'MSD': None,
                    'OMDC': None,
                    'UMDC': None
                },
                'success': False
            })
    
    # Final progress update
    progress_recorder.set_progress(100, 100, description=f"Computation complete! {results['completed']}/{total_pairs} successful")
    
    logger.info(f"Spatial overlap computation complete: {results['completed']} successful, {results['failed']} failed")
    
    return results


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


@shared_task(bind=True)
def generate_visualization_task(self, image_series_id, roi_names, include_staple=True, 
                                window_center=None, window_width=None):
    """
    Celery task to generate visualizations for ROIs.
    Generates all slices for interactive viewing.
    
    Args:
        image_series_id: ID of the image series
        roi_names: List of ROI names to visualize
        include_staple: Whether to include STAPLE contours
        window_center: Window center for CT windowing
        window_width: Window width for CT windowing
        
    Returns:
        Dictionary with visualization results
    """
    from app.utils.nifti_visualizer import visualize_patient_rois
    
    progress_recorder = ProgressRecorder(self)
    
    progress_recorder.set_progress(10, 100, description="Starting visualization generation...")
    
    try:
        # Generate visualizations (all slices)
        visualizations = visualize_patient_rois(
            image_series_id=image_series_id,
            roi_names=roi_names,
            include_staple=include_staple,
            num_slices=None,  # Generate all slices
            window_center=window_center,
            window_width=window_width
        )
        
        progress_recorder.set_progress(100, 100, description="Visualization complete!")
        
        result = {
            'success': True,
            'visualizations': visualizations,
            'roi_count': len(visualizations),
            'errors': []
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Error generating visualizations: {str(e)}"
        logger.error(error_msg)
        
        result = {
            'success': False,
            'visualizations': {},
            'roi_count': 0,
            'errors': [error_msg]
        }
        
        return result


__all__ = ['convert_series_to_nifti', 'compute_staple_task', 'compute_batch_staple_task', 'generate_visualization_task']
