from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from app.models import DICOMFileArchive, Patient, DICOMStudy, DICOMSeries, DICOMInstance, RTStructROI
from app.utils.dicom_processor import process_dicom_archive
from app.utils.extract_roi_information import extract_roi_information
import os
import logging

logger = logging.getLogger(__name__)


def _delete_instance_files(instances):
    """Helper to delete files for a queryset or list of instances."""
    for instance in instances:
        if instance.instance_file_path:
            try:
                if os.path.exists(instance.instance_file_path):
                    os.remove(instance.instance_file_path)
            except Exception:
                pass  # Continue even if file deletion fails


def _get_patient_instances(patient):
    """Get all instances for a patient through the cascade."""
    return DICOMInstance.objects.filter(
        series__study__patient=patient
    )


def _get_study_instances(study):
    """Get all instances for a study through the cascade."""
    return DICOMInstance.objects.filter(
        series__study=study
    )


def _get_series_instances(series):
    """Get all instances for a series."""
    return DICOMInstance.objects.filter(series=series)


def home(request):
    """Home page view."""
    return render(request, "app/home.html")


def dicom_archive_list(request):
    """View to list all uploaded DICOM archives."""
    archives = DICOMFileArchive.objects.all().order_by("-created_at")
    return render(request, "app/archive_list.html", {"archives": archives})


def dicom_archive_upload(request):
    """View to upload a new DICOM zip file."""
    if request.method == "POST":
        if "file" in request.FILES:
            uploaded_file = request.FILES["file"]
            # Check if file is a zip
            if not uploaded_file.name.endswith(".zip"):
                messages.error(request, "Please upload a ZIP file containing DICOM files.")
                return redirect("dicom_archive_upload")
            
            # Save the file
            archive = DICOMFileArchive(file=uploaded_file)
            archive.save()
            messages.success(request, f"File '{uploaded_file.name}' uploaded successfully.")
            return redirect("dicom_archive_list")
        else:
            messages.error(request, "Please select a file to upload.")
    
    return render(request, "app/archive_upload.html")


def dicom_archive_detail(request, pk):
    """View to show details of a specific DICOM archive and process it."""
    archive = get_object_or_404(DICOMFileArchive, pk=pk)
    return render(request, "app/archive_detail.html", {"archive": archive})


@require_POST
def dicom_archive_process(request, pk):
    """View to process a DICOM archive using Celery."""
    archive = get_object_or_404(DICOMFileArchive, pk=pk)
    
    # Enqueue the processing task using Celery
    task = process_dicom_archive.delay(archive_id=pk)
    
    # Return immediately with task_id for celery-progress
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": "Processing task queued successfully",
            "task_status": "queued",
            "task_id": task.id
        })
    
    messages.info(request, "DICOM processing task has been queued and will run in the background.")
    return redirect("dicom_archive_detail", pk=pk)


def dicom_archive_delete(request, pk):
    """View to delete a DICOM archive."""
    archive = get_object_or_404(DICOMFileArchive, pk=pk)
    
    if request.method == "POST":
        archive_name = archive.file.name
        archive.delete()
        messages.success(request, f"Archive '{archive_name}' deleted successfully.")
        return redirect("dicom_archive_list")
    
    return render(request, "app/archive_confirm_delete.html", {"archive": archive})


def patient_list(request):
    """View to list all patients."""
    patients = Patient.objects.all().order_by("-created_at")
    return render(request, "app/patient_list.html", {"patients": patients})


def patient_detail(request, pk):
    """View to show patient details with their studies."""
    patient = get_object_or_404(Patient, pk=pk)
    studies = DICOMStudy.objects.filter(patient=patient).prefetch_related("dicomseries_set")
    return render(request, "app/patient_detail.html", {"patient": patient, "studies": studies})


def study_detail(request, pk):
    """View to show study details with series."""
    study = get_object_or_404(DICOMStudy, pk=pk)
    series_list = DICOMSeries.objects.filter(study=study).prefetch_related("dicominstance_set")
    return render(request, "app/study_detail.html", {"study": study, "series_list": series_list})


def rtstruct_list(request):
    """View to list all RT Structure Set series with their instances."""
    from django.db.models import Q
    
    # Get search/filter parameters
    search_patient_id = request.GET.get('patient_id', '').strip()
    search_patient_name = request.GET.get('patient_name', '').strip()
    search_study_date = request.GET.get('study_date', '').strip()
    search_modality = request.GET.get('modality', '').strip()
    
    # Build base queryset
    rtstruct_series_qs = DICOMSeries.objects.filter(
        modality='RTSTRUCT'
    ).select_related('study', 'study__patient').prefetch_related('dicominstance_set')
    
    # Apply filters
    if search_patient_id:
        rtstruct_series_qs = rtstruct_series_qs.filter(
            study__patient__patient_id__icontains=search_patient_id
        )
    
    if search_patient_name:
        rtstruct_series_qs = rtstruct_series_qs.filter(
            study__patient__patient_name__icontains=search_patient_name
        )
    
    if search_study_date:
        rtstruct_series_qs = rtstruct_series_qs.filter(
            study__study_date=search_study_date
        )
    
    if search_modality:
        rtstruct_series_qs = rtstruct_series_qs.filter(
            modality__icontains=search_modality
        )
    
    rtstruct_series = rtstruct_series_qs
    
    return render(request, "app/rtstruct_list.html", {
        "rtstruct_series": rtstruct_series,
        "search_patient_id": search_patient_id,
        "search_patient_name": search_patient_name,
        "search_study_date": search_study_date,
        "search_modality": search_modality,
    })


@require_POST
def rtstruct_extract(request):
    """View to extract ROI information from selected RTSTRUCT instances using Celery."""
    instance_ids = request.POST.getlist('instance_ids')
    
    if not instance_ids:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "No RTSTRUCT instances selected."
            })
        messages.error(request, "No RTSTRUCT instances selected.")
        return redirect("rtstruct_list")
    
    # Convert to integers
    try:
        instance_ids = [int(iid) for iid in instance_ids]
    except ValueError:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Invalid instance IDs provided."
            })
        messages.error(request, "Invalid instance IDs provided.")
        return redirect("rtstruct_list")
    
    # Enqueue the extraction task using Celery
    task = extract_roi_information.delay(instance_ids=instance_ids)
    
    # Return immediately with task_id for celery-progress
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": "ROI extraction task queued successfully",
            "task_status": "queued",
            "task_id": task.id,
            "total_instances": len(instance_ids)
        })
    
    messages.info(request, f"ROI extraction task has been queued for {len(instance_ids)} instance(s).")
    return redirect("rtstruct_list")


def roi_list(request):
    """View to list all ROIs grouped by referenced image series (CT/MR) with associated RTStructs."""
    from django.db.models import Count, Q
    
    # Get search/filter parameters
    search_patient_id = request.GET.get('patient_id', '').strip()
    search_patient_name = request.GET.get('patient_name', '').strip()
    search_study_date = request.GET.get('study_date', '').strip()
    search_modality = request.GET.get('modality', '').strip()
    
    # Build base queryset for referenced series
    referenced_series_qs = DICOMSeries.objects.filter(
        referenced_series_uid__isnull=False  # RTStruct instances reference this series
    ).select_related(
        'study', 
        'study__patient'
    )
    
    # Apply filters
    if search_patient_id:
        referenced_series_qs = referenced_series_qs.filter(
            study__patient__patient_id__icontains=search_patient_id
        )
    
    if search_patient_name:
        referenced_series_qs = referenced_series_qs.filter(
            study__patient__patient_name__icontains=search_patient_name
        )
    
    if search_study_date:
        referenced_series_qs = referenced_series_qs.filter(
            study__study_date=search_study_date
        )
    
    if search_modality:
        referenced_series_qs = referenced_series_qs.filter(
            modality__icontains=search_modality
        )
    
    # Get all series that are referenced by RTStruct instances (these are CT/MR image series)
    referenced_series = referenced_series_qs.annotate(
        rtstruct_count=Count('referenced_series_uid__series', distinct=True)
    ).distinct().order_by(
        'study__patient__patient_id',
        'series_instance_uid'
    )
    
    # For each referenced series, get the RTStruct series that reference it
    series_with_rtstructs = []
    for ref_series in referenced_series:
        # Get all RTStruct series that have instances referencing this series
        rtstruct_series_qs = DICOMSeries.objects.filter(
            modality='RTSTRUCT',
            dicominstance__referenced_series_instance_uid=ref_series
        ).select_related(
            'study',
            'study__patient'
        ).annotate(
            roi_count=Count('dicominstance__rtstructroi')
        ).distinct()
        
        # Add structure_set_label from the first instance of each series
        rtstruct_series = []
        for series in rtstruct_series_qs:
            first_instance = series.dicominstance_set.first()
            series.structure_set_label = first_instance.structure_set_label if first_instance else None
            rtstruct_series.append(series)
        
        series_with_rtstructs.append({
            'referenced_series': ref_series,
            'rtstruct_series': rtstruct_series
        })
    
    return render(request, "app/roi_list.html", {
        "series_with_rtstructs": series_with_rtstructs,
        "search_patient_id": search_patient_id,
        "search_patient_name": search_patient_name,
        "search_study_date": search_study_date,
        "search_modality": search_modality,
    })


def roi_detail(request, series_id):
    """View to show all ROIs for a specific structure set (series), grouped by referenced series."""
    series = get_object_or_404(
        DICOMSeries.objects.select_related('study', 'study__patient'),
        id=series_id,
        modality='RTSTRUCT'
    )
    
    # Get all ROIs for this series with their instance and referenced series info
    rois = RTStructROI.objects.filter(
        instance__series=series
    ).select_related(
        'instance',
        'instance__referenced_series_instance_uid'
    ).order_by('instance__instance_number', 'roi_number')
    
    # Group ROIs by referenced series instance UID
    rois_by_ref_series = {}
    for roi in rois:
        ref_series = roi.instance.referenced_series_instance_uid
        ref_series_uid = ref_series.series_instance_uid if ref_series else 'Unknown'
        ref_series_key = ref_series.id if ref_series else None
        
        if ref_series_key not in rois_by_ref_series:
            rois_by_ref_series[ref_series_key] = {
                'series': ref_series,
                'series_uid': ref_series_uid,
                'rois': []
            }
        rois_by_ref_series[ref_series_key]['rois'].append(roi)
    
    return render(request, "app/roi_detail.html", {
        "series": series, 
        "rois_by_ref_series": rois_by_ref_series
    })


@require_POST
def patient_delete_multiple(request):
    """Delete multiple patients and all their associated data including files."""
    patient_ids = request.POST.getlist('patient_ids')
    
    if not patient_ids:
        messages.warning(request, "No patients selected for deletion.")
        return redirect("patient_list")
    
    deleted_count = 0
    error_count = 0
    
    for patient_id in patient_ids:
        try:
            patient = Patient.objects.get(pk=patient_id)
            # First delete all associated files
            instances = _get_patient_instances(patient)
            _delete_instance_files(instances)
            # Then delete the patient (cascades to database records)
            patient.delete()
            deleted_count += 1
        except Patient.DoesNotExist:
            error_count += 1
        except Exception as e:
            error_count += 1
            messages.error(request, f"Error deleting patient {patient_id}: {str(e)}")
    
    if deleted_count > 0:
        messages.success(request, f"Successfully deleted {deleted_count} patient(s) and all their files.")
    if error_count > 0:
        messages.warning(request, f"Failed to delete {error_count} patient(s).")
    
    return redirect("patient_list")


@require_POST
def patient_delete(request, pk):
    """Delete a patient and all associated data including files."""
    patient = get_object_or_404(Patient, pk=pk)
    patient_name = patient.patient_id
    
    try:
        # First delete all associated files
        instances = _get_patient_instances(patient)
        _delete_instance_files(instances)
        # Then delete the patient (cascades to database records)
        patient.delete()
        messages.success(request, f"Patient '{patient_name}' and all associated files deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting patient: {str(e)}")
    
    return redirect("patient_list")


@require_POST
def study_delete(request, pk):
    """Delete a study and all associated series, instances, and files."""
    study = get_object_or_404(DICOMStudy, pk=pk)
    study_uid = study.study_instance_uid[:20]
    patient_pk = study.patient.pk
    
    try:
        # First delete all associated files
        instances = _get_study_instances(study)
        _delete_instance_files(instances)
        # Then delete the study (cascades to series and instances in database)
        study.delete()
        messages.success(request, f"Study '{study_uid}...' and all associated files deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting study: {str(e)}")
    
    return redirect("patient_detail", pk=patient_pk)


@require_POST
def series_delete(request, pk):
    """Delete a series and all associated instances with their files."""
    series = get_object_or_404(DICOMSeries, pk=pk)
    series_uid = series.series_instance_uid[:20]
    study_pk = series.study.pk
    
    try:
        # First delete all associated files
        instances = _get_series_instances(series)
        _delete_instance_files(instances)
        # Then delete the series (cascades to instances in database)
        series.delete()
        messages.success(request, f"Series '{series_uid}...' and all associated files deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting series: {str(e)}")
    
    return redirect("study_detail", pk=study_pk)


@require_POST
def instance_delete(request, pk):
    """Delete an instance and its associated DICOM file."""
    instance = get_object_or_404(DICOMInstance, pk=pk)
    sop_uid = instance.sop_instance_uid[:20]
    series_pk = instance.series.pk
    
    try:
        # Delete the DICOM file
        if instance.instance_file_path:
            import os
            try:
                if os.path.exists(instance.instance_file_path):
                    os.remove(instance.instance_file_path)
            except Exception as e:
                messages.warning(request, f"Could not delete file: {str(e)}")
        
        instance.delete()
        messages.success(request, f"Instance '{sop_uid}...' deleted successfully.")
    except Exception as e:
        messages.error(request, f"Error deleting instance: {str(e)}")
    
    return redirect("study_detail", pk=instance.series.study.pk)


@require_POST
def nifti_convert(request):
    """View to trigger NIfTI conversion for selected series using Celery."""
    from app.tasks import convert_series_to_nifti
    
    series_ids = request.POST.getlist('series_ids')
    
    if not series_ids:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "No series selected for conversion"
            })
        messages.error(request, "Please select at least one series to convert.")
        return redirect("roi_list")
    
    # Validate that all series IDs are valid image series (not RTSTRUCT)
    valid_series = DICOMSeries.objects.filter(
        id__in=series_ids
    ).exclude(modality='RTSTRUCT')
    
    valid_series_ids = list(valid_series.values_list('id', flat=True))
    
    if not valid_series_ids:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "No valid image series selected (RTStruct series cannot be converted directly)"
            })
        messages.error(request, "No valid image series selected.")
        return redirect("roi_list")
    
    # Enqueue the conversion task using Celery
    task = convert_series_to_nifti.delay(valid_series_ids)
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": f"NIfTI conversion queued for {len(valid_series_ids)} series",
            "task_status": "queued",
            "task_id": task.id,
            "series_count": len(valid_series_ids)
        })
    
    messages.info(request, f"NIfTI conversion task queued for {len(valid_series_ids)} series. This will run in the background.")
    return redirect("roi_list")


def nifti_list(request):
    """View to list all patients with their NIfTI structure sets and ROI information."""
    import json
    from pathlib import Path
    from collections import defaultdict, Counter
    
    # Get all patients who have any NIfTI data (either image or RTStruct)
    patients_with_nifti = Patient.objects.filter(
        dicomstudy__dicomseries__nifti_file_path__isnull=False
    ).exclude(
        dicomstudy__dicomseries__nifti_file_path=''
    ).distinct().order_by('patient_id')
    
    patients_data = []
    
    for patient in patients_with_nifti:
        # Collect all image series across all studies for this patient
        all_image_series = []
        all_structure_sets = []
        all_roi_names = []
        
        # Get all studies for this patient that have NIfTI data
        studies = DICOMStudy.objects.filter(
            patient=patient,
            dicomseries__nifti_file_path__isnull=False
        ).exclude(
            dicomseries__nifti_file_path=''
        ).distinct()
        
        for study in studies:
            # Get image series with NIfTI (exclude RTSTRUCT - only get CT/MR/PT/etc.)
            image_series_list = DICOMSeries.objects.filter(
                study=study,
                nifti_file_path__isnull=False
            ).exclude(
                nifti_file_path=''
            ).exclude(
                modality='RTSTRUCT'
            )
            
            for img_series in image_series_list:
                all_image_series.append({
                    'series': img_series,
                    'study': study
                })
                
                # Find all RTStruct series that reference this image series and have NIfTI
                rtstruct_series = DICOMSeries.objects.filter(
                    modality='RTSTRUCT',
                    dicominstance__referenced_series_instance_uid=img_series,
                    nifti_file_path__isnull=False
                ).exclude(
                    nifti_file_path=''
                ).distinct()
                
                for rtstruct in rtstruct_series:
                    metadata_path = Path(settings.MEDIA_ROOT) / rtstruct.nifti_file_path / "rtstruct_metadata.json"
                    roi_count = 0
                    roi_names = []
                    
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                roi_count = metadata.get('converted_count', 0)
                                roi_names = [roi['name'] for roi in metadata.get('rois', [])]
                                all_roi_names.extend(roi_names)
                        except Exception:
                            pass
                    
                    all_structure_sets.append({
                        'series': rtstruct,
                        'series_id': rtstruct.id,
                        'roi_count': roi_count,
                        'roi_names': roi_names,
                        'roi_names_with_staple': [],  # Will be populated later
                        'series_uid_short': rtstruct.series_instance_uid[:30] + '...' if len(rtstruct.series_instance_uid) > 30 else rtstruct.series_instance_uid,
                        'image_series': img_series,
                        'study': study
                    })
        
        # Find common ROIs (appearing in 2+ structure sets)
        roi_counter = Counter(all_roi_names)
        common_rois = [
            {
                'name': name,
                'count': count
            }
            for name, count in roi_counter.items()
            if count >= 2
        ]
        common_rois.sort(key=lambda x: (-x['count'], x['name']))
        
        # Check for STAPLE contours using database relationships
        # Get all ROI names that have STAPLE computed for this patient
        staple_rois = set()
        
        # Query RTStructROI entries that belong to this patient and have staple_roi set
        patient_rtstruct_rois = RTStructROI.objects.filter(
            instance__series__study__patient=patient,
            staple_roi__isnull=False
        ).values_list('roi_name', flat=True).distinct()
        
        staple_rois = set(patient_rtstruct_rois)
        
        # Mark which ROIs have STAPLE in each structure set
        for ss in all_structure_sets:
            ss['roi_names_with_staple'] = [
                {'name': roi_name, 'has_staple': roi_name in staple_rois}
                for roi_name in ss['roi_names']
            ]
        
        # Only add patient if they have structure sets
        if all_structure_sets:
            patients_data.append({
                'patient': patient,
                'image_series_list': all_image_series,
                'structure_sets': all_structure_sets,
                'total_structure_sets': len(all_structure_sets),
                'common_rois': common_rois,
                'total_unique_rois': len(roi_counter),
                'total_staple_rois': len(staple_rois),
                'staple_rois': sorted(list(staple_rois))
            })
    
    return render(request, "app/nifti_list.html", {"patients_data": patients_data})


@require_POST
def compute_staple(request):
    """View to trigger STAPLE contour computation using Celery."""
    from app.tasks import compute_staple_task
    import json
    
    # Get parameters from POST request
    image_series_id = request.POST.get('image_series_id')
    structure_name = request.POST.get('structure_name')
    rtstruct_series_ids = request.POST.getlist('rtstruct_series_ids[]')
    threshold = float(request.POST.get('threshold', 0.95))
    
    # Validate inputs
    if not image_series_id or not structure_name or not rtstruct_series_ids:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Missing required parameters"
            })
        messages.error(request, "Missing required parameters for STAPLE computation.")
        return redirect("nifti_list")
    
    # Convert to integers
    try:
        image_series_id = int(image_series_id)
        rtstruct_series_ids = [int(sid) for sid in rtstruct_series_ids]
    except ValueError:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Invalid series IDs"
            })
        messages.error(request, "Invalid series IDs.")
        return redirect("nifti_list")
    
    # Validate minimum number of structure sets
    if len(rtstruct_series_ids) < 2:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "At least 2 structure sets are required for STAPLE computation"
            })
        messages.error(request, "At least 2 structure sets are required for STAPLE computation.")
        return redirect("nifti_list")
    
    # Enqueue the STAPLE computation task
    task = compute_staple_task.delay(
        image_series_id=image_series_id,
        structure_name=structure_name,
        rtstruct_series_ids=rtstruct_series_ids,
        threshold=threshold
    )
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": f"STAPLE computation queued for structure '{structure_name}'",
            "task_status": "queued",
            "task_id": task.id,
            "structure_name": structure_name,
            "num_segmentations": len(rtstruct_series_ids)
        })
    
    messages.info(request, f"STAPLE computation task queued for structure '{structure_name}'. This will run in the background.")
    return redirect("nifti_list")


@require_POST
def compute_batch_staple(request):
    """View to trigger batch STAPLE contour computation for multiple ROIs across multiple patients."""
    from app.tasks import compute_batch_staple_task
    import json
    from collections import defaultdict
    from pathlib import Path
    
    # Get batch requests from POST data
    # New format: [{roi_name, threshold}, ...]
    try:
        batch_data = json.loads(request.POST.get('batch_data', '[]'))
    except json.JSONDecodeError:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Invalid batch data format"
            })
        messages.error(request, "Invalid batch data format.")
        return redirect("staple_computation")
    
    if not batch_data:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "No STAPLE computations selected"
            })
        messages.error(request, "No STAPLE computations selected.")
        return redirect("staple_computation")
    
    # Build STAPLE requests from ROI-centric data
    # For each ROI, find all patients where it appears in 2+ structure sets
    staple_requests = []
    
    for item in batch_data:
        try:
            roi_name = item['roi_name']
            threshold = float(item.get('threshold', 0.95))
            
            # Find all RTStruct series with NIfTI that contain this ROI
            rtstruct_series = DICOMSeries.objects.filter(
                modality='RTSTRUCT',
                nifti_file_path__isnull=False
            ).exclude(
                nifti_file_path=''
            ).select_related('study__patient').prefetch_related('dicominstance_set')
            
            # Group by patient and image series
            patient_image_rtstruct = defaultdict(lambda: defaultdict(list))
            
            for rtstruct in rtstruct_series:
                # Check if this RTStruct has the ROI
                metadata_path = Path(settings.MEDIA_ROOT) / rtstruct.nifti_file_path / "rtstruct_metadata.json"
                if not metadata_path.exists():
                    continue
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        roi_names = [roi['name'] for roi in metadata.get('rois', [])]
                        
                        if roi_name not in roi_names:
                            continue
                        
                        # Get the referenced image series
                        image_series = None
                        for instance in rtstruct.dicominstance_set.all():
                            if instance.referenced_series_instance_uid:
                                image_series = instance.referenced_series_instance_uid
                                break
                        
                        if not image_series:
                            continue
                        
                        patient_id = rtstruct.study.patient.patient_id
                        patient_image_rtstruct[patient_id][image_series.id].append(rtstruct.id)
                        
                except Exception as e:
                    logger.warning(f"Failed to read metadata for {rtstruct.series_instance_uid}: {e}")
                    continue
            
            # Create STAPLE requests for patients with 2+ structure sets
            for patient_id, image_series_dict in patient_image_rtstruct.items():
                for image_series_id, rtstruct_ids in image_series_dict.items():
                    if len(rtstruct_ids) >= 2:
                        staple_requests.append({
                            'image_series_id': image_series_id,
                            'structure_name': roi_name,
                            'rtstruct_series_ids': rtstruct_ids,
                            'threshold': threshold
                        })
                        
        except (KeyError, ValueError) as e:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return JsonResponse({
                    "success": False,
                    "error": f"Invalid request data: {str(e)}"
                })
            messages.error(request, f"Invalid request data: {str(e)}")
            return redirect("staple_computation")
    
    if not staple_requests:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "No eligible STAPLE computations found (need 2+ structure sets per patient)"
            })
        messages.error(request, "No eligible STAPLE computations found.")
        return redirect("staple_computation")
    
    # Enqueue the batch STAPLE computation task
    task = compute_batch_staple_task.delay(staple_requests)
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": f"Batch STAPLE computation queued for {len(staple_requests)} ROIs",
            "task_status": "queued",
            "task_id": task.id,
            "total_requests": len(staple_requests)
        })
    
    messages.info(request, f"Batch STAPLE computation task queued for {len(staple_requests)} ROIs. This will run in the background.")
    return redirect("staple_computation")


def batch_staple_status(request, task_id):
    """API endpoint to check the status of a batch STAPLE computation task."""
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id)
    
    response_data = {
        'state': task.state,
        'current': 0,
        'total': 0,
        'status': str(task.info),
        'complete': False
    }
    
    if task.state == 'PENDING':
        response_data.update({
            'current': 0,
            'total': 0,
            'status': 'Task is waiting to start...'
        })
    elif task.state == 'PROGRESS':
        info = task.info
        response_data.update({
            'current': info.get('current', 0),
            'total': info.get('total', 0),
            'status': info.get('description', 'Processing...')
        })
    elif task.state == 'SUCCESS':
        result = task.result
        response_data.update({
            'current': result.get('total_requests', 0),
            'total': result.get('total_requests', 0),
            'status': 'Complete!',
            'complete': True,
            'result': result
        })
    elif task.state == 'FAILURE':
        response_data.update({
            'status': 'Task failed',
            'complete': True,
            'error': str(task.info)
        })
    
    return JsonResponse(response_data)


def visualize_patient_series(request, series_id):
    """View to display visualization options for a patient's image series."""
    from app.models import DICOMSeries, DICOMInstance
    from pathlib import Path
    import json
    
    series = get_object_or_404(
        DICOMSeries.objects.select_related('study', 'study__patient'),
        id=series_id
    )
    
    # Check if NIfTI file exists
    if not series.nifti_file_path:
        messages.error(request, "NIfTI file not found for this series. Please convert to NIfTI first.")
        return redirect("nifti_list")
    
    # Find all RTStruct series that reference this image series
    rtstruct_series = DICOMSeries.objects.filter(
        modality='RTSTRUCT',
        dicominstance__referenced_series_instance_uid=series,
        nifti_file_path__isnull=False
    ).exclude(nifti_file_path='').distinct()
    
    # Collect all available ROIs with their structure sets
    all_rois = {}
    for rtstruct in rtstruct_series:
        nifti_dir = Path(settings.MEDIA_ROOT) / rtstruct.nifti_file_path
        metadata_path = nifti_dir / "rtstruct_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    for roi in metadata.get('rois', []):
                        roi_name = roi['name']
                        if roi_name not in all_rois:
                            all_rois[roi_name] = {
                                'name': roi_name,
                                'structure_sets': [],
                                'has_staple': False
                            }
                        
                        all_rois[roi_name]['structure_sets'].append({
                            'series_id': rtstruct.id,
                            'series_uid': rtstruct.series_instance_uid
                        })
            except Exception as e:
                logger.warning(f"Failed to read metadata for {rtstruct.id}: {e}")
    
    # Check for STAPLE contours
    from app.utils.dcm_to_nifti_converter import sanitize_for_path
    patient_id = sanitize_for_path(series.study.patient.patient_id)
    study_uid = sanitize_for_path(series.study.study_instance_uid)
    series_uid = sanitize_for_path(series.series_instance_uid)
    staple_dir = Path(settings.MEDIA_ROOT) / "nifti_files" / patient_id / study_uid / series_uid / "staple"
    
    if staple_dir.exists():
        for roi_name in all_rois.keys():
            safe_roi_name = sanitize_for_path(roi_name)
            staple_path = staple_dir / f"staple_{safe_roi_name}.nii.gz"
            if staple_path.exists():
                all_rois[roi_name]['has_staple'] = True
    
    # Sort ROIs by name
    rois_list = sorted(all_rois.values(), key=lambda x: x['name'])
    
    return render(request, "app/visualize_series.html", {
        "series": series,
        "rois": rois_list,
        "patient": series.study.patient,
        "study": series.study
    })


@require_POST
def generate_visualization(request):
    """Generate visualization for selected ROIs using Celery."""
    from app.tasks import generate_visualization_task
    import json
    
    series_id = request.POST.get('series_id')
    roi_names = request.POST.getlist('roi_names[]')
    include_staple = request.POST.get('include_staple', 'true').lower() == 'true'
    window_center = request.POST.get('window_center', '')
    window_width = request.POST.get('window_width', '')
    
    # Parse windowing parameters
    try:
        window_center = float(window_center) if window_center else None
        window_width = float(window_width) if window_width else None
    except ValueError:
        window_center = None
        window_width = None
    
    if not series_id or not roi_names:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Missing required parameters"
            })
        messages.error(request, "Missing required parameters for visualization.")
        return redirect("nifti_list")
    
    try:
        series_id = int(series_id)
    except ValueError:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "Invalid series ID"
            })
        messages.error(request, "Invalid series ID.")
        return redirect("nifti_list")
    
    # Enqueue the visualization task (all slices will be generated)
    task = generate_visualization_task.delay(
        image_series_id=series_id,
        roi_names=roi_names,
        include_staple=include_staple,
        window_center=window_center,
        window_width=window_width
    )
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({
            "success": True,
            "message": f"Visualization task queued for {len(roi_names)} ROI(s)",
            "task_status": "queued",
            "task_id": task.id,
            "roi_count": len(roi_names)
        })
    
    messages.info(request, f"Visualization task queued for {len(roi_names)} ROI(s). This will run in the background.")
    return redirect("nifti_list")


def get_series_rois(request, series_id):
    """API endpoint to get all ROIs for a given image series."""
    import json
    from pathlib import Path
    
    try:
        image_series = get_object_or_404(DICOMSeries, id=series_id)
        
        # Find all RTStruct series that reference this image series
        rtstruct_series = DICOMSeries.objects.filter(
            modality='RTSTRUCT',
            dicominstance__referenced_series_instance_uid=image_series,
            nifti_file_path__isnull=False
        ).exclude(
            nifti_file_path=''
        ).distinct()
        
        # Collect all ROIs
        all_rois = {}
        
        for rtstruct in rtstruct_series:
            metadata_path = Path(settings.MEDIA_ROOT) / rtstruct.nifti_file_path / "rtstruct_metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        for roi in metadata.get('rois', []):
                            roi_name = roi['name']
                            if roi_name not in all_rois:
                                all_rois[roi_name] = {
                                    'name': roi_name,
                                    'count': 0,
                                    'has_staple': False
                                }
                            all_rois[roi_name]['count'] += 1
                except Exception as e:
                    logger.warning(f"Failed to read metadata: {e}")
        
        # Check for STAPLE contours using database relationships
        # Get ROI names that have STAPLE computed for this patient
        patient = image_series.study.patient
        staple_roi_names = set(RTStructROI.objects.filter(
            instance__series__study__patient=patient,
            staple_roi__isnull=False
        ).values_list('roi_name', flat=True).distinct())
        
        # Mark which ROIs have STAPLE
        for roi_name in all_rois.keys():
            if roi_name in staple_roi_names:
                all_rois[roi_name]['has_staple'] = True
        
        # Convert to list and sort by name
        rois_list = sorted(all_rois.values(), key=lambda x: x['name'])
        
        return JsonResponse({
            'success': True,
            'rois': rois_list,
            'total': len(rois_list)
        })
        
    except Exception as e:
        logger.error(f"Error fetching ROIs for series {series_id}: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'rois': []
        }, status=500)


def view_visualizations(request, series_id):
    """View to display generated visualizations for a series."""
    from app.models import DICOMSeries
    from pathlib import Path
    import glob
    
    series = get_object_or_404(
        DICOMSeries.objects.select_related('study', 'study__patient'),
        id=series_id
    )
    
    # Find all visualization images for this series
    vis_dir = Path(settings.MEDIA_ROOT) / "visualizations"
    visualizations = []
    
    if vis_dir.exists():
        # Look for visualization files
        # We'll need to store metadata about which visualizations belong to which series
        # For now, we'll list all recent visualizations
        vis_files = sorted(vis_dir.glob("vis_*.png"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for vis_file in vis_files[:20]:  # Show last 20 visualizations
            relative_path = vis_file.relative_to(settings.MEDIA_ROOT)
            visualizations.append({
                'path': str(relative_path),
                'filename': vis_file.name,
                'created': vis_file.stat().st_mtime
            })
    
    return render(request, "app/view_visualizations.html", {
        "series": series,
        "visualizations": visualizations,
        "patient": series.study.patient,
        "study": series.study
    })


def staple_computation(request):
    """
    STAPLE computation page - ROI-centric view.
    Shows all ROIs with NIfTI files, grouped by ROI name.
    Displays structure set count and patient count for each ROI.
    """
    import json
    from pathlib import Path
    from collections import defaultdict
    
    # Dictionary to store ROI information: roi_name -> {patients: set, structure_sets: list, image_series: set}
    roi_data = defaultdict(lambda: {
        'patients': set(),
        'structure_sets': [],
        'image_series': set(),
        'patient_structure_counts': defaultdict(int)  # patient_id -> count of structure sets with this ROI
    })
    
    # Get all RTSTRUCT series with NIfTI files
    rtstruct_series = DICOMSeries.objects.filter(
        modality='RTSTRUCT',
        nifti_file_path__isnull=False
    ).exclude(
        nifti_file_path=''
    ).select_related(
        'study__patient'
    ).prefetch_related(
        'dicominstance_set'
    )
    
    for rtstruct in rtstruct_series:
        # Read metadata to get ROI names
        metadata_path = Path(settings.MEDIA_ROOT) / rtstruct.nifti_file_path / "rtstruct_metadata.json"
        
        if not metadata_path.exists():
            continue
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                rois = metadata.get('rois', [])
                
                # Get the referenced image series
                image_series = None
                for instance in rtstruct.dicominstance_set.all():
                    if instance.referenced_series_instance_uid:
                        image_series = instance.referenced_series_instance_uid
                        break
                
                if not image_series:
                    continue
                
                patient = rtstruct.study.patient
                patient_id = patient.patient_id
                
                for roi in rois:
                    roi_name = roi['name']
                    
                    # Add patient
                    roi_data[roi_name]['patients'].add(patient_id)
                    
                    # Add structure set info
                    roi_data[roi_name]['structure_sets'].append({
                        'series_id': rtstruct.id,
                        'series_uid': rtstruct.series_instance_uid,
                        'patient_id': patient_id,
                        'patient_name': patient.patient_name,
                        'image_series_id': image_series.id,
                        'image_modality': image_series.modality
                    })
                    
                    # Add image series
                    roi_data[roi_name]['image_series'].add(image_series.id)
                    
                    # Count structure sets per patient for this ROI
                    roi_data[roi_name]['patient_structure_counts'][patient_id] += 1
                    
        except Exception as e:
            logger.warning(f"Failed to read metadata for {rtstruct.series_instance_uid}: {e}")
            continue
    
    # Convert to list format for template
    roi_list = []
    for roi_name, data in roi_data.items():
        # Check if STAPLE can be computed (ROI appears in multiple structure sets for at least one patient)
        can_compute_staple = any(count >= 2 for count in data['patient_structure_counts'].values())
        
        # Get patients where this ROI appears in multiple structure sets
        staple_eligible_patients = [
            patient_id for patient_id, count in data['patient_structure_counts'].items() if count >= 2
        ]
        
        # Check if STAPLE has been computed for this ROI (check database)
        # Get patients where STAPLE has been computed for this ROI
        staple_computed_patients = RTStructROI.objects.filter(
            roi_name=roi_name,
            staple_roi__isnull=False
        ).values_list('instance__series__study__patient__patient_id', flat=True).distinct()
        
        staple_computed_patients_list = list(staple_computed_patients)
        staple_computed_count = len(staple_computed_patients_list)
        
        # Determine status
        if staple_computed_count > 0:
            staple_status = 'computed'
            staple_status_label = f'Computed ({staple_computed_count} patient{"s" if staple_computed_count > 1 else ""})'
            staple_status_class = 'success'
        elif can_compute_staple:
            staple_status = 'eligible'
            staple_status_label = 'Not computed'
            staple_status_class = 'warning'
        else:
            staple_status = 'ineligible'
            staple_status_label = 'Not eligible'
            staple_status_class = 'secondary'
        
        roi_list.append({
            'name': roi_name,
            'patient_count': len(data['patients']),
            'structure_set_count': len(data['structure_sets']),
            'image_series_count': len(data['image_series']),
            'structure_sets': data['structure_sets'],
            'can_compute_staple': can_compute_staple,
            'staple_eligible_patients': staple_eligible_patients,
            'staple_eligible_count': len(staple_eligible_patients),
            'staple_status': staple_status,
            'staple_status_label': staple_status_label,
            'staple_status_class': staple_status_class,
            'staple_computed_count': staple_computed_count,
            'staple_computed_patients': staple_computed_patients_list
        })
    
    # Sort by ROI name
    roi_list.sort(key=lambda x: x['name'])
    
    return render(request, "app/staple_computation.html", {
        "roi_list": roi_list,
        "total_rois": len(roi_list)
    })


def spatial_overlap_metrics(request):
    """
    Spatial Overlap Metrics computation page - ROI name-based approach.
    Users select ROI names, and system auto-generates all pairwise combinations.
    """
    from app.utils.spatial_overlap_metrics import get_rois_for_series, get_roi_nifti_path
    from collections import defaultdict
    import json
    
    # Build ROI name-based data structure
    # roi_name -> {'regular_instances': [...], 'staple_instances': [...]}
    roi_data = defaultdict(lambda: {
        'roi_name': '',
        'regular_instances': [],
        'staple_instances': [],
        'total_instances': 0,
        'has_staple': False,
        'seen_staple_ids': set()  # Track STAPLE IDs per ROI name
    })
    
    # Get all image series with NIfTI files
    image_series_list = DICOMSeries.objects.filter(
        nifti_file_path__isnull=False
    ).exclude(
        nifti_file_path='',
        modality='RTSTRUCT'
    ).select_related('study__patient').order_by('study__patient__patient_id', 'modality')
    
    for img_series in image_series_list:
        # Get all ROIs for this series (already filtered for NIfTI availability)
        rois = get_rois_for_series(img_series.series_instance_uid)
        logger.info(f"Series {img_series.series_instance_uid[-8:]}: Found {len(rois)} ROIs")
        
        for roi in rois:
            # STAPLE ROIs have staple_roi set AND instance is None
            # Regular ROIs have instance set (even if they're linked to a STAPLE via staple_roi)
            roi_type = 'STAPLE' if (roi.staple_roi and not roi.instance) else 'RTStruct'
            logger.info(f"  ROI: {roi.roi_name} (ID: {roi.id}, Type: {roi_type}, instance: {roi.instance is not None}, staple_roi: {roi.staple_roi is not None})")
            roi_name = roi.roi_name
            
            # Normalize ROI name - remove STAPLE_ prefix for grouping
            # STAPLE ROIs are named "STAPLE_<structure_name>" but should group with "<structure_name>"
            base_roi_name = roi_name.replace('STAPLE_', '') if roi_name.startswith('STAPLE_') else roi_name
            
            # Get source structure set information
            if roi.instance:
                source_series = roi.instance.series
                source_series_uid_short = source_series.series_instance_uid[-8:]
                source_label = f"RTStruct (...{source_series_uid_short})"
            elif roi.staple_roi:
                source_label = 'STAPLE Consensus'
            else:
                source_label = 'Unknown'
            
            instance_data = {
                'roi_id': roi.id,
                'roi_name': roi_name,  # Keep original name for display
                'roi_type': roi_type,
                'series_id': img_series.id,
                'series_uid': img_series.series_instance_uid,
                'series_modality': img_series.modality,
                'patient_id': img_series.study.patient.patient_id,
                'patient_name': img_series.study.patient.patient_name or '',
                'source_label': source_label
            }
            
            # Use base_roi_name for grouping
            roi_data[base_roi_name]['roi_name'] = base_roi_name
            
            if roi_type == 'STAPLE':
                # Only add STAPLE instance once per ROI name (deduplicate by ROI ID)
                if roi.id not in roi_data[base_roi_name]['seen_staple_ids']:
                    roi_data[base_roi_name]['seen_staple_ids'].add(roi.id)
                    roi_data[base_roi_name]['staple_instances'].append(instance_data)
                    roi_data[base_roi_name]['has_staple'] = True
                    roi_data[base_roi_name]['total_instances'] += 1
                    logger.info(f"    Added STAPLE instance for {base_roi_name}, total STAPLE: {len(roi_data[base_roi_name]['staple_instances'])}")
                else:
                    logger.info(f"    Skipped duplicate STAPLE ID {roi.id} for {base_roi_name}")
            else:
                roi_data[base_roi_name]['regular_instances'].append(instance_data)
                roi_data[base_roi_name]['total_instances'] += 1
                logger.info(f"    Added regular instance for {base_roi_name}, total regular: {len(roi_data[base_roi_name]['regular_instances'])}")
    
    # Convert to lists and sort
    roi_list = []
    roi_list_json = []
    
    logger.info("=== FINAL ROI DATA SUMMARY ===")
    for roi_name, data in roi_data.items():
        logger.info(f"{roi_name}: {len(data['regular_instances'])} regular, {len(data['staple_instances'])} STAPLE")
        roi_list.append(data)
        roi_list_json.append({
            'roi_name': roi_name,
            'regular_instances': data['regular_instances'],
            'staple_instances': data['staple_instances'],
            'total_instances': data['total_instances'],
            'has_staple': data['has_staple']
        })
    
    # Sort by ROI name
    roi_list.sort(key=lambda x: x['roi_name'])
    roi_list_json.sort(key=lambda x: x['roi_name'])
    
    # Generate automatic pair suggestions
    suggested_pairs = []
    
    for roi_name, data in roi_data.items():
        regular_instances = data['regular_instances']
        staple_instances = data['staple_instances']
        
        # Group regular instances by series_id to find multiple structure sets for same series
        series_groups = defaultdict(list)
        for inst in regular_instances:
            series_groups[inst['series_id']].append(inst)
        
        # Type 1: Multiple structure sets with same ROI for same image series
        for series_id, instances in series_groups.items():
            if len(instances) > 1:
                # Generate pairwise combinations (1vs2, 1vs3, 2vs3, etc.)
                for i in range(len(instances)):
                    for j in range(i + 1, len(instances)):
                        suggested_pairs.append({
                            'reference_roi_id': instances[i]['roi_id'],
                            'reference_roi_name': instances[i]['roi_name'],
                            'reference_roi_type': instances[i]['roi_type'],
                            'target_roi_id': instances[j]['roi_id'],
                            'target_roi_name': instances[j]['roi_name'],
                            'target_roi_type': instances[j]['roi_type'],
                            'series_id': instances[i]['series_id'],
                            'series_modality': instances[i]['series_modality'],
                            'patient_id': instances[i]['patient_id'],
                            'patient_name': instances[i]['patient_name'],
                            'reference_source': instances[i]['source_label'],
                            'target_source': instances[j]['source_label'],
                            'suggestion_type': 'multiple_structuresets'
                        })
        
        # Type 2: STAPLE vs all regular instances for same series
        if staple_instances:
            for staple_inst in staple_instances:
                staple_series_id = staple_inst['series_id']
                # Find all regular instances for the same series
                matching_regular = [inst for inst in regular_instances if inst['series_id'] == staple_series_id]
                for regular_inst in matching_regular:
                    suggested_pairs.append({
                        'reference_roi_id': staple_inst['roi_id'],
                        'reference_roi_name': staple_inst['roi_name'],
                        'reference_roi_type': staple_inst['roi_type'],
                        'target_roi_id': regular_inst['roi_id'],
                        'target_roi_name': regular_inst['roi_name'],
                        'target_roi_type': regular_inst['roi_type'],
                        'series_id': staple_inst['series_id'],
                        'series_modality': staple_inst['series_modality'],
                        'patient_id': staple_inst['patient_id'],
                        'patient_name': staple_inst['patient_name'],
                        'reference_source': staple_inst['source_label'],
                        'target_source': regular_inst['source_label'],
                        'suggestion_type': 'staple_vs_regular'
                    })
    
    logger.info(f"Generated {len(suggested_pairs)} automatic pair suggestions")
    
    return render(request, "app/spatial_overlap_metrics.html", {
        "roi_list": roi_list,
        "roi_data_json": json.dumps(roi_list_json),
        "suggested_pairs_json": json.dumps(suggested_pairs),
        "total_roi_names": len(roi_list),
        "total_suggested_pairs": len(suggested_pairs)
    })


@require_POST
def compute_overlap_metrics(request):
    """API endpoint to trigger Celery task for computing spatial overlap metrics."""
    from app.tasks import compute_spatial_overlap_task_parallel
    import json
    
    try:
        # Get parameters from POST request
        roi_pairs_json = request.POST.get('roi_pairs')
        batch_size = int(request.POST.get('batch_size', 4))  # Default to 4 parallel tasks
        
        if not roi_pairs_json:
            return JsonResponse({
                "success": False,
                "error": "No ROI pairs provided"
            })
        
        roi_pairs = json.loads(roi_pairs_json)
        
        if not roi_pairs or len(roi_pairs) == 0:
            return JsonResponse({
                "success": False,
                "error": "No ROI pairs selected"
            })
        
        # Trigger parallel Celery task
        logger.info(f"Triggering parallel Celery task for {len(roi_pairs)} ROI pairs (batch_size={batch_size})")
        task = compute_spatial_overlap_task_parallel.delay(roi_pairs, batch_size=batch_size)
        
        return JsonResponse({
            "success": True,
            "task_id": task.id,
            "message": f"Started computation for {len(roi_pairs)} ROI pair(s)",
            "total_pairs": len(roi_pairs)
        })
        
    except Exception as e:
        logger.error(f"Error starting overlap metrics task: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


def spatial_overlap_results(request, task_id):
    """Display spatial overlap computation results for a completed task."""
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id)
    
    context = {
        'task_id': task_id,
        'task_state': task.state,
        'task_ready': task.ready(),
    }
    
    if task.ready():
        if task.successful():
            result = task.result
            context['success'] = True
            context['results'] = result.get('pair_results', [])
            context['total_pairs'] = result.get('total_pairs', 0)
            context['completed'] = result.get('completed', 0)
            context['failed'] = result.get('failed', 0)
            context['errors'] = result.get('errors', [])
        else:
            context['success'] = False
            context['error'] = str(task.result) if task.result else 'Unknown error'
    
    return render(request, 'app/spatial_overlap_results.html', context)


def spatial_overlap_metrics_list(request):
    """Display all computed spatial overlap metrics from database in a table."""
    from collections import defaultdict
    from django.db.models import Q
    from app.models import StructureROIPair
    
    # Get search/filter parameters
    search_patient_id = request.GET.get('patient_id', '').strip()
    search_patient_name = request.GET.get('patient_name', '').strip()
    search_roi_name = request.GET.get('roi_name', '').strip()
    search_structure_set_label = request.GET.get('structure_set_label', '').strip()
    
    # Build base queryset
    pairs_qs = StructureROIPair.objects.select_related(
        'reference_rt_structure_roi__instance__series__study__patient',
        'target_rt_structure_roi__instance__series__study__patient',
        'reference_rt_structure_roi__staple_roi',
        'target_rt_structure_roi__staple_roi'
    )
    
    # Apply filters
    if search_patient_id:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__series__study__patient__patient_id__icontains=search_patient_id) |
            Q(target_rt_structure_roi__instance__series__study__patient__patient_id__icontains=search_patient_id)
        )
    
    if search_patient_name:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__series__study__patient__patient_name__icontains=search_patient_name) |
            Q(target_rt_structure_roi__instance__series__study__patient__patient_name__icontains=search_patient_name)
        )
    
    if search_roi_name:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__roi_name__icontains=search_roi_name) |
            Q(target_rt_structure_roi__roi_name__icontains=search_roi_name)
        )
    
    if search_structure_set_label:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__structure_set_label__icontains=search_structure_set_label) |
            Q(target_rt_structure_roi__instance__structure_set_label__icontains=search_structure_set_label)
        )
    
    # Get all StructureROIPair entries
    pairs = pairs_qs.order_by('-created_at')
    
    # Group metrics by ROI pair
    grouped_results = defaultdict(lambda: {
        'reference_roi': None,
        'target_roi': None,
        'reference_roi_name': '',
        'target_roi_name': '',
        'reference_roi_description': '',
        'reference_roi_generation_algorithm': '',
        'target_roi_description': '',
        'target_roi_generation_algorithm': '',
        'reference_type': '',
        'target_type': '',
        'patient_id': '',
        'metrics': {},
        'created_at': None
    })
    
    for pair in pairs:
        # Create unique key for this ROI pair
        key = f"{pair.reference_rt_structure_roi.id}_{pair.target_rt_structure_roi.id}"
        
        if not grouped_results[key]['reference_roi']:
            # First time seeing this pair, set basic info
            ref_roi = pair.reference_rt_structure_roi
            target_roi = pair.target_rt_structure_roi
            
            grouped_results[key]['reference_roi'] = ref_roi
            grouped_results[key]['target_roi'] = target_roi
            grouped_results[key]['reference_roi_name'] = ref_roi.roi_name
            grouped_results[key]['target_roi_name'] = target_roi.roi_name
            grouped_results[key]['reference_roi_description'] = ref_roi.roi_description
            grouped_results[key]['reference_roi_generation_algorithm'] = ref_roi.roi_generation_algorithm
            grouped_results[key]['target_roi_description'] = target_roi.roi_description
            grouped_results[key]['target_roi_generation_algorithm'] = target_roi.roi_generation_algorithm
            # STAPLE ROIs have instance=NULL, regular ROIs have instance set
            grouped_results[key]['reference_type'] = 'STAPLE' if ref_roi.instance is None else 'RTStruct'
            grouped_results[key]['target_type'] = 'STAPLE' if target_roi.instance is None else 'RTStruct'
            grouped_results[key]['created_at'] = pair.created_at
            
            # Get patient ID (handle both regular ROIs and STAPLE ROIs)
            if ref_roi.instance:
                grouped_results[key]['patient_id'] = ref_roi.instance.series.study.patient.patient_id
            elif ref_roi.staple_roi and ref_roi.staple_roi.instance:
                grouped_results[key]['patient_id'] = ref_roi.staple_roi.instance.series.study.patient.patient_id
            elif target_roi.instance:
                grouped_results[key]['patient_id'] = target_roi.instance.series.study.patient.patient_id
            elif target_roi.staple_roi and target_roi.staple_roi.instance:
                grouped_results[key]['patient_id'] = target_roi.staple_roi.instance.series.study.patient.patient_id
        
        # Add metric to this pair
        if pair.metric_calculated:
            grouped_results[key]['metrics'][pair.metric_calculated] = pair.metric_value
    
    # Convert to list and sort by creation date
    results_list = sorted(
        grouped_results.values(),
        key=lambda x: x['created_at'] if x['created_at'] else '',
        reverse=True
    )
    
    context = {
        'results': results_list,
        'total_pairs': len(results_list),
        'search_patient_id': search_patient_id,
        'search_patient_name': search_patient_name,
        'search_roi_name': search_roi_name,
        'search_structure_set_label': search_structure_set_label,
    }
    
    return render(request, 'app/spatial_overlap_metrics_list.html', context)


def spatial_overlap_metrics_csv(request):
    """Export spatial overlap metrics to CSV with same filtering as list view."""
    import csv
    from django.http import HttpResponse
    from collections import defaultdict
    from django.db.models import Q
    from app.models import StructureROIPair
    
    # Get search/filter parameters (same as list view)
    search_patient_id = request.GET.get('patient_id', '').strip()
    search_patient_name = request.GET.get('patient_name', '').strip()
    search_roi_name = request.GET.get('roi_name', '').strip()
    search_structure_set_label = request.GET.get('structure_set_label', '').strip()
    
    # Build base queryset (same as list view)
    pairs_qs = StructureROIPair.objects.select_related(
        'reference_rt_structure_roi__instance__series__study__patient',
        'target_rt_structure_roi__instance__series__study__patient',
        'reference_rt_structure_roi__staple_roi',
        'target_rt_structure_roi__staple_roi'
    )
    
    # Apply filters (same as list view)
    if search_patient_id:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__series__study__patient__patient_id__icontains=search_patient_id) |
            Q(target_rt_structure_roi__instance__series__study__patient__patient_id__icontains=search_patient_id)
        )
    
    if search_patient_name:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__series__study__patient__patient_name__icontains=search_patient_name) |
            Q(target_rt_structure_roi__instance__series__study__patient__patient_name__icontains=search_patient_name)
        )
    
    if search_roi_name:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__roi_name__icontains=search_roi_name) |
            Q(target_rt_structure_roi__roi_name__icontains=search_roi_name)
        )
    
    if search_structure_set_label:
        pairs_qs = pairs_qs.filter(
            Q(reference_rt_structure_roi__instance__structure_set_label__icontains=search_structure_set_label) |
            Q(target_rt_structure_roi__instance__structure_set_label__icontains=search_structure_set_label)
        )
    
    pairs = pairs_qs.order_by('-created_at')
    
    # Group metrics by ROI pair (same as list view)
    grouped_results = defaultdict(lambda: {
        'reference_roi': None,
        'target_roi': None,
        'reference_roi_name': '',
        'target_roi_name': '',
        'reference_roi_description': '',
        'reference_roi_generation_algorithm': '',
        'target_roi_description': '',
        'target_roi_generation_algorithm': '',
        'reference_type': '',
        'target_type': '',
        'patient_id': '',
        'metrics': {},
        'created_at': None
    })
    
    for pair in pairs:
        key = f"{pair.reference_rt_structure_roi.id}_{pair.target_rt_structure_roi.id}"
        
        if not grouped_results[key]['reference_roi']:
            ref_roi = pair.reference_rt_structure_roi
            target_roi = pair.target_rt_structure_roi
            
            grouped_results[key]['reference_roi'] = ref_roi
            grouped_results[key]['target_roi'] = target_roi
            grouped_results[key]['reference_roi_name'] = ref_roi.roi_name
            grouped_results[key]['target_roi_name'] = target_roi.roi_name
            grouped_results[key]['reference_roi_description'] = ref_roi.roi_description
            grouped_results[key]['reference_roi_generation_algorithm'] = ref_roi.roi_generation_algorithm
            grouped_results[key]['target_roi_description'] = target_roi.roi_description
            grouped_results[key]['target_roi_generation_algorithm'] = target_roi.roi_generation_algorithm
            # STAPLE ROIs have instance=NULL, regular ROIs have instance set
            grouped_results[key]['reference_type'] = 'STAPLE' if ref_roi.instance is None else 'RTStruct'
            grouped_results[key]['target_type'] = 'STAPLE' if target_roi.instance is None else 'RTStruct'
            grouped_results[key]['created_at'] = pair.created_at
            
            # Get patient ID (handle both regular ROIs and STAPLE ROIs)
            if ref_roi.instance:
                grouped_results[key]['patient_id'] = ref_roi.instance.series.study.patient.patient_id
            elif ref_roi.staple_roi and ref_roi.staple_roi.instance:
                grouped_results[key]['patient_id'] = ref_roi.staple_roi.instance.series.study.patient.patient_id
            elif target_roi.instance:
                grouped_results[key]['patient_id'] = target_roi.instance.series.study.patient.patient_id
            elif target_roi.staple_roi and target_roi.staple_roi.instance:
                grouped_results[key]['patient_id'] = target_roi.staple_roi.instance.series.study.patient.patient_id
        
        if pair.metric_calculated:
            grouped_results[key]['metrics'][pair.metric_calculated] = pair.metric_value
    
    results_list = sorted(
        grouped_results.values(),
        key=lambda x: x['created_at'] if x['created_at'] else '',
        reverse=True
    )
    
    # Create CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="spatial_overlap_metrics.csv"'
    
    writer = csv.writer(response)
    
    # Write header
    writer.writerow([
        'Patient ID',
        'Reference ROI Name',
        'Reference ROI Type',
        'Reference ROI Description',
        'Reference ROI Generation Algorithm',
        'Target ROI Name',
        'Target ROI Type',
        'Target ROI Description',
        'Target ROI Generation Algorithm',
        'DSC',
        'Jaccard',
        'HD95 (mm)',
        'MSD (mm)',
        'APL (mm)',
        'OMDC (mm)',
        'UMDC (mm)',
        'MDC (mm)',
        'VOE',
        'VI',
        'Cosine',
        'Surface DSC',
        'Computed Date'
    ])
    
    # Write data rows
    for result in results_list:
        writer.writerow([
            result['patient_id'],
            result['reference_roi_name'],
            result['reference_type'],
            result['reference_roi_description'] or '',
            result['reference_roi_generation_algorithm'] or '',
            result['target_roi_name'],
            result['target_type'],
            result['target_roi_description'] or '',
            result['target_roi_generation_algorithm'] or '',
            result['metrics'].get('DSC', ''),
            result['metrics'].get('Jaccard', ''),
            result['metrics'].get('HD95', ''),
            result['metrics'].get('MSD', ''),
            result['metrics'].get('APL', ''),
            result['metrics'].get('OMDC', ''),
            result['metrics'].get('UMDC', ''),
            result['metrics'].get('MDC', ''),
            result['metrics'].get('VOE', ''),
            result['metrics'].get('VI', ''),
            result['metrics'].get('Cosine', ''),
            result['metrics'].get('SurfaceDSC', ''),
            result['created_at'].strftime('%Y-%m-%d %H:%M') if result['created_at'] else ''
        ])
    
    return response


def get_series_rois_with_nifti(request, series_id):
    """API endpoint to get all ROIs with available NIfTI files for a given image series."""
    from app.utils.spatial_overlap_metrics import get_rois_for_series, get_roi_nifti_path
    
    try:
        image_series = get_object_or_404(DICOMSeries, id=series_id)
        
        # Get all ROIs for this series
        rois = get_rois_for_series(image_series.series_instance_uid)
        
        # Format ROI data
        rois_data = []
        for roi in rois:
            nifti_path = get_roi_nifti_path(roi)
            
            # Determine ROI type
            roi_type = 'STAPLE' if roi.staple_roi else 'RTStruct'
            
            # Get source information
            if roi.instance:
                source_series_uid = roi.instance.series.series_instance_uid[:30] + '...'
            elif roi.staple_roi:
                source_series_uid = 'STAPLE Consensus'
            else:
                source_series_uid = 'Unknown'
            
            rois_data.append({
                'id': roi.id,
                'name': roi.roi_name,
                'type': roi_type,
                'source': source_series_uid,
                'has_nifti': nifti_path is not None
            })
        
        return JsonResponse({
            'success': True,
            'series_id': series_id,
            'series_uid': image_series.series_instance_uid,
            'modality': image_series.modality,
            'rois': rois_data,
            'total': len(rois_data)
        })
        
    except Exception as e:
        logger.error(f"Error fetching ROIs for series {series_id}: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'rois': []
        }, status=500)


def visualize_niivue(request, series_id):
    """
    WebGL-based visualization using niivue library.
    Provides instant, GPU-accelerated rendering without pre-generating PNG slices.
    """
    from app.models import DICOMSeries
    from app.utils.niivue_visualizer import get_available_rois
    
    series = get_object_or_404(
        DICOMSeries.objects.select_related('study', 'study__patient'),
        id=series_id
    )
    
    # Check if NIfTI file exists
    if not series.nifti_file_path:
        messages.error(request, "NIfTI file not found for this series. Please convert to NIfTI first.")
        return redirect("nifti_list")
    
    # Get available ROIs
    try:
        rois = get_available_rois(series_id)
    except Exception as e:
        logger.error(f"Error getting ROIs: {e}")
        rois = []
    
    return render(request, "app/visualize_niivue.html", {
        "series": series,
        "rois": rois,
        "patient": series.study.patient,
        "study": series.study
    })


def get_niivue_data(request, series_id):
    """
    API endpoint to get NIfTI file paths and metadata for niivue visualization.
    Returns JSON with base image and overlay paths.
    """
    from app.utils.niivue_visualizer import prepare_niivue_data
    
    try:
        # Get parameters from request
        roi_names = request.GET.getlist('roi_names[]')
        include_staple = request.GET.get('include_staple', 'true').lower() == 'true'
        
        # Prepare data for niivue
        data = prepare_niivue_data(
            image_series_id=series_id,
            roi_names=roi_names if roi_names else None,
            include_staple=include_staple
        )
        
        return JsonResponse({
            'success': True,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error preparing niivue data: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
