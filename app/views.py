from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from app.models import DICOMFileArchive, Patient, DICOMStudy, DICOMSeries, DICOMInstance
from app.utils.dicom_processor import process_dicom_archive


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
    
    if archive.archive_extracted:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "success": False,
                "error": "This archive has already been processed."
            })
        messages.warning(request, "This archive has already been processed.")
        return redirect("dicom_archive_detail", pk=pk)
    
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
