import os
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

import pydicom
from celery import shared_task
from celery_progress.backend import ProgressRecorder
from django.core.cache import cache
from django.utils import timezone

from app.models import (
    DICOMFileArchive,
    DICOMInstance,
    DICOMSeries,
    DICOMStudy,
    Patient,
)


def _update_progress(archive_id, stage, current, total, message=""):
    """Update progress in cache for the given archive."""
    progress_key = f"dicom_processing_{archive_id}"
    progress_data = {
        "stage": stage,
        "current": current,
        "total": total,
        "percent": int((current / total * 100)) if total > 0 else 0,
        "message": message,
        "is_complete": False,
    }
    cache.set(progress_key, progress_data, timeout=3600)


def _mark_complete(archive_id, result):
    """Mark processing as complete with result."""
    progress_key = f"dicom_processing_{archive_id}"
    progress_data = {
        "stage": "complete",
        "current": 100,
        "total": 100,
        "percent": 100,
        "message": "Processing complete",
        "is_complete": True,
        "result": result,
    }
    cache.set(progress_key, progress_data, timeout=3600)


def get_processing_progress(archive_id):
    """Get the current processing progress for an archive."""
    progress_key = f"dicom_processing_{archive_id}"
    return cache.get(progress_key, {
        "stage": "unknown",
        "current": 0,
        "total": 0,
        "percent": 0,
        "message": "No processing data",
        "is_complete": False,
    })


def clear_processing_progress(archive_id):
    """Clear processing progress from cache."""
    progress_key = f"dicom_processing_{archive_id}"
    cache.delete(progress_key)


@shared_task(bind=True)
def process_dicom_archive(self, archive_id, progress_callback=None):
    """
    Process a DICOM file archive by extracting it and saving DICOM metadata to the database.

    Args:
        self: Celery task instance
        archive_id: The ID of the DICOMFileArchive to process
        progress_callback: Optional callback function(stage, current, total, message)

    Returns:
        dict: A summary of the processing results
    """
    progress_recorder = ProgressRecorder(self)
    
    def report_progress(stage, current, total, message=""):
        percent = int((current / total * 100)) if total > 0 else 0
        progress_recorder.set_progress(current, total, description=message)
        if progress_callback:
            progress_callback(stage, current, total, message)

    try:
        archive = DICOMFileArchive.objects.get(id=archive_id)
    except DICOMFileArchive.DoesNotExist:
        result = {"success": False, "error": f"Archive with ID {archive_id} not found"}
        _mark_complete(archive_id, result)
        return result

    # Initialize counters and tracking
    processed_files = 0
    skipped_files = 0
    errors = []

    # Data structures to hold records for bulk creation/update
    patients_data = {}
    studies_data = {}
    series_data = {}
    instances_data = {}

    # Create the dicom_files directory if it doesn't exist
    base_dicom_dir = Path("dicom_files")
    base_dicom_dir.mkdir(exist_ok=True)

    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        report_progress("extracting", 0, 100, "Extracting ZIP archive...")

        # Extract the zip file
        try:
            with zipfile.ZipFile(archive.file.path, "r") as zip_ref:
                zip_ref.extractall(temp_path)
        except zipfile.BadZipFile:
            result = {"success": False, "error": "Invalid zip file"}
            _mark_complete(archive_id, result)
            return result
        except Exception as e:
            result = {"success": False, "error": f"Error extracting zip: {str(e)}"}
            _mark_complete(archive_id, result)
            return result

        # Count total files for progress calculation
        total_files = 0
        for root, dirs, files in os.walk(temp_path):
            total_files += len(files)

        if total_files == 0:
            result = {"success": False, "error": "No files found in archive"}
            _mark_complete(archive_id, result)
            return result

        report_progress("scanning", 0, total_files, f"Found {total_files} files to process...")

        current_file = 0

        # Walk through all files in the extracted directory
        for root, dirs, files in os.walk(temp_path):
            for filename in files:
                current_file += 1
                file_path = Path(root) / filename

                # Update progress every 5 files or at milestones
                if current_file % 5 == 0 or current_file == total_files:
                    report_progress(
                        "processing",
                        current_file,
                        total_files,
                        f"Processing file {current_file} of {total_files}... ({processed_files} valid DICOM files found)"
                    )

                try:
                    # Try to read the file as DICOM
                    try:
                        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True, force=True)
                    except Exception:
                        skipped_files += 1
                        continue

                    # Check for required tags (PatientID and Modality)
                    if not hasattr(ds, "PatientID") or not hasattr(ds, "Modality"):
                        skipped_files += 1
                        continue

                    patient_id = ds.PatientID
                    modality = ds.Modality

                    # Extract metadata
                    patient_name = getattr(ds, "PatientName", None)
                    patient_dob = getattr(ds, "PatientBirthDate", None)
                    study_instance_uid = getattr(ds, "StudyInstanceUID", None)
                    study_date = getattr(ds, "StudyDate", None)
                    series_instance_uid = getattr(ds, "SeriesInstanceUID", None)
                    series_date = getattr(ds, "SeriesDate", None)
                    sop_instance_uid = getattr(ds, "SOPInstanceUID", None)
                    frame_of_reference_uid = getattr(ds, "FrameOfReferenceUID", None)
                    instance_number = getattr(ds, "InstanceNumber", None)

                    # Skip if essential UIDs are missing
                    if not all([study_instance_uid, series_instance_uid, sop_instance_uid]):
                        skipped_files += 1
                        continue

                    # Parse dates
                    if patient_dob and len(str(patient_dob)) == 8:
                        try:
                            patient_dob = datetime.strptime(str(patient_dob), "%Y%m%d").date()
                        except ValueError:
                            patient_dob = None
                    else:
                        patient_dob = None

                    if study_date and len(str(study_date)) == 8:
                        try:
                            study_date = datetime.strptime(str(study_date), "%Y%m%d").date()
                        except ValueError:
                            study_date = None
                    else:
                        study_date = None

                    if series_date and len(str(series_date)) == 8:
                        try:
                            series_date = datetime.strptime(str(series_date), "%Y%m%d").date()
                        except ValueError:
                            series_date = None
                    else:
                        series_date = None

                    # Parse instance number
                    if instance_number is not None:
                        try:
                            instance_number = int(instance_number)
                        except (ValueError, TypeError):
                            instance_number = None

                    # Build the directory structure and save the file
                    patient_dir = base_dicom_dir / patient_id
                    study_dir = patient_dir / study_instance_uid
                    series_dir = study_dir / series_instance_uid
                    series_dir.mkdir(parents=True, exist_ok=True)

                    # Save the DICOM file
                    dicom_filename = f"{sop_instance_uid}.dcm"
                    dicom_file_path = series_dir / dicom_filename

                    # Use pydicom's save_as to save the file
                    ds.save_as(str(dicom_file_path))

                    # Get absolute path
                    absolute_file_path = str(dicom_file_path.absolute())

                    # Store data for bulk processing
                    patients_data[patient_id] = {
                        "patient_id": patient_id,
                        "patient_name": str(patient_name) if patient_name else None,
                        "patient_dob": patient_dob,
                        "patient_gender": getattr(ds, "PatientSex", None),
                    }

                    study_key = (patient_id, study_instance_uid)
                    studies_data[study_key] = {
                        "patient_id": patient_id,
                        "study_instance_uid": study_instance_uid,
                        "study_date": study_date,
                    }

                    series_key = (study_instance_uid, series_instance_uid)
                    series_data[series_key] = {
                        "study_instance_uid": study_instance_uid,
                        "series_instance_uid": series_instance_uid,
                        "modality": modality,
                        "frame_of_reference_uid": frame_of_reference_uid,
                        "series_date": series_date,
                    }

                    instances_data[sop_instance_uid] = {
                        "series_instance_uid": series_instance_uid,
                        "sop_instance_uid": sop_instance_uid,
                        "instance_number": instance_number,
                        "instance_file_path": absolute_file_path,
                    }

                    processed_files += 1

                except Exception as e:
                    errors.append(f"Error processing {filename}: {str(e)}")
                    continue

    report_progress("saving", 95, 100, f"Saving {len(patients_data)} patients, {len(studies_data)} studies to database...")

    # Bulk create/update database records
    try:
        # Process Patients
        existing_patients = {
            p.patient_id: p
            for p in Patient.objects.filter(patient_id__in=list(patients_data.keys()))
        }

        patients_to_create = []
        patients_to_update = []

        for patient_id, data in patients_data.items():
            if patient_id in existing_patients:
                patient = existing_patients[patient_id]
                patient.patient_name = data["patient_name"]
                patient.patient_dob = data["patient_dob"]
                patient.patient_gender = data["patient_gender"]
                patients_to_update.append(patient)
            else:
                patients_to_create.append(Patient(**data))

        if patients_to_create:
            Patient.objects.bulk_create(patients_to_create)
        if patients_to_update:
            Patient.objects.bulk_update(
                patients_to_update, ["patient_name", "patient_dob", "patient_gender"]
            )

        # Refresh patient mapping
        patient_objs = Patient.objects.filter(patient_id__in=list(patients_data.keys()))
        patient_id_to_obj = {p.patient_id: p for p in patient_objs}

        # Process Studies
        existing_studies = {
            s.study_instance_uid: s
            for s in DICOMStudy.objects.filter(
                study_instance_uid__in=[
                    data["study_instance_uid"] for data in studies_data.values()
                ]
            )
        }

        studies_to_create = []
        studies_to_update = []

        for study_key, data in studies_data.items():
            study_uid = data["study_instance_uid"]
            patient = patient_id_to_obj.get(data["patient_id"])

            if study_uid in existing_studies:
                study = existing_studies[study_uid]
                study.patient = patient
                study.study_date = data["study_date"]
                studies_to_update.append(study)
            else:
                studies_to_create.append(
                    DICOMStudy(
                        patient=patient,
                        study_instance_uid=study_uid,
                        study_date=data["study_date"],
                    )
                )

        if studies_to_create:
            DICOMStudy.objects.bulk_create(studies_to_create)
        if studies_to_update:
            DICOMStudy.objects.bulk_update(studies_to_update, ["patient", "study_date"])

        # Refresh study mapping
        study_objs = DICOMStudy.objects.filter(
            study_instance_uid__in=[
                data["study_instance_uid"] for data in studies_data.values()
            ]
        )
        study_uid_to_obj = {s.study_instance_uid: s for s in study_objs}

        # Process Series
        existing_series = {
            s.series_instance_uid: s
            for s in DICOMSeries.objects.filter(
                series_instance_uid__in=[
                    data["series_instance_uid"] for data in series_data.values()
                ]
            )
        }

        series_to_create = []
        series_to_update = []

        for series_key, data in series_data.items():
            series_uid = data["series_instance_uid"]
            study = study_uid_to_obj.get(data["study_instance_uid"])

            if series_uid in existing_series:
                s = existing_series[series_uid]
                s.study = study
                s.modality = data["modality"]
                s.frame_of_reference_uid = data["frame_of_reference_uid"]
                s.series_date = data["series_date"]
                series_to_update.append(s)
            else:
                series_to_create.append(
                    DICOMSeries(
                        study=study,
                        series_instance_uid=series_uid,
                        modality=data["modality"],
                        frame_of_reference_uid=data["frame_of_reference_uid"],
                        series_date=data["series_date"],
                    )
                )

        if series_to_create:
            DICOMSeries.objects.bulk_create(series_to_create)
        if series_to_update:
            DICOMSeries.objects.bulk_update(
                series_to_update,
                ["study", "modality", "frame_of_reference_uid", "series_date"],
            )

        # Refresh series mapping
        series_objs = DICOMSeries.objects.filter(
            series_instance_uid__in=[
                data["series_instance_uid"] for data in series_data.values()
            ]
        )
        series_uid_to_obj = {s.series_instance_uid: s for s in series_objs}

        # Process Instances
        existing_instances = {
            i.sop_instance_uid: i
            for i in DICOMInstance.objects.filter(
                sop_instance_uid__in=list(instances_data.keys())
            )
        }

        instances_to_create = []
        instances_to_update = []

        for sop_uid, data in instances_data.items():
            series = series_uid_to_obj.get(data["series_instance_uid"])

            if sop_uid in existing_instances:
                inst = existing_instances[sop_uid]
                inst.series = series
                inst.instance_number = data["instance_number"]
                inst.instance_file_path = data["instance_file_path"]
                instances_to_update.append(inst)
            else:
                instances_to_create.append(
                    DICOMInstance(
                        series=series,
                        sop_instance_uid=sop_uid,
                        instance_number=data["instance_number"],
                        instance_file_path=data["instance_file_path"],
                    )
                )

        if instances_to_create:
            DICOMInstance.objects.bulk_create(instances_to_create)
        if instances_to_update:
            DICOMInstance.objects.bulk_update(
                instances_to_update, ["series", "instance_number", "instance_file_path"]
            )

        # Update archive status
        archive.archive_extracted = True
        archive.archive_extraction_date_time = timezone.now()
        archive.save()

    except Exception as e:
        result = {
            "success": False,
            "error": f"Database error: {str(e)}",
            "processed_files": int(processed_files),
            "skipped_files": int(skipped_files),
        }
        _mark_complete(archive_id, result)
        return result

    result = {
        "success": True,
        "processed_files": int(processed_files),
        "skipped_files": int(skipped_files),
        "errors": [str(error) for error in errors],
        "patients_created": int(len(patients_data)),
        "studies_created": int(len(studies_data)),
        "series_created": int(len(series_data)),
        "instances_created": int(len(instances_data)),
    }

    _mark_complete(archive_id, result)
    return result