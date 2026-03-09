from django.urls import path
from app import views

urlpatterns = [
    path("", views.home, name="home"),
    path("archives/", views.dicom_archive_list, name="dicom_archive_list"),
    path("archives/upload/", views.dicom_archive_upload, name="dicom_archive_upload"),
    path("archives/<int:pk>/", views.dicom_archive_detail, name="dicom_archive_detail"),
    path("archives/<int:pk>/process/", views.dicom_archive_process, name="dicom_archive_process"),
    path("archives/<int:pk>/delete/", views.dicom_archive_delete, name="dicom_archive_delete"),
    path("patients/", views.patient_list, name="patient_list"),
    path("patients/<int:pk>/", views.patient_detail, name="patient_detail"),
    path("patients/<int:pk>/delete/", views.patient_delete, name="patient_delete"),
    path("patients/delete-multiple/", views.patient_delete_multiple, name="patient_delete_multiple"),
    path("studies/<int:pk>/", views.study_detail, name="study_detail"),
    path("studies/<int:pk>/delete/", views.study_delete, name="study_delete"),
    path("series/<int:pk>/delete/", views.series_delete, name="series_delete"),
    path("instances/<int:pk>/delete/", views.instance_delete, name="instance_delete"),
    path("rtstruct/", views.rtstruct_list, name="rtstruct_list"),
    path("rtstruct/extract/", views.rtstruct_extract, name="rtstruct_extract"),
    path("rois/", views.roi_list, name="roi_list"),
    path("rois/<int:series_id>/", views.roi_detail, name="roi_detail"),
    path("nifti/convert/", views.nifti_convert, name="nifti_convert"),
    path("nifti/", views.nifti_list, name="nifti_list"),
    path("nifti/staple/compute/", views.compute_staple, name="compute_staple"),
    path("nifti/staple/batch/", views.compute_batch_staple, name="compute_batch_staple"),
]
