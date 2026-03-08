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
    path("studies/<int:pk>/", views.study_detail, name="study_detail"),
]
