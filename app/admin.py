from django.contrib import admin
from app.models import Patient, DICOMStudy, DICOMSeries, DICOMInstance

# Register your models here.

class PatientAdmin(admin.ModelAdmin):
    list_display = ('patient_id', 'patient_name', 'patient_dob', 'patient_gender')
    search_fields = ('patient_id', 'patient_name')
    list_filter = ('patient_gender',)

admin.site.register(Patient, PatientAdmin)


class DICOMStudyAdmin(admin.ModelAdmin):
    list_display = ( 'patient','study_instance_uid', 'study_date')
    search_fields = ('study_instance_uid', 'patient__patient_id', 'patient__patient_name')
    list_filter = ('study_date',)

admin.site.register(DICOMStudy, DICOMStudyAdmin)

class DICOMSeriesAdmin(admin.ModelAdmin):
    list_display = ('get_patient_id', 'get_study_id', 'series_instance_uid', 'modality', 'series_date')
    search_fields = ('series_instance_uid', 'study__study_instance_uid', 'study__patient__patient_id')
    list_filter = ('modality', 'study__study_date', 'series_date')
    
    @admin.display(ordering='study__study_instance_uid', description='Study ID')
    def get_study_id(self, obj):
        return obj.study.study_instance_uid if obj.study else '-'
    
    @admin.display(ordering='study__patient__patient_id', description='Patient ID')
    def get_patient_id(self, obj):
        return obj.study.patient.patient_id if obj.study and obj.study.patient else '-'

admin.site.register(DICOMSeries, DICOMSeriesAdmin)

class DICOMInstanceAdmin(admin.ModelAdmin):
    list_display = ('get_patient_id', 'get_study_id', 'get_series_id', 'sop_instance_uid', 'instance_number')
    search_fields = ('sop_instance_uid', 'series__series_instance_uid', 'series__study__study_instance_uid', 'series__study__patient__patient_id')
    list_filter = ('series__modality', 'series__study__study_date')
    
    @admin.display(ordering='series__series_instance_uid', description='Series ID')
    def get_series_id(self, obj):
        return obj.series.series_instance_uid if obj.series else '-'
    
    @admin.display(ordering='series__study__study_instance_uid', description='Study ID')
    def get_study_id(self, obj):
        return obj.series.study.study_instance_uid if obj.series and obj.series.study else '-'
    
    @admin.display(ordering='series__study__patient__patient_id', description='Patient ID')
    def get_patient_id(self, obj):
        return obj.series.study.patient.patient_id if obj.series and obj.series.study and obj.series.study.patient else '-'

admin.site.register(DICOMInstance, DICOMInstanceAdmin)
