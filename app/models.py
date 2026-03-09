from django.db import models
from django.core.validators import FileExtensionValidator
from django.db.models import Q

# Create your models here.

class Patient(models.Model):
    """
    This model will store information about the Patient.
    """
    patient_id = models.CharField(max_length=255, unique=True)
    patient_name = models.CharField(max_length=255, blank=True, null=True)
    patient_dob = models.DateField(blank=True, null=True)
    patient_gender = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.patient_id

class DICOMStudy(models.Model):
    '''
    This model will store information about the DICOM Study. It is linked to the Patient model.
    '''
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    study_instance_uid = models.CharField(max_length=255, unique=True)
    study_date = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.study_instance_uid

class DICOMSeries(models.Model):
    '''
    This model will store information about the DICOM Series. It is linked to the DICOMStudy model.
    '''
    study = models.ForeignKey(DICOMStudy, on_delete=models.CASCADE)
    series_instance_uid = models.CharField(max_length=255, unique=True)
    modality = models.CharField(max_length=255, blank=True, null=True)
    frame_of_reference_uid = models.CharField(max_length=255, blank=True, null=True)
    series_date = models.DateField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    nifti_file_path = models.CharField(max_length=512, blank=True, null=True)
    
    def __str__(self):
        return self.series_instance_uid

class DICOMInstance(models.Model):
    '''
    This model will store information about the DICOM Instance. It is linked to the DICOMSeries model.
    '''
    series = models.ForeignKey(DICOMSeries, on_delete=models.CASCADE)
    sop_instance_uid = models.CharField(max_length=255, unique=True)
    referenced_series_instance_uid = models.ForeignKey(DICOMSeries, on_delete=models.SET_NULL,related_name="referenced_series_uid",null=True,blank=True,help_text="This is the reference series instance UID for the RTStruct File only")
    instance_number = models.IntegerField(blank=True, null=True)
    instance_file_path = models.CharField(max_length=512, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.sop_instance_uid


class StapleROI(models.Model):
    '''
    This model will store information about the STAPLE ROI. It is linked to the RTStructROI model.
    '''
    instance = models.ForeignKey(DICOMInstance, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.instance.sop_instance_uid


class RTStructROI(models.Model):
    '''
    This model will store information about the RT Struct ROI. It is linked to the DICOMInstance model.
    '''
    instance = models.ForeignKey(DICOMInstance, on_delete=models.CASCADE,null=True,blank=True)
    staple_roi = models.ForeignKey(StapleROI, on_delete=models.CASCADE,null=True,blank=True)
    roi_number = models.IntegerField(null=True,blank = True)
    roi_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    
    def __str__(self):
        return self.roi_name



class DICOMFileArchive(models.Model):
    '''
    This model will store information about the DICOM file archive which will be uploaded for a given project
    '''
    file = models.FileField(upload_to='dicom_zip_files/', validators=[FileExtensionValidator(['zip'])])
    archive_extracted = models.BooleanField(default=False)
    archive_extraction_date_time = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.file.name


class StructureROIPair(models.Model):
    '''
    This model will store the structure pairs for which the spatial overlap metrics will be calculated along with the metric value
    '''
    reference_rt_structure_roi = models.ForeignKey(RTStructROI, on_delete=models.CASCADE, related_name='reference_structure_roi')
    target_rt_structure_roi = models.ForeignKey(RTStructROI, on_delete=models.CASCADE, related_name='target_structure_roi')
    metric_calculated = models.CharField(max_length=255, blank=True, null=True)
    metric_value = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.reference_rt_structure_roi.roi_name} - {self.target_rt_structure_roi.roi_name}"
