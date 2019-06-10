from django.db import models
from django.utils.timezone import now

class forecast_img(models.Model):
    img = models.ImageField(upload_to='forecast_img')
    name = models.CharField(max_length=100)

class forecastol_img(models.Model):
    '''
    name = models.CharField(max_length=50, default="")
    time = models.DateTimeField(default=now())
    material = models.CharField(max_length=50, default="")
    origin = models.CharField(max_length=50, default="")
    ifRot = models.CharField(max_length=50, default="")
    ifRot_result=models.CharField(max_length=50, default="")
    '''
    imgroot=models.CharField(max_length=100, default="")



