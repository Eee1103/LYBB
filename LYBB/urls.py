from django.contrib import admin
from django.urls import path,include
from Segment import views
urlpatterns = [
    path('admin/', admin.site.urls),

    path('Segment/', include('Segment.urls')),
    path('mobileforecastleaf', views.mobile_forecast_leaf, name="mobileforecast_leaf"),
]
