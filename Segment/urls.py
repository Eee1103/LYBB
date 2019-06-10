from django.urls import path
from Segment import views
app_name = 'Segment'
urlpatterns = [
    #path('',views.Console_index,name="Console_index"),

    #path('forecast',views.forecast, name='forecast'),
    #path('forecast/ol',views.forecast_ol,name='forecast_ol'),
    path('forecast/ol_update', views.forecastol_update_leaf, name='forecastol_update_leaf'),
   # path('forecastresult',views.forecastresult,name="forecastresult"),

]