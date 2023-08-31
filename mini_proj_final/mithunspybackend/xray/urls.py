from django.urls import path
from . import views

urlpatterns = [
    path('mofothesepplare/',views.getPredForXrayImg),
]