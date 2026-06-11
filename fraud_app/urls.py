from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("dashboard/",views.dashboard,name="dashboard"),
    path("download-report/",views.download_report,name="download_report"),
    path("export-csv/",views.export_csv,name="export_csv"),
]