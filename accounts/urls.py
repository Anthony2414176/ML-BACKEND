from django.urls import path
from accounts.views.auth import login_user
from accounts.views.cleaning import CleanDataView, list_cleaned_files, delete_cleaned_file
from accounts.views.cleaning import log_repush_action,list_anomalies_files,get_anomaly_file_content

urlpatterns = [
    path('login/', login_user, name='login'),
    path('clean-data/', CleanDataView.as_view(), name='clean-data'),
    path('cleaned-files/', list_cleaned_files),
    path('delete-cleaned-file/', delete_cleaned_file, name='delete-cleaned-file'),
    path('log-repush-action/', log_repush_action, name='log_repush_action'),
    path('anomalies-files/', list_anomalies_files, name='list_anomalies_files'),
    path('anomaly-file/<str:filename>/', get_anomaly_file_content, name='get_anomaly_file_content'),
]
