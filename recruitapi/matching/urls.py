from django.urls import path
from .views import HealthView, RankView, ReloadModelView, WarmupView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("rank/", RankView.as_view(), name="rank"),
    path("reload-model/", ReloadModelView.as_view(), name="reload-model"),
    path("warmup/", WarmupView.as_view()),
]
