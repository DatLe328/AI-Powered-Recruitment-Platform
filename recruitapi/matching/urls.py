from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    HealthView, ReloadModelView, WarmupView,
    RankView, FaissSearchView,
    CVViewSet, JDViewSet,
    RegisterView, MeView,
)

router = DefaultRouter()
router.register(r"cvs", CVViewSet, basename="cv")
router.register(r"jds", JDViewSet, basename="jd")

urlpatterns = [
    path("health/", HealthView.as_view()),
    path("reload-model/", ReloadModelView.as_view()),
    path("warmup/", WarmupView.as_view()),
    path("rank/", RankView.as_view()),
    path("faiss/search", FaissSearchView.as_view()),
    # auth
    path("auth/register/", RegisterView.as_view()),
    path("auth/me/", MeView.as_view()),
    path("", include(router.urls)),
]
