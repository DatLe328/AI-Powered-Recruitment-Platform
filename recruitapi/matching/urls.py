from django.urls import path, include
from .views import FaissSearchView, FaissReloadView, HealthView, RankView, ReloadModelView, WarmupView
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import RegisterView, CVViewSet, JDViewSet

router = DefaultRouter()
router.register(r"cvs", CVViewSet, basename="cv")
router.register(r"jds", JDViewSet, basename="jd")

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("rank/", RankView.as_view(), name="rank"),
    path("reload-model/", ReloadModelView.as_view(), name="reload-model"),
    path("warmup/", WarmupView.as_view()),
    path("faiss/search", FaissSearchView.as_view(), name="faiss-search"),
    path("faiss/reload", FaissReloadView.as_view(), name="faiss-reload"),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # resources
    path("", include(router.urls)),
]
