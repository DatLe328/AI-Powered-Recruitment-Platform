from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ml.apis import is_loaded, reload_model, rank_cv_for_jd

class HealthView(APIView):
    authentication_classes = []
    permission_classes = []
    def get(self, request):
        return Response({"status": "ok" if is_loaded() else "warming"}, status=200)

class ReloadModelView(APIView):
    def post(self, request):
        reload_model()
        return Response({"status":"reloaded"}, status=200)

class RankView(APIView):
    def post(self, request):
        payload = request.data
        try:
            results = rank_cv_for_jd(
                job_requirement=payload.get("job_requirement"),
                job_description=payload.get("job_description"),
                candidates=payload.get("candidates", []),
                topk=payload.get("topk"),
            )
            return Response({"results": results}, status=200)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class WarmupView(APIView):
    authentication_classes = []
    permission_classes = []
    def post(self, request):
        from ml.apis import load_model, rank_cv_for_jd
        try:
            load_model()
            _ = rank_cv_for_jd(
                job_requirement="python fastapi postgresql docker",
                job_description="ci/cd monitoring",
                candidates=[
                    {"cv_id":"warm-1","resume_text":"Python FastAPI PostgreSQL Docker"},
                    {"cv_id":"warm-2","resume_text":"Java Spring MySQL Docker"}
                ],
                topk=1,
            )
            return Response({"status":"warmed"}, status=200)
        except Exception as e:
            return Response({"status":"error","detail":str(e)}, status=500)

from ml.embeddings import embed_texts
from ml.vectorstore.faiss_store import is_loaded as faiss_is_loaded, search as faiss_search

class FaissSearchView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        q = request.data.get("query", "")
        topk = int(request.data.get("topk", 10))
        if not q:
            return Response({"detail": "query is required"}, status=400)
        if not faiss_is_loaded():
            return Response({"detail": "FAISS index not loaded"}, status=503)
        try:
            q_emb = embed_texts([q])  # (1, D) float32
            results = faiss_search(q_emb, topk=topk)[0]  # [(cv_id, score), ...]
            return Response({"results": [{"cv_id": cid, "score": s} for cid, s in results]})
        except Exception as e:
            return Response({"detail": str(e)}, status=500)

class FaissReloadView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        from ml.vectorstore.faiss_store import load as faiss_load
        ok = faiss_load()
        return Response({"reloaded": bool(ok)})

        
from .services import (
    upsert_cv_embedding_and_update_faiss,
    delete_cv_embedding,
)

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django.db.models import QuerySet
from .models import CV, JD
from .serializers import CVSerializer, JDSerializer, RegisterSerializer, UserSerializer
from .permissions import IsOwnerOrReadOnly
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import NotAuthenticated

class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]
    def post(self, request):
        ser = RegisterSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = ser.save()
        return Response(UserSerializer(user).data, status=201)

class CVViewSet(viewsets.ModelViewSet):
    serializer_class = CVSerializer
    # ✅ BẮT BUỘC AUTH + check sở hữu
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]

    def get_queryset(self):
        return CV.objects.filter(user=self.request.user, is_active=True).order_by("-updated_at")

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:  # phòng thủ
            raise NotAuthenticated("Bạn cần đăng nhập")
        cv = serializer.save(user=self.request.user)
        upsert_cv_embedding_and_update_faiss(cv.id, cv.resume_text)

    def perform_update(self, serializer):
        if not self.request.user.is_authenticated:
            raise NotAuthenticated("Bạn cần đăng nhập")
        cv = serializer.save()
        upsert_cv_embedding_and_update_faiss(cv.id, cv.resume_text)

    def perform_destroy(self, instance: CV):
        instance.is_active = False
        instance.save(update_fields=["is_active"])
        delete_cv_embedding(instance.id)

class JDViewSet(viewsets.ModelViewSet):
    serializer_class = JDSerializer
    # ✅ BẮT BUỘC AUTH + check sở hữu
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]

    def get_queryset(self):
        return JD.objects.filter(owner=self.request.user, is_active=True).order_by("-updated_at")

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise NotAuthenticated("Bạn cần đăng nhập")
        serializer.save(owner=self.request.user)

    def perform_update(self, serializer):
        if not self.request.user.is_authenticated:
            raise NotAuthenticated("Bạn cần đăng nhập")
        serializer.save()

    def perform_destroy(self, instance: JD):
        instance.is_active = False
        instance.save(update_fields=["is_active"])