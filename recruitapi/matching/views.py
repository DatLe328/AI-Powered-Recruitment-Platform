from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.exceptions import NotAuthenticated, PermissionDenied
from rest_framework_simplejwt.authentication import JWTAuthentication

from django.contrib.auth.models import User

from .models import CV, JD, Application
from .serializers import (
    CVSerializer, JDSerializer,
    RankRequestSerializer, RegisterSerializer, UserSerializer,
)
from .services import add_one_to_faiss
from ml.apis import is_loaded as model_is_loaded, reload_model, rank_cv_for_jd
from ml.embeddings import embed_texts
from ml.vectorstore.faiss_store import is_loaded as faiss_is_loaded, load as faiss_load, search as faiss_search

# ===== Helpers =====
def require_role(user, role: str):
    if not hasattr(user, "profile"):
        raise PermissionDenied("User profile not found")
    if user.profile.role != role:
        raise PermissionDenied(f"Only {role} can perform this action")

# ===== Health (public) =====
class HealthView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    def get(self, request):
        return Response({
            "ml_model": "ready" if model_is_loaded() else "warming",
            "faiss": "ready" if faiss_is_loaded() else ("loaded" if faiss_load() else "empty"),
        }, status=200)

# ===== Reload model (JWT) =====
class ReloadModelView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        reload_model()
        return Response({"status":"reloaded"}, status=200)

# ===== Warmup (JWT) =====
class WarmupView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        from ml.apis import load_model
        from ml.vectorstore import faiss_store
        load_model()         # ép load ranker + embedder
        faiss_store.load()   # ép load FAISS từ file
        _ = embed_texts(["warmup sentence for SBERT model"]*2)
        return Response({"status":"warmed"}, status=200)

# ===== FAISS search (JWT) =====
class FaissSearchView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def post(self, request):
        q = (request.data.get("query") or "").strip()
        topk = int(request.data.get("topk", 10))
        if not q:
            return Response({"detail":"query is empty"}, status=400)
        if not faiss_is_loaded():
            if not faiss_load():
                return Response({"detail":"FAISS index not found"}, status=503)
        q_emb = embed_texts([q])
        results = faiss_search(q_emb, topk=topk)[0]  # [(cv_id, score), ...]
        return Response({"results": [{"cv_id": str(cid), "score": float(score)} for cid, score in results]}, status=200)

# ===== Ranking (JWT) =====
class RankView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        ser = RankRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        jd_id = data.get("jd_id")
        job_requirement = data.get("job_requirement")
        job_description = data.get("job_description")

        # 1) Nếu có jd_id -> lấy JD từ DB
        if jd_id:
            try:
                jd = JD.objects.get(id=jd_id, is_active=True)
            except JD.DoesNotExist:
                return Response({"detail": f"JD id={jd_id} không tồn tại"}, status=404)
            job_requirement = jd.job_requirement
            job_description = jd.job_description

        # 2) Nếu chưa có candidates mà có jd_id -> tự gom CV đã apply vào JD
        candidates = data.get("candidates")
        if candidates is None and jd_id:
            # Lấy tất cả application trỏ tới JD này, kèm CV tương ứng
            apps = (Application.objects
                    .filter(jd_id=jd_id)
                    .select_related("cv"))
            # Build candidates theo format đầu vào của ranker
            candidates = []
            for a in apps:
                cv = a.cv
                # Tùy serializer/ranker của bạn cần gì; tối thiểu là cv_id + resume_text
                candidates.append({
                    "cv_id": cv.id,
                    "resume_text": cv.resume_text or "",
                    # nếu ranker dùng location/name thì thêm:
                    "cv_location": getattr(cv, "cv_location", "") or "",
                    "name": getattr(cv, "name", "") or "",
                })

        # 3) Validate tối thiểu
        if not job_requirement:
            return Response({"detail": "Thiếu job_requirement (hoặc jd_id không hợp lệ)"}, status=400)
        if not candidates:
            return Response({"detail": "Không có candidates (chưa apply ai vào JD này?)"}, status=400)

        # 4) Gọi ranker
        ranked = rank_cv_for_jd(
            job_requirement=job_requirement,
            job_description=job_description,
            candidates=candidates,
            topk=data.get("topk"),
        )
        return Response({"results": ranked}, status=200)

    # def post(self, request):
    #     ser = RankRequestSerializer(data=request.data)
    #     ser.is_valid(raise_exception=True)
    #     data = ser.validated_data
    #     ranked = rank_cv_for_jd(
    #         job_requirement=data.get("job_requirement"),
    #         job_description=data.get("job_description"),
    #         candidates=data["candidates"],
    #         topk=data.get("topk"),
    #     )
    #     return Response({"results": ranked}, status=200)

# ===== Auth =====
class RegisterView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    def post(self, request):
        ser = RegisterSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = ser.save()
        return Response(UserSerializer(user).data, status=201)

class MeView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request):
        return Response(UserSerializer(request.user).data, status=200)

# ===== CV/JD CRUD (JWT + role) =====
class CVViewSet(viewsets.ModelViewSet):
    serializer_class = CVSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return CV.objects.filter(owner=self.request.user, is_active=True).order_by("-updated_at")

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise NotAuthenticated("Bạn cần đăng nhập")
        require_role(self.request.user, "candidate")
        cv = serializer.save(owner=self.request.user)
        try:
            add_one_to_faiss(cv.id, cv.resume_text)
        except Exception:
            pass

    def perform_update(self, serializer):
        require_role(self.request.user, "candidate")
        cv = serializer.save()
        try:
            add_one_to_faiss(cv.id, cv.resume_text)
        except Exception:
            pass

    def perform_destroy(self, instance: CV):
        require_role(self.request.user, "candidate")
        instance.is_active = False
        instance.save(update_fields=["is_active"])

class JDViewSet(viewsets.ModelViewSet):
    serializer_class = JDSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return JD.objects.filter(owner=self.request.user, is_active=True).order_by("-updated_at")

    def perform_create(self, serializer):
        if not self.request.user.is_authenticated:
            raise NotAuthenticated("Bạn cần đăng nhập")
        require_role(self.request.user, "recruiter")
        serializer.save(owner=self.request.user)

    def perform_update(self, serializer):
        require_role(self.request.user, "recruiter")
        serializer.save()

    def perform_destroy(self, instance: JD):
        require_role(self.request.user, "recruiter")
        instance.is_active = False
        instance.save(update_fields=["is_active"])
