from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ml.ai_gateway import load_model, reload_model, rank_cv_for_jd

class HealthView(APIView):
    authentication_classes = []
    permission_classes = []
    def get(self, request):
        try:
            load_model()
            return Response({"status":"ok"}, status=200)
        except Exception as e:
            return Response({"status":"error","detail":str(e)}, status=500)

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
        from ml.ai_gateway import load_model, rank_cv_for_jd
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