from django.contrib.auth.models import User
from rest_framework import serializers
from .models import CV, JD, Application, UserProfile

# ===== User / Auth =====
class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    role = serializers.ChoiceField(choices=UserProfile.ROLE_CHOICES)

    class Meta:
        model = User
        fields = ("id", "username", "email", "password", "role")

    def create(self, validated_data):
        role = validated_data.pop("role", "candidate")
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email", ""),
            password=validated_data["password"],
        )
        # cập nhật role cho profile (profile được tạo bởi signal)
        user.profile.role = role
        user.profile.save()
        return user

class UserSerializer(serializers.ModelSerializer):
    role = serializers.CharField(source="profile.role", read_only=True)
    class Meta:
        model = User
        fields = ("id", "username", "email", "role")

# ===== Domain =====
class CVSerializer(serializers.ModelSerializer):
    class Meta:
        model = CV
        fields = ["id","resume_text","cv_location","is_active","created_at","updated_at"]

class JDSerializer(serializers.ModelSerializer):
    class Meta:
        model = JD
        fields = ["id","job_requirement","job_description","job_location","is_active","created_at","updated_at"]

class ApplicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Application
        fields = "__all__"

# ===== Ranking payload =====
class CVItemSerializer(serializers.Serializer):
    cv_id = serializers.CharField(required=False, allow_blank=True)
    resume_text = serializers.CharField()
    cv_location = serializers.CharField(required=False, allow_blank=True)

class RankRequestSerializer(serializers.Serializer):
    jd_id = serializers.IntegerField(required=False)
    job_requirement = serializers.CharField(required=False, allow_blank=True)
    job_description = serializers.CharField(required=False, allow_blank=True)
    topk = serializers.IntegerField(required=False)

    candidates = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )