from rest_framework import serializers

class CVItemSerializer(serializers.Serializer):
    cv_id = serializers.CharField(required=False, allow_blank=True)
    resume_text = serializers.CharField()
    cv_location = serializers.CharField(required=False, allow_blank=True)

class RankRequestSerializer(serializers.Serializer):
    job_requirement = serializers.CharField(required=False, allow_blank=True)
    job_description = serializers.CharField(required=False, allow_blank=True)
    job_location = serializers.CharField(required=False, allow_blank=True)
    candidates = CVItemSerializer(many=True)
    topk = serializers.IntegerField(required=False, min_value=1)
    # bonus flags
    use_location_bonus = serializers.BooleanField(required=False, default=True)
    loc_alpha = serializers.FloatField(required=False, min_value=0.0, max_value=0.5, default=0.1)
    use_seniority_bonus = serializers.BooleanField(required=False, default=False)
    level_alpha = serializers.FloatField(required=False, min_value=0.0, max_value=0.5, default=0.1)

class RankResponseItemSerializer(serializers.Serializer):
    jd_id = serializers.CharField()
    cv_id = serializers.CharField()
    pred = serializers.FloatField()
    score = serializers.FloatField()   # final score sau khi bonus
