from django.contrib import admin
from .models import CV, JD, Application, UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "role", "company", "phone")

@admin.register(CV)
class CVAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "cv_location", "is_active", "created_at", "updated_at")
    search_fields = ("resume_text", "cv_location")
    list_filter = ("is_active", "created_at")

@admin.register(JD)
class JDAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "job_location", "is_active", "created_at", "updated_at")
    search_fields = ("job_requirement", "job_description", "job_location")
    list_filter = ("is_active", "created_at")

@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    list_display = ("id", "cv", "jd", "status", "created_at")
    search_fields = ("status",)
