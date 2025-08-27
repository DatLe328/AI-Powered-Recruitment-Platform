from django.contrib import admin
from .models import CV, JD

@admin.register(CV)
class CVAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "title", "is_active", "updated_at")
    search_fields = ("title", "resume_text", "user__username")
    list_filter = ("is_active",)

@admin.register(JD)
class JDAdmin(admin.ModelAdmin):
    list_display = ("id", "owner", "title", "is_active", "updated_at")
    search_fields = ("title", "description", "requirements", "owner__username")
    list_filter = ("is_active",)
