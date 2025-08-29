from django.db import models
from django.contrib.auth.models import User

# ===== Base =====

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True 

# ===== User Profile (role) =====
class UserProfile(models.Model):
    ROLE_CHOICES = [("candidate","Candidate"), ("recruiter","Recruiter")]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    phone = models.CharField(max_length=20, blank=True)
    company = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"

# ===== Domain models =====
class CV(BaseModel):
    # map tới cột DB cũ nếu trước đó là user_id
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="cvs",
        null=True,
        blank=True,
        db_column="user_id",   # quan trọng nếu DB cũ dùng user_id
    )
    resume_text = models.TextField()
    cv_location = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        return f"CV#{self.pk}"

class JD(BaseModel):
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="jds",
        null=True,
        blank=True,
        db_column="user_id",
    )
    job_title = models.CharField(max_length=255, blank=True, default="")
    job_requirement = models.TextField(blank=True, default="")
    job_description = models.TextField(blank=True, default="")
    job_location = models.CharField(max_length=255, blank=True, default="")

    def __str__(self):
        return f"JD#{self.pk}"

class Application(BaseModel):
    cv = models.ForeignKey(CV, on_delete=models.CASCADE)
    jd = models.ForeignKey(JD, on_delete=models.CASCADE)
    status = models.CharField(max_length=50, default="applied")

# ===== Signals: auto-create profile =====
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    from .models import UserProfile  # tránh vòng import khi migrate
    if created:
        # mặc định candidate
        UserProfile.objects.create(user=instance, role="candidate")
    else:
        if hasattr(instance, "profile"):
            instance.profile.save()
