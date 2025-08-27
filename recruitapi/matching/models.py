from django.db import models
from django.contrib.auth.models import User

# Postgres features
from django.contrib.postgres.fields import ArrayField
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVector, SearchVectorField
from pgvector.django import VectorField, HnswIndex  # HNSW (nếu không hỗ trợ, có thể đổi sang IvfflatIndex)



class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        abstract = True


EMPLOYMENT_TYPES = [
    ("full-time", "Full-time"),
    ("part-time", "Part-time"),
    ("contract",  "Contract"),
    ("intern",    "Intern"),
]

APPLICATION_STATUS = [
    ("applied",   "Applied"),
    ("reviewing", "Reviewing"),
    ("rejected",  "Rejected"),
    ("accepted",  "Accepted"),
]


class CV(TimeStampedModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="cvs")
    title = models.CharField(max_length=255)

    # Nội dung
    resume_text = models.TextField(blank=True, default="")
    summary = models.TextField(blank=True, default="")
    skills = ArrayField(models.CharField(max_length=100), blank=True, default=list)
    experience = models.JSONField(blank=True, default=list)  # [{company, role, years, description}]
    file_url = models.URLField(blank=True, null=True)

    # Tìm kiếm full-text: dùng cột tsvector
    search_vector = SearchVectorField(null=True)

    # Embedding (nếu có pgvector)
    emb = VectorField(dimensions=768, null=True, blank=True, editable=True)

    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.title} ({self.user.username})"

    def save(self, *args, **kwargs):
        """
        Lưu bình thường rồi cập nhật search_vector bằng biểu thức DB.
        (Làm 2 bước vì SearchVector là expression, không thể gán trực tiếp trước khi insert.)
        """
        super().save(*args, **kwargs)
        type(self).objects.filter(pk=self.pk).update(
            search_vector=(
                SearchVector("title", config="simple") +
                SearchVector("summary", config="simple") +
                SearchVector("resume_text", config="simple")
            )
        )
    class Meta:
        indexes = [
            GinIndex(fields=["search_vector"], name="cv_search_gin"),
            models.Index(fields=["-updated_at"], name="cv_updated_idx"),
        ] + ([HnswIndex(fields=["emb"], name="cv_emb_hnsw")])


class JD(TimeStampedModel):
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="jds")
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    requirements = models.TextField(blank=True, default="")
    location = models.CharField(max_length=255, blank=True, default="")

    skills = ArrayField(models.CharField(max_length=100), blank=True, default=list)
    employment_type = models.CharField(max_length=20, choices=EMPLOYMENT_TYPES, default="full-time")
    salary_min = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    salary_max = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)

    search_vector = SearchVectorField(null=True)

    emb = VectorField(dimensions=768, null=True, blank=True, editable=True)

    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        type(self).objects.filter(pk=self.pk).update(
            search_vector=(
                SearchVector("title", config="simple") +
                SearchVector("description", config="simple") +
                SearchVector("requirements", config="simple")
            )
        )

    class Meta:
        indexes = [
            GinIndex(fields=["search_vector"], name="jd_search_gin"),
            models.Index(fields=["-created_at"], name="jd_posted_idx"),
        ] + ([HnswIndex(fields=["emb"], name="jd_emb_hnsw")])


class Application(TimeStampedModel):
    job = models.ForeignKey(JD, on_delete=models.CASCADE, related_name="applications")
    candidate = models.ForeignKey(User, on_delete=models.CASCADE, related_name="applications")
    cv = models.ForeignKey(CV, on_delete=models.SET_NULL, null=True, blank=True, related_name="applications")
    status = models.CharField(max_length=16, choices=APPLICATION_STATUS, default="applied")

    class Meta:
        unique_together = (("job", "candidate"),)
        indexes = [
            models.Index(fields=["job", "candidate"], name="app_job_cand_idx"),
            models.Index(fields=["-created_at"], name="app_created_idx"),
        ]

    def __str__(self):
        return f"{self.candidate.username} → {self.job.title} ({self.status})"
