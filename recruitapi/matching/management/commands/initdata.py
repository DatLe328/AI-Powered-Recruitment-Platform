import json, random
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model
from matching.models import CV, JD, Application   # sửa import nếu khác

class Command(BaseCommand):
    help = "python manage.py initdata matching/fixtures/sample_data.json --apply-random 2"
    # python manage.py initdata data/cvs_jds.json 


    def add_arguments(self, parser):
        parser.add_argument("json_file", type=str, help="Đường dẫn file JSON có cvs/jds")
        parser.add_argument("--owner", type=str, default="seeduser",
                            help="username gán làm owner cho dữ liệu mới")
        # Cách nối CV ↔ JD (chọn 1 trong 3)
        parser.add_argument("--apply-jd", type=int,
                            help="ID JD sẵn có, mọi CV mới sẽ apply vào JD này")
        parser.add_argument("--apply-to-created", action="store_true",
                            help="Mọi CV mới apply vào toàn bộ JD vừa tạo từ file")
        parser.add_argument("--apply-random", type=int, metavar="K",
                            help="Mỗi CV apply ngẫu nhiên vào K JD trong DB")
        parser.add_argument("--random-seed", type=int, default=42)

    def handle(self, *args, **opts):
        random.seed(opts["random_seed"])
        # 1) đọc file
        path = opts["json_file"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            raise CommandError(f"Không đọc được file {path}: {e}")

        # 2) owner
        User = get_user_model()
        owner, _ = User.objects.get_or_create(
            username=opts["owner"],
            defaults={"email": f'{opts["owner"]}@example.com'}
        )
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser(
                username="admin",
                email="admin@example.com",
                password="admin123"
            )
            self.stdout.write(self.style.SUCCESS("Đã tạo admin mặc định (admin / admin123)"))

        # 3) tạo JD
        created_jds = []
        for jd in payload.get("jds", []):
            obj = JD.objects.create(
                owner=owner,
                job_title=jd.get("job_title", "Software developer"),
                job_requirement=jd.get("job_requirement", ""),
                job_description=jd.get("job_description", ""),
                job_location=jd.get("job_location", ""),
            )
            created_jds.append(obj)

        # 4) tạo CV
        created_cvs = []
        for cv in payload.get("cvs", []):
            obj = CV.objects.create(
                owner=owner,
                resume_text=cv.get("resume_text", ""),
                cv_location=cv.get("cv_location", ""),
            )
            created_cvs.append(obj)

        jd_targets = []
        if opts.get("apply_jd"):
            try:
                jd_targets = [JD.objects.get(id=opts["apply_jd"])]
            except JD.DoesNotExist:
                raise CommandError(f'JD id={opts["apply_jd"]} không tồn tại.')
        elif opts.get("apply_to_created"):
            jd_targets = created_jds[:]   # tất cả JD mới tạo
        elif opts.get("apply_random"):
            all_jds = list(JD.objects.all())
        else:
            jd_targets = []

        # 6) tạo Application nối CV ↔ JD
        created_apps = 0
        for cv in created_cvs:
            targets = jd_targets
            if opts.get("apply_random"):
                k = min(opts["apply_random"], len(all_jds))
                targets = random.sample(all_jds, k) if k > 0 else []

            for jd in targets:
                Application.objects.get_or_create(cv=cv, jd=jd)  # tránh trùng
                created_apps += 1

        self.stdout.write(self.style.SUCCESS(
            f"Seed xong: CV={len(created_cvs)}, JD={len(created_jds)}, Application={created_apps}"
        ))
