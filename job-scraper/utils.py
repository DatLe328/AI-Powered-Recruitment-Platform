import csv
from items import *

def save_jobs_to_csv(jobs, filename="jobs.csv"):
    with open(filename, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["title", "company", "location", "description", "url"]
        )
        writer.writeheader()
        writer.writerows([job.to_dict() for job in jobs])