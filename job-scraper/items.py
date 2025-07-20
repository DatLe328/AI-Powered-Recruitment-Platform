import csv
class Jobs:
    def __init__(self, title, company, location, description,url):
        self.title = title
        self.company = company
        self.location = location
        self.description = description
        self.url = url

    def __repr__(self):
        return f"<JobDescription {self.title} @ {self.company}>"

    def to_dict(self):
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "url": self.url,
            "description": self.description
        }

    def summary(self):
        return f"{self.title} at {self.company} in {self.location}\nURL: {self.url}"

    def print(self):
        print("Job title:", self.title)
        print("Company:", self.company)
        print("Location:", self.location)
        print("Description:", self.description)
        print("URL:", self.url)