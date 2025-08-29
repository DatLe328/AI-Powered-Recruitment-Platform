import os
import django
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recruitapi.settings")

@pytest.fixture(scope="session", autouse=True)
def django_setup():
    django.setup()
