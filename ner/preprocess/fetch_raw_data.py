import requests
import json
import pandas as pd
from pandas import json_normalize


auth_endpoint = "https://auth.emsicloud.com/connect/token"

client_id = "uwak52pvokbwz3cc"
client_secret = "ta7FNnY0"
scope = "emsi_open"

payload = f"client_id={client_id}&client_secret={client_secret}&grant_type=client_credentials&scope={scope}"
headers = {'content-type': 'application/x-www-form-urlencoded'}
access_token = json.loads((requests.request("POST", auth_endpoint, data=payload, headers=headers)).text)['access_token']


def fetch_skills_list() -> pd.DataFrame:
    all_skills_endpoint = "https://emsiservices.com/skills/versions/latest/skills"
    auth = f"Authorization: Bearer {access_token}"
    headers = {'authorization': auth}
    response = requests.request(
        "GET", all_skills_endpoint, headers=headers)
    response = response.json()['data']

    return response

if __name__ == "__main__":
    print()
    # skills = fetch_skills_list()
    # df = json_normalize(skills)
    # df.to_json("../data/raw/raw_skills_list.json", index=False)