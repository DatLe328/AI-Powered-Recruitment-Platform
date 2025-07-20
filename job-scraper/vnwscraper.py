from selenium import webdriver
from bs4 import BeautifulSoup
import time
from items import Jobs
from cleaner import *
import fasttext
import utils
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

model = fasttext.load_model("lid.176.bin")

def scroll_to_bottom(driver):
    scroll_pause = 0.15
    step = 300
    last_height = driver.execute_script("return document.body.scrollHeight")
    current_pos = 0

    while current_pos < last_height:
        current_pos += step
        driver.execute_script(f"window.scrollTo(0, {current_pos});")
        time.sleep(scroll_pause)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height > last_height:
            last_height = new_height
        elif current_pos >= new_height:
            break

def parse_job_list(driver, url):
    driver.get(url)
    scroll_to_bottom(driver)

    # Get HTML source
    html = driver.page_source
    time.sleep(1)

    soup = BeautifulSoup(html, 'html.parser')

    # Get job links
    try:
        job_links = []
        jobs = soup.select('div[data-cname="preview-job"]')
        for job in jobs:
            a_tag = job.find('a')
            if a_tag:
                link = a_tag.get('href')
                if link.startswith('/'):
                    link = "https://www.vietnamworks.com" + link
                job_links.append(link)
        if len(job_links) == 0:
            print("No jobs available")
            return
        print(f"Tìm thấy {len(job_links)} công việc")
        print("-" * 50)
    except Exception as e:
        print(e)
        return

    job_titles = []
    company_names = []
    location_list = []
    all_cards = soup.select('.new-job-card')
    for i in all_cards:
        span_tags = i.find_all('span')
        a_tags = i.find_all('a')

        job_titles.append(clean_job_title(a_tags[0].get('title')))
        # location_list.append(span_tags[3].get_text())
        location_list.append("None")
        company_names.append(clean_link_in_text(a_tags[2].get_text()))
        print(job_titles[-1])
        print(location_list[-1])
        print(company_names[-1])

    jobs_items = []
    card_idx = 0
    for _, link in enumerate(job_links, 1):
        [safe, job] = parse_job(link, company_names[card_idx], location_list[card_idx])
        if safe:
            jobs_items.append(job)
        card_idx += 1
    for i in jobs_items:
        i.print()
    utils.save_jobs_to_csv(jobs_items)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    current_page_list = query_params.get('page')
    if current_page_list:
        current_page = int(current_page_list[0])
    else:
        current_page = 1
    next_page = current_page + 1
    query_params['page'] = [str(next_page)]
    new_query_string = urlencode(query_params, doseq=True)

    next_url = urlunparse((parsed_url.scheme,
                          parsed_url.netloc,
                          parsed_url.path,
                          parsed_url.params,
                          new_query_string,
                          parsed_url.fragment))
    parse_job_list(driver, next_url)


def parse_job(link, company_name, location_name):
    driver.get(link)
    time.sleep(2)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    jd_title = soup.select_one('h1[name="title"]').get_text()

    container = soup.select_one('#vnwLayout__col > div > div.sc-1671001a-0.bbOaea')

    paragraphs = container.find_all('p')
    print("Current at:", link)
    print(jd_title)
    desc = ""
    skip_list = [
        "Mô tả công việc",
        "Yêu cầu công việc"
    ]
    for p in paragraphs:
        if len(p.get_text()) < 10:
            continue
        if p.get_text(strip=True) in skip_list:
            continue
        text = clean_front_punctuation_mark(p.get_text(strip=True))

        desc = desc + text
        pred = model.predict(text)

        lang = pred[0][0].replace("__label__", "")

        if lang != 'en':
            print("Error at:", text)
            return [False, None]
    return [True, Jobs(title=jd_title, company=company_name, location=location_name, url=link,description=desc)]

if __name__=="__main__":
    url = "https://www.vietnamworks.com/viec-lam?g=25"

    # cấu hình Chrome
    options = webdriver.ChromeOptions()
    options.add_argument("--start-minimized")

    driver = webdriver.Chrome(options=options)

    parse_job_list(driver, url)