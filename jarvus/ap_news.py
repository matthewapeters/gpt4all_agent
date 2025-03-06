"""
jarvus.ap_news
"""

from bs4 import BeautifulSoup
import requests
import lxml
from datetime import datetime
import os

url = "http://apnews.com"


def get_latest_news():
    resp = requests.get(url, allow_redirects=True)
    print(resp.status_code)
    results: bytes = resp.content.decode("utf8")
    soup = BeautifulSoup(results, features="lxml")
    items = soup.find_all("li")
    stories = []
    for item in items:
        for link in item.find_all("a"):
            if (
                "https://apnews.com/article/" in link.attrs["href"]
                and len(link.text.strip()) > 1
            ):
                stories.append(link)

    stories.sort(key=lambda x: x.text)
    stories = set(stories)
    for s in stories:
        print(s.text, s.attrs["href"])

    with open("./daily_news/today.txt", "w", encoding="utf8") as feed:
        for s in stories:
            feed.write(f"[headline:] {s.text}\n[url:] {s.attrs['href']}\n")


def get_article(story_url: str):
    today = datetime.now().strftime(r"%Y%m%d")

    article_req = requests.get(story_url, allow_redirects=True)
    article = BeautifulSoup(article_req.content.decode("utf8"), "html.parser")
    story = story_url.split("/")[-1].split("?")[0]
    os.makedirs(f"./daily_news/{today}")
    with open(f"./daily_news/{today}/{story}.txt", "w", encoding="utf8") as fh:
        for paragraph in article.find(class_="RichTextStoryBody RichTextBody").find_all(
            "p"
        ):
            fh.write(f"{paragraph.text}\n")
