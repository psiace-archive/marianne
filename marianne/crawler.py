"""The web crawler for marianne."""
# marianne/crawler.py

from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .db import insert_metadata


def url_crawler(url, limit=0):
    try:
        url = url_sanitize(url)
        # requestuest, if this fails, it jumps to except
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        result = urlopen(request)
        result_string = result.read().decode("utf8")
        result.close()
        print("[!] Success fetching ->", url)
    except Exception as e:
        print("[!] Error fetching a submitted website (", url, ") -> ", e)
    else:
        # the website has been reached (its contents too),
        # add it to the db then
        html = BeautifulSoup(result_string, "html.parser")
        title = get_title(html)  # type: ignore
        # found a bug here with some websites
        # here it is a fix
        if title is None:
            title = "No title provided"
        desc = get_description(html)  # type: ignore
        if desc is None:
            desc = "No description provided"
        insert_metadata((str(title), url, str(desc)))

        if limit > 0:
            crawl_more_url(url, html, limit)


def crawl_more_url(url, html, limit):
    """Crawl the web."""
    # now, crawl the result of the website with a limit
    count = 0
    urls = []
    for tag in html.find_all("a"):
        urls.append(tag.get("href"))
    for i in urls:
        if count <= limit and i != url and i:
            if i.startswith("https://") or i.startswith("http://"):
                url_crawler(i, 0)
            else:
                url_crawler(urljoin(url, i), 0)
            count += 1


def url_sanitize(url):
    """Sanitize url."""
    if (url.startswith('"') and url.endswith('"')) or (
        url.startswith("'") and url.endswith("'")
    ):
        url = url[1:-1]
    if url.startswith("//"):
        url = "https://" + url[2:]
    if url.endswith("/"):
        url = urljoin(url, ".")
    while "/./" in url:
        url = url.replace("/./", "/")
    return url


def get_title(html):
    """Scrape page title."""
    title = None
    if html.find("title"):
        title = html.find("title").string
    elif html.find("meta", property="og:title"):
        title = html.find("meta", property="og:title").get("content")
    elif html.find("meta", property="twitter:title"):
        title = html.find("meta", property="twitter:title").get("content")
    elif html.find("h1"):
        title = html.find("h1").string
    return title


def get_description(html):
    """Scrape page description."""
    description = None
    if html.find("meta", property="description"):
        description = html.find("meta", property="description").get("content")
    elif html.find("meta", property="og:description"):
        description = html.find("meta", property="og:description").get("content")
    elif html.find("meta", property="twitter:description"):
        description = html.find("meta", property="twitter:description").get("content")
    elif html.find("p"):
        description = html.find("p").contents
    return description
