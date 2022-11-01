"""To get data from database"""
# marianne/search.py

from flask import Blueprint, render_template, request

from .crawler import url_crawler
from .db import select_all_metadata

bp = Blueprint("search engine", __name__)


@bp.route("/")
def index():
    query = request.args.get("query")
    page = request.args.get("page")
    if query is None or not query.strip():
        return render_template("index.html")
    else:
        if page is None:
            page = 1
        else:
            try:
                page = int(page)
            except BaseException:
                page = 1
        list = search_metadata_by_query(query)
        start = 0 if page == 1 else 10 * (page - 1)
        results = list[start : start + 10]
        page_num = 0
        if len(list) % 10 == 0 & len(list) != 0:
            page_num = len(list) / 10
        else:
            page_num = int(len(list) / 10) + 1
        return render_template(
            "query.html",
            query=query,
            results=results,
            page_num=page_num,
            result_num=len(list),
        )


@bp.route("/about")
def about():
    return render_template("about.html")


@bp.route("/submit")
def new():
    query = request.args.get("query")
    if query is None or not query.strip():
        return render_template("submit.html")
    else:
        if len(query) <= 100:
            url_crawler(query)
            return "Your URL has been submitted, it should be available for everyone!"
        else:
            return "That URL is over 100 characters long!"


def search_metadata_by_query(query):
    """Search metadata by query"""
    rows = select_all_metadata()
    result = []
    for row in rows:
        if (
            query in row[1]  # URL
            and row[1].count("/") <= 3
            and (
                row[1].count(".") == 1
                or (row[1].startswith("https://www.") and row[1].count(".") == 2)
            )
            and "?" not in row[1]
        ):
            result.insert(0, row)
        elif any(query in s for s in row):
            result.append(row)
    return result
