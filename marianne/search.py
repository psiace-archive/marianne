"""To get data from database"""
# marianne/search.py

from .db import select_all_metadata


def search_metadata_by_query(query):
    """Search metadata by query"""
    rows = select_all_metadata()
    result = []
    for row in rows:
        if (query in row[1] # URL
            and row[1].count('/') <= 3
            and (row[1].count('.') == 1
            or (row[1].startswith('https://www.')
                and row[1].count('.') == 2))
                and '?' not in row[1]):
            result.insert(0, row)
        elif any(query in s for s in row):
            result.append(row)
    return result
