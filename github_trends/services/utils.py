from datetime import datetime


def parse_github_datetime(datetime_string):
    return datetime.strptime(datetime_string, "%Y-%m-%dT%H:%M:%SZ")