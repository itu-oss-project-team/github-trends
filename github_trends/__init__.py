from datetime import date

import csv
import os
from collections import defaultdict

import yaml

with open(os.path.join(os.path.dirname(__file__), 'secret_config.yaml'), 'r') as ymlfile:
    secret_config = yaml.load(ymlfile)

with open(os.path.join(os.path.dirname(__file__), 'research_repos.csv'), 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    category_repos = defaultdict(list)
    for row in reader:
        category_repos[row['category']].append(row)


date_range_start = date(2014, 1, 1)
date_range_end = date(2017, 12, 31)

