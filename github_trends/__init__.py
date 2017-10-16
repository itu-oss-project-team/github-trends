import os
import yaml

with open(os.path.join(os.path.dirname(__file__), 'secret_config.yaml'), 'r') as ymlfile:
    secret_config = yaml.load(ymlfile)
