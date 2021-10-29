import os

import yaml

# BUGFIX: for jupyter, cd in root directory
os.chdir( os.path.dirname(__file__) )

with open('config.yaml') as file:
    config: dict = yaml.load(file.read(), Loader=yaml.FullLoader)
