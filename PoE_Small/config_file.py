import argparse
import json
from main import RunFramework

parser = argparse.ArgumentParser(description='Pool of Experts - command line interface.')
parser.add_argument('--config-file',
                    type=str,
                    help='Path to config file',
                    required=True)

# Parse the arguments
args = parser.parse_args()
config_file = args.config_file
#  load the config file( that is a json)
with open(config_file, 'r') as json_file:
    args_dict = json.load(json_file)

print('\n\n\n\t\t\t\t***** starting experiment ******\n\n\n')
RunFramework(args_dict)
