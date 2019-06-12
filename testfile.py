import argparse
import warnings
import os
from pprint import pprint

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='test about card number search')
parser.add_argument('--log-dir', default=None, type=str)
