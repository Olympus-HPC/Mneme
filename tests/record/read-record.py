import json
import glob
import sys
from pathlib import Path


for fn in glob.glob("./*.json"):
    print(fn)
