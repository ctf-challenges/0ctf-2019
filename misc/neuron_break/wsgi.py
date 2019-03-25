#!/usr/bin/python3
#this file is used for release...

import os
import sys
__path = os.path.dirname(os.path.abspath(__file__))
os.chdir(__path)
sys.path.insert(0, __path)
from app import app as application

if __name__ == "__main__":
    application.run()
