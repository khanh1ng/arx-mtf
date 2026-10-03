import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arxmtf import data
if __name__ == "__main__":
    print("universe:", len(data.build_cache()))
