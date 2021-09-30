import gzip
import pandas as pd

for line in gzip.open("/mnt/liv/freebase-rdf-latest.gz", "r"):
    line = line.decode("utf-8").split("\t")[:-1]
    if "contains" in line[1]:
        print(line)
