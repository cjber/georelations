import gzip
import spacy
from itertools import groupby
from pathlib import Path
from spacy.training import offsets_to_biluo_tags
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm
from urllib.error import HTTPError
from urllib.parse import unquote


def construct_query(item):
    return f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbr: <http://dbpedia.org/resource>
    PREFIX dbo: <http://dbpedia.org/ontology>

    SELECT DISTINCT ?obj WHERE {{
        {item} rdf:type ?obj
        FILTER strstarts(str(?obj), str(dbo:))
    }}
    """


SPARQL = SPARQLWrapper("http://dbpedia.org/sparql")
SPARQL.setReturnFormat(JSON)
DBPEDIA = [
    "http://dbpedia.org/ontology/Place",
    "http://dbpedia.org/ontology/Location",
    "http://dbpedia.org/ontology/AdministrativeRegion",
]

for file in tqdm(Path("data/dbpedia_abs").iterdir()):
    all_items = []
    with gzip.open(file, "rb") as f:
        for divider, lines in tqdm(groupby(f, lambda x: x.decode().strip() == "")):
            lines = [line.decode() for line in lines]
            if divider or lines[0].startswith("@prefix"):
                continue
            elif "nif:Context" in lines[1]:
                try:
                    all_items.append(item)
                except NameError:
                    print("Starting new item.")
                item = {}
                entry = [line.strip() for line in lines]
                passage = entry.pop(2).split('"""')[1]
                item["content"] = passage
                item["entities"] = []

                url = lines[0].split("abstract#")[0]
                item["url"] = url
                name = unquote(url.split("/")[-2].replace("_", " "))
                try:
                    SPARQL.setQuery(construct_query(url + ">"))
                    results = SPARQL.query().convert()
                    for result in results["results"]["bindings"]:  # type: ignore
                        if result["obj"]["value"] in DBPEDIA:
                            try:
                                begin_idx = passage.index(name)
                                end_idx = begin_idx + len(name)
                                item["entities"].append((begin_idx, end_idx, "PLACE"))
                                print(f"Adding {name} as place!: {begin_idx, end_idx}")
                            except ValueError:
                                print(f"{name} not found in text!")
                except:
                    print(f"Error finding context: {entry}!")
            else:
                entry = [line.strip().split() for line in lines[1:]]
                entry = {i[0]: i[1] for i in entry}
                try:
                    SPARQL.setQuery(construct_query(entry["itsrdf:taIdentRef"]))
                    results = SPARQL.query().convert()
                    for result in results["results"]["bindings"]:  # type: ignore
                        if result["obj"]["value"] in DBPEDIA:
                            begin_idx = int(entry["nif:beginIndex"].split('"')[1])
                            end_idx = int(entry["nif:endIndex"].split('"')[1])

                            span = (begin_idx, end_idx, "PLACE")
                            if span not in item["entities"]:
                                item["entities"].append(span)
                                print(f"Adding place: {begin_idx, end_idx}")
                except:
                    print(f"Error finding {entry}!")

nlp = spacy.load("en_core_web_sm")

with open("data/ger/distant.txt", "w") as fp:
    for item in all_items:
        doc = nlp(item["context"])
        entities = item["entities"]
        tags = offsets_to_biluo_tags(doc, entities)
        for token, tag in zip(doc, tags):
            fp.write(f"{token} {tag}\n")
        fp.write("\n")
