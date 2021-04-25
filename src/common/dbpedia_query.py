from io import BytesIO

from SPARQLWrapper import CSV, SPARQLWrapper
import pandas as pd


def dbpedia_query(max_returns: int):
    csv = pd.DataFrame()
    # loop to overcome 10000 query limit
    for offset in range(0, max_returns, 10000):
        print(f"Getting {offset} of {max_returns}")
        query = """
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX res: <http://dbpedia.org/resource/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX dbp: <http://dbpedia.org/property/>
        PREFIX georss: <http://www.georss.org/georss/>

        SELECT DISTINCT ?label ?text ?point
        WHERE {
                { ?uri dbo:country res:England } UNION
                { ?uri dbo:country res:United_Kingdom } UNION
                { ?uri dbo:country res:Scotland } UNION
                { ?uri dbo:country res:Wales } UNION
                { ?uri dbo:location res:England } UNION
                { ?uri dbo:location res:United_Kingdom } UNION
                { ?uri dbo:location res:Scotland } UNION
                { ?uri dbo:location res:Wales } .

                { ?uri rdf:type dbo:Place }
                  ?uri rdfs:label ?label . FILTER (lang(?label) = 'en')
                  ?uri dbo:abstract ?text . FILTER (lang(?text) = 'en')
                  OPTIONAL { ?uri georss:point ?point }
        }
        LIMIT 10000 OFFSET
        """ + str(
            offset
        )
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat(CSV)
        results = sparql.query().convert()
        df = pd.read_csv(BytesIO(results), dtype=str)
        csv = csv.append(df)
    return csv


def clean_text(csv):
    csv = csv.drop_duplicates(subset=["text"]).reset_index().drop("index", axis=1)
    csv["text"] = csv["text"].str.encode("ascii", "ignore").str.decode("ascii")
    csv["text"] = csv["text"].str.replace(r'"', "")
    csv["text"] = csv["text"].str.replace(r"'", "")
    csv["text"] = csv["text"].str.replace(r"-", " ")
    csv["text"] = csv["text"].str.replace(r"*", "", regex=False)
    # remove brackets and contents
    csv["text"] = csv["text"].str.replace(r"\(.*\)", "", regex=True)
    csv["text"] = csv["text"].str.replace(r"\(|\)", "", regex=True)
    # removes double spaces
    csv["text"] = csv["text"].str.split().str.join(" ")
    # remove south georgia
    csv = csv[~csv["text"].str.contains("South Georgia")]
    csv = csv[csv["text"].str.len() > 50]  # remove very short abstracts
    return csv
