import pandas as pd
from tqdm import tqdm
from xml.etree import ElementTree

RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
DEFINEDBY = "{http://www.w3.org/2000/01/rdf-schema#}isDefinedBy"
NAME = "{http://www.geonames.org/ontology#}name"
ALTNAME = "{http://www.geonames.org/ontology#}alternateName"
COUNTRYCODE = "{http://www.geonames.org/ontology#}countryCode"
PARENT = "{http://www.geonames.org/ontology#}parentFeature"
PARENTC = "{http://www.geonames.org/ontology#}parentCountry"
NEARBY = "{http://www.geonames.org/ontology#}nearbyFeatures"


items = []
with open("data/distant_data/all-geonames-rdf.txt", "r") as f:
    for line in tqdm(f):
        if not line.startswith("https://sws.geonames.org"):
            line_xml = ElementTree.fromstring(line)[0]
            alt_names = []
            nearby_ids = []
            for child in line_xml:
                if child.tag == DEFINEDBY:
                    place_id = list(child.attrib.values())[0].split("/")[-2]
                    # print(f"{place_id=}")
                elif child.tag == NAME:
                    place_name = child.text
                    # print(f"{place_name=}")
                elif child.tag == ALTNAME:
                    alt_name = child.text
                    # print(f"{alt_name=}")
                    alt_names.append(alt_name)
                elif child.tag == PARENT:
                    parent_id = list(child.attrib.values())[0].split("/")[-2]
                    # print(f"{parent_id=}")
                elif child.tag == PARENTC:
                    parentc_id = list(child.attrib.values())[0].split("/")[-2]
                    # print(f"{parentc_id=}")
                elif child.tag == NEARBY:
                    nearby_id = list(child.attrib.values())[0].split("/")[-2]
                    nearby_ids.append(nearby_id)
                if child.tag == COUNTRYCODE and child.text == "GB":
                    items.append(
                        {
                            DEFINEDBY.split("#}")[1]: place_id,
                            NAME.split("#}")[1]: place_name,
                            ALTNAME.split("#}")[1]: alt_names,
                            PARENT.split("#}")[1]: parent_id,
                            PARENTC.split("#}")[1]: parentc_id,
                            NEARBY.split("#}")[1]: nearby_ids,
                        }
                    )


df = pd.DataFrame(items)


df = df.set_index("isDefinedBy")
df_parent = df.set_index("parentFeature")

df_parent.join(df, rsuffix="_parent")[["name", "name_parent"]].dropna().sample(10)
