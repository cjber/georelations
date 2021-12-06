import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

df = pd.read_csv("data/out/triples.csv")
counts = (
    df[df["rel"] == "contains"][["head", "tail"]]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={0: "weight"})
)
df = df[df["rel"] == "contains"][["head", "tail"]].drop_duplicates()
G = nx.from_pandas_edgelist(
    counts, source="tail", target="head", edge_attr="weight", create_using=nx.DiGraph
)
nx.draw(G, with_labels=True)
plt.show()
