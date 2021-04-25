from datasets import load_dataset

dataset = load_dataset("text", data_files={"val": "data/relations.txt"})
dataset["val"]["text"]
