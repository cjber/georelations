from allennlp_models.pretrained import load_predictor
import torch
from tqdm import tqdm

from src.common.dbpedia_query import clean_text, dbpedia_query


if __name__ == "__main__":
    text = dbpedia_query(max_returns=100_000)
    text = clean_text(text)
    text = text.text.sample(n=200).tolist()

    predictor = load_predictor("coref-spanbert")
    predictor.cuda_device = 0 if torch.cuda.is_available else -1

    resolved = [predictor.coref_resolved(i) for i in tqdm(text)]
