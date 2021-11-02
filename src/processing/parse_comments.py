import jsonlines
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from itertools import filterfalse
from src.common.utils import Const
from tqdm import tqdm

text_processor = TextPreProcessor(
    normalize=Const.NORMALIZE,
    annotate={"hashtag"},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    dicts=[emoticons],
)

liverpool = []
with jsonlines.open("data/reddit_comments/comments.json") as f:
    for line in tqdm(f, total=23_609_335):
        if line["subreddit"] == "Liverpool":
            liverpool.append(line["text"])


liverpool_processed = filterfalse(
    lambda x: len(x) < 5, text_processor.pre_process_docs(liverpool)
)
next(liverpool_processed)
