import logging
from psaw import PushshiftAPI
from tqdm import tqdm

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger = logging.getLogger("psaw")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

SUBREDDITS = [
    "unitedkingdom",
    "Britain",
    "England",
    "thenorth",
    "birkenhead",
    "carlisle",
    "Chester",
    "cheshire",
    "cumbria",
    "darwen",
    "fylde",
    "Lancaster_uk",
    "LancashireProblems",
    "Liverpool",
    "liverpoolcity",
    "Manchester",
    "cyclemcr",
    "merseyside",
    "mossley",
    "preston",
    "southport",
    "warrington",
    "wigan",
    "wirral",
    "northeast",
    "alnwick",
    "ashington",
    "durhamuk",
    "morpeth",
    "newcastleupontyne",
    "Northumberland",
    "peterlee",
    "sunderland",
    "teesside",
    "Tyneside",
    "whitleybay",
    "bradford",
    "dewsbury",
    "doncaster",
    "goldsborough",
    "grimsby",
    "harrogate",
    "huddersfield",
    "hull",
    "leeds",
    "leedscycling",
    "lincolnshire",
    "ripponden",
    "scarboroughuk",
    "Sheffield",
    "wakefield",
    "westyorkshire",
    "York",
    "yorkshire",
    "midlands",
    "westmidlands",
    "brum",
    "theblackcountry",
    "coventry",
    "hereford",
    "kenilworth",
    "malvern/",
    "shrewsburyuk",
    "shropshire",
    "stafford",
    "staffordshire",
    "warwickshire",
    "wolverhampton",
    "worcester",
    "eastmidlands",
    "beeston",
    "caistor",
    "chesterfielduk",
    "cleethorpes",
    "corby",
    "daventry",
    "derby",
    "derbyshire",
    "duffield",
    "leicester",
    "lincolnshire",
    "louth",
    "northamptonians",
    "nottingham",
    "nottinghamshire",
    "rutland",
    "worksop",
    "westcountry",
    "bath",
    "bournemouth",
    "braunton",
    "bristol",
    "BristolCycling",
    "chippenham",
    "cirencester",
    "cornwall",
    "dartmoor",
    "devonuk",
    "dorset",
    "exeter",
    "exmoor",
    "gloucestershire",
    "plymouth",
    "salisburyUK",
    "saltash",
    "PooleBayCity",
    "swindon",
    "westlynndevon",
    "wiltshire",
    "yeovil",
    "SouthEastEngland",
    "amersham",
    "basingstoke",
    "bexhill",
    "brighton",
    "broadstairs",
    "canterbury",
    "chichester",
    "crawley",
    "egham",
    "graveney",
    "hampshire",
    "hastings",
    "highwycombearea",
    "isleofwight",
    "britishkent",
    "margate",
    "medway",
    "midhurst",
    "miltonkeynes",
    "newforest",
    "oxford",
    "Portsmouth",
    "ramsgate",
    "reading",
    "slough",
    "southampton",
    "Staines",
    "surrey",
    "tadley",
    "thanet",
    "Tunbridgewells/",
    "winchesterUK",
    "wokingham",
    "worthing",
    "London",
    "aces",
    "bromley",
    "clapham",
    "croydon",
    "islington",
    "raynes_park",
    "romford",
    "eastanglia",
    "bedfordshire",
    "cambridge",
    "cambridgeshire",
    "cambridgecycling",
    "ely",
    "Essex",
    "fenland",
    "greatyarmouth",
    "harpenden",
    "ipswichuk",
    "kingslynn",
    "luton",
    "maldon",
    "norfolkuk",
    "norwich",
    "rayleigh",
    "stalbans",
    "sudburysuffolk",
    "suffolk",
    "watford",
    "northernireland",
    "Belfast",
    "carrickfergus",
    "derrylondonderry",
    "lisburn",
    "Newry",
    "scotland",
    "aberdeen",
    "annan",
    "arbroath",
    "ayrshire",
    "dundee",
    "DumfriesAndGalloway",
    "Edinburgh",
    "eastkilbride",
    "elginscotland",
    "falkirk",
    "fife",
    "Glasgow",
    "glencoe",
    "inverness",
    "irvinescotland",
    "kettins",
    "kilmarnock",
    "lanarkshire",
    "orkney",
    "shetland",
    "stirlingscotland",
    "stornoway",
    "westernisles",
    "whitburn",
    "Wales",
    "HistoryWales",
    "abergavenny",
    "abertillery",
    "aberystwyth",
    "bangor",
    "bridgend",
    "builthwells",
    "cardiff",
    "flintshire",
    "holyhead",
    "newportSW",
    "pembrokeshire",
    "porthcawl",
    "southwales",
    "swansea",
    "wrexham",
    "isleofman",
    "guernsey",
]

api = PushshiftAPI()
for subreddit in SUBREDDITS:
    gen = api.search_comments(subreddit=subreddit)

    with open(f"data/comments/{subreddit}.txt", "w") as f:
        for comment in tqdm(gen):
            f.write(comment.body)
            f.write("\n")