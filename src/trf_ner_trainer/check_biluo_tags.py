import spacy
from spacy import displacy
from spacy.training import offsets_to_biluo_tags

from src.trf_ner_trainer.tokenizer import custom_tokenizer

# Example text and entities
texts = [
    "Kaos S 01 E 04 720p NF WEBRip x264 GalaxyTV",
    "Searching 2018 1080p WEBRip x264 [YTS AM]",
    "[ENTE] Dragon Ball (1986) S 06 E 06 [AV1] [OPUS] [DVD] [480p]",
    "The Big Bang Theory S 11 E 22 720p HDTV x264 AVS [eztv]",
    "What If 2021 S 01 E 08 720p WEBRip x265 MiNX",
    "Spider Man Far From Home 2019 NEW 720p HD TC X264 1XBET"
    ]
#entities = [(0, 4, 'NAME'), (5, 7, 'SEASON'), (8, 12, 'EPISODE')]  # Replace with your actual entities

# Load spaCy model
# nlp = spacy.load("en")
nlp = spacy.load("trained_ner_model2")
nlp.tokenizer = custom_tokenizer(nlp)

# Create a spaCy doc from the text
for text in texts:
    doc = nlp(text)

    # Access tokens, named entities, and parts of speech
    print(f"---- {text}")
    tokens = [token.text for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Tokens {tokens}")
    print(f"Entities {entities}")