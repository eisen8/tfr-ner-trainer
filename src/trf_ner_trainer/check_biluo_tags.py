import spacy
from spacy import displacy
from spacy.training import offsets_to_biluo_tags

# Example text and entities
text = "Kaos S01E04 720p NF WEBRip x264 GalaxyTV"
entities = [(0, 4, 'NAME'), (5, 7, 'SEASON'), (8, 12, 'EPISODE')]  # Replace with your actual entities

# Load spaCy model
nlp = spacy.load("trained_ner_model")

# Create a spaCy doc from the text
doc = nlp.make_doc(text)

# Use offsets_to_biluo_tags to check alignment
biluo_tags = offsets_to_biluo_tags(doc, entities)

# Print BILOU tags for verification
print(f"BILOU Tags: {biluo_tags}")