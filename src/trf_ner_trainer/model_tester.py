import json
import re
import sqlite3

import spacy
from spacy.training import offsets_to_biluo_tags
from common.constants import Constants as C
from tokenizer import custom_tokenizer

def add_indices(data):
    for entry in data:
        filename = entry["filename"]
        used_indices = set()  # Track already used character positions

        for annotation in entry["annotations"]:
            text = annotation["text"]

            # Find all occurrences of the text in the filename
            matches = [m.start() for m in re.finditer(re.escape(text), filename)]

            # Find the first unused index
            start = next((m for m in matches if m not in used_indices), -1)

            if start == -1:
                print(entry)
                #raise Exception(f"Could not find {text} in {filename}")
                print(f"Could not find {text} in {filename}")

            end = start + len(text)
            annotation["start"] = start
            annotation["end"] = end

            # Mark this range as used
            used_indices.update(range(start, end))

    return data


def verify_indices(data):
    for entry in data:
        filename = entry["filename"]
        used_indices = set()
        for annotation in entry["annotations"]:
            start = annotation.get("start")
            end = annotation.get("end")
            text = annotation["text"]

            # Check if start and end are valid
            if start is None or end is None:
                raise Exception(f"Start or end missing in label {text} for {filename}")

            extracted_text = filename[start:end]
            if extracted_text != text:
                print(f"Invalid annotation: '{text}' does not match extracted text '{extracted_text}' (from indices {start}-{end}) for {filename}")

            # Check for overlap
            overlap = any(index in used_indices for index in range(start, end))
            if overlap:
                print(f"Overlap detected: Annotation '{text}' overlaps with a previous annotation (from indices {start}-{end}) for {filename}")

            used_indices.update(range(start, end))


if __name__ == "__main__":
    # Load trained model
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.blank("en")
    nlp.tokenizer = custom_tokenizer(nlp)

    # Load training data
    # with open(C.TRAINING_DATA_FOLDER_PATH / "response_0_updated.json", "r", encoding="utf-8") as f:
    #   data = json.load(f)
    data = []
    print(C.DB_FILE_PATH.resolve())
    _conn = sqlite3.connect(str(C.DB_FILE_PATH))
    _conn.row_factory = sqlite3.Row  # Treat rows as dictionaries rather than tuples
    _cursor = _conn.cursor()
    _cursor.execute("SELECT annotation_json FROM annotations WHERE annotation_json IS NOT NULL")
    rows = [row[0] for row in _cursor.fetchall()]
    for row in rows:
        j = json.loads(row)
        data.append(j)

    data = add_indices(data)
    verify_indices(data)

    #with open(C.TRAINING_DATA_FOLDER_PATH / "response_0_updated.json", "r", encoding="utf-8") as f:
    #    json_data = json.load(f)

    # Convert data into spaCy format
    training_data = []
    for entry in data:
        text = entry["filename"]
        entities = []
        for ann in entry["annotations"]:
            entities.append((ann['start'], ann['end'], ann["label"]))
            # ner.add_label(ann["label"])
        training_data.append((text, {"entities": entities}))

    filtered_data = training_data
    # filtered_data = [item for item in training_data if item[0] == "Legacies.S01E12.iNTERNAL.480p.x264-mSD[eztv]"]


    # Test it on a new filename
    for data in filtered_data:
        text = data[0]
        text = text.replace(".", " ")
        text = text.replace("[", " ")
        text = text.replace("]", " ")
        print(f"--- processing {text} ----")
        print(data[1]['entities'])
        doc = nlp(text)
        for ent in doc.ents:
            print(f"doc ents {ent.text} -> {ent.label_}")
        print(f"Tokens {[token.text for token in doc]}")
        biluo_tags = offsets_to_biluo_tags(doc, data[1]['entities'])
        print(f"BILOU Tags: {biluo_tags}")


    #
    #
    # text = "Boob.S05E08.721p.NF.WEBRip.x265-GalaxyTV"
    # entities = [(0, 4, 'NAME'), (6, 8, 'SEASON'), (9, 11, 'EPISODE'), (12, 16, 'VIDEO_QUALITY'), (17, 19, 'SOURCE'), (20, 26, 'SOURCE'), (27, 31, 'COMPRESSION'), (32, 40, 'RELEASE_GROUP')]  # Replace with your actual entities
    # doc = nlp(text)
    #
    # # Print detected entities
    # for ent in doc.ents:
    #     print(f"{ent.text} -> {ent.label_}")
    #
    # print([token.text for token in doc])
    #
    # # Create a spaCy doc from the text
    # doc = nlp.make_doc(text)
    #
    # # Use offsets_to_biluo_tags to check alignment
    # biluo_tags = offsets_to_biluo_tags(doc, entities)
    #
    # # Print BILOU tags for verification
    # print(f"BILOU Tags: {biluo_tags}")