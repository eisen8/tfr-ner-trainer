import re
import sqlite3

import spacy
import json
import random
from tokenizer import custom_tokenizer
from spacy.training.example import Example
from common.constants import Constants as C


def add_indices(data):
    fail = False
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
                print(f"Could not find '{text}' in '{filename}'")
                fail = True

            end = start + len(text)
            annotation["start"] = start
            annotation["end"] = end

            # Mark this range as used
            used_indices.update(range(start, end))

    if fail:
        raise Exception("Error in annotations")
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


def evaluate(eval_data):
    # After training, evaluate the model on the eval data
    for text, annotations in eval_data:
        doc = nlp(text)  # Get the predicted NER results
        true_entities = annotations["entities"]
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Compare true_entities vs predicted_entities to compute performance metrics (precision, recall, F1-score)
        print("True:", true_entities)
        print("Predicted:", predicted_entities)


if __name__ == "__main__":
    # with open(C.TRAINING_DATA_FOLDER_PATH / "response_0.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    #
    # new_data = add_indices(data)
    # verify_indices(new_data)
    #
    # with open(C.TRAINING_DATA_FOLDER_PATH / "response_0_updated.json", "w", encoding="utf-8") as f:
    #     json.dump(new_data, f, indent=4)

    # Config
    model_name = "trained_ner_model2"

    if spacy.prefer_gpu():
        spacy.require_gpu()  # Use GPU if available
        device = 0  # First GPU
        print("Training on GPU")
    else:
        device = -1  # Use CPU
        print("Training on CPU")
    nlp = spacy.blank("en")  # Use a blank English model
    nlp.tokenizer = custom_tokenizer(nlp)

    # Add NER component if not already present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Load training data
    # with open(C.TRAINING_DATA_FOLDER_PATH / "response_0_updated.json", "r", encoding="utf-8") as f:
    #   data = json.load(f)
    data = []
    _conn = sqlite3.connect(str(C.DB_FILE_PATH))
    _conn.row_factory = sqlite3.Row  # Treat rows as dictionaries rather than tuples
    _cursor = _conn.cursor()
    _cursor.execute("SELECT annotation_json FROM annotations WHERE annotation_json IS NOT NULL")
    rows = [row[0] for row in _cursor.fetchall()]
    print(f"Found {len(rows)} of training data")
    for row in rows:
        j = json.loads(row)
        data.append(j)

    data = add_indices(data)
    verify_indices(data)

    # Convert data into spaCy format
    training_data = []
    for entry in data:
        text = entry["filename"]
        entities = []
        for ann in entry["annotations"]:
            entities.append((ann['start'], ann['end'], ann["label"]))
            # ner.add_label(ann["label"])
        training_data.append((text, {"entities": entities}))

    # Disable other pipeline components for efficiency
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):

        # Training loop
        optimizer = nlp.begin_training()
        for epoch in range(100):  # 100 iterations
            random.shuffle(training_data)
            losses = {}
            for text, annotations in training_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            print(f"Epoch {epoch + 1}, Loss: {losses}")

    # Save the trained model
    nlp.to_disk(model_name)
    print(f"Training complete. Model saved to '{model_name}'.")
