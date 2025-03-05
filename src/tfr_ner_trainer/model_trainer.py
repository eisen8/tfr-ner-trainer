from datetime import datetime
import re
import sqlite3
import time
from pathlib import Path

import spacy
import json
import random
import cupy as cp
from tokenizer import custom_tokenizer
from spacy.training.example import Example
from common.constants import Constants as C

def tprint(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"

if __name__ == "__main__":
    # Config
    start_time = time.time()
    model_name = "tfr-model"
    if spacy.prefer_gpu():
        spacy.require_gpu()  # Use GPU if available
        tprint("Training on GPU")
    else:
        tprint("Training on CPU")

    nlp = spacy.blank("en")  # Use a blank English model
    nlp.tokenizer = custom_tokenizer(nlp)

    # Add NER component if not already present
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Load training data
    data = []
    _conn = sqlite3.connect(str(C.DB_FILE_PATH))
    _conn.row_factory = sqlite3.Row  # Treat rows as dictionaries rather than tuples
    _cursor = _conn.cursor()
    _cursor.execute("SELECT annotation_json_indiced FROM annotations WHERE annotation_json_indiced IS NOT NULL")
    rows = [row[0] for row in _cursor.fetchall()]
    tprint(f"Found {len(rows)} of training data")
    for row in rows:
        j = json.loads(row)
        data.append(j)

    # Convert data into spaCy format
    training_data = []
    for entry in data:
        text = entry["filename"]
        entities = []
        for ann in entry["annotations"]:
            entities.append((ann['start'], ann['end'], ann["label"]))
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
                try:
                    example = Example.from_dict(nlp.make_doc(text), annotations)
                    nlp.update([example], drop=0.5, losses=losses)
                except Exception as e:
                    tprint(f"Error with text {text} - {e}")
            tprint(f"Epoch {epoch + 1}, Loss: {losses}")

    # Save the trained model
    nlp.to_disk(model_name)
    tprint(f"Training complete. Model saved to '{model_name}'.")
    tprint(f"Run time: {format_time(time.time() - start_time)}")
