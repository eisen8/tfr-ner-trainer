import json
import re
import sqlite3

from src.trf_ner_trainer.model_trainer import tprint
from common.constants import Constants as C


def add_indices(data):
    fail = []
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
                tprint(entry)
                #raise Exception(f"Could not find {text} in {filename}")
                tprint(f"Could not find '{text}' in '{filename}'")
                fail.append(f"Could not find '{text}' in '{filename}'")

            end = start + len(text)
            annotation["start"] = start
            annotation["end"] = end

            # Mark this range as used
            used_indices.update(range(start, end))

    if len(fail) > 0:
        raise Exception(f"Error in annotations - {fail}")
    return data


def verify_indices(data):
    fail = []
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
                tprint(f"Invalid annotation: '{text}' does not match extracted text '{extracted_text}' (from indices {start}-{end}) for {filename}")
                fail.append(f"Invalid annotation: '{text}' does not match extracted text '{extracted_text}' (from indices {start}-{end}) for {filename}")

            # Check for overlap
            overlap = any(index in used_indices for index in range(start, end))
            if overlap:
                tprint(f"Overlap detected: Annotation '{text}' overlaps with a previous annotation (from indices {start}-{end}) for {filename}")
                fail.append(f"Overlap detected: Annotation '{text}' overlaps with a previous annotation (from indices {start}-{end}) for {filename}")

            used_indices.update(range(start, end))

    if len(fail) > 0:
        raise Exception(f"Error in annotations - {fail}")

if __name__ == "__main__":

    data = []
    _conn = sqlite3.connect(str(C.DB_FILE_PATH))
    _conn.row_factory = sqlite3.Row  # Treat rows as dictionaries rather than tuples
    _cursor = _conn.cursor()
    _cursor.execute("SELECT annotation_json FROM annotations WHERE annotation_json IS NOT NULL")
    rows = [row[0] for row in _cursor.fetchall()]
    tprint(f"Found {len(rows)} of training data")
    for row in rows:
        j = json.loads(row)
        data.append(j)

    data = add_indices(data)
    verify_indices(data)

    for entry in data:
        filename = entry["filename"]
        json_str = json.dumps(entry, indent=4)
        _cursor.execute("UPDATE annotations SET annotation_json_indiced = ? WHERE filename = ?", (json_str, filename))
        _conn.commit()