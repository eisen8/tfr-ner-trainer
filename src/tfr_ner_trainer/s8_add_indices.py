import json
import re
import time

from common.logger import Logger as L
from common.database import Database as DB
from src.tfr_ner_trainer.common.time_helper import format_time


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
                L.info(entry)
                # raise Exception(f"Could not find {text} in {filename}")
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
                fail.append(f"Invalid annotation: '{text}' does not match extracted text '{extracted_text}' (from indices {start}-{end}) for {filename}")

            # Check for overlap
            overlap = any(index in used_indices for index in range(start, end))
            if overlap:
                fail.append(f"Overlap detected: Annotation '{text}' overlaps with a previous annotation (from indices {start}-{end}) for {filename}")

            used_indices.update(range(start, end))

    if len(fail) > 0:
        raise Exception(f"Error in annotations - {fail}")


if __name__ == "__main__":
    # -- SCRIPT --
    start_time = time.time()
    data = []
    rows = DB.get_annotated_files()
    L.info(f"Found {len(rows)} of training data")
    for row in rows:
        j = json.loads(row)
        data.append(j)

    data = add_indices(data)
    verify_indices(data)

    for entry in data:
        filename = entry["filename"]
        json_str = json.dumps(entry, indent=4)
        DB.update_indiced_annotation(filename, json_str)

    # Summary
    L.info(f"---- Script has finished. ----")
    L.info(f"Run time: {format_time(time.time() - start_time)}")
    L.info(f"Results: ")
    L.info(f"{rows} rows processed.")
    L.info(f'{L.num_errors} errors occurred')
    L.print_error_messages()
