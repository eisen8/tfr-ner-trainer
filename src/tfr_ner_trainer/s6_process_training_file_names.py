import os
import random
import time

from common.database import Database as DB
from common.logger import Logger as L
from src.tfr_ner_trainer.common.time_helper import format_time
from src.tfr_ner_trainer.preprocessor import preprocessor


def _contains_non_ascii_characters(filename: str) -> bool:
    non_ascii = [char for char in filename if ord(char) > 127]
    return len(non_ascii) > 0


if __name__ == "__main__":
    # -- CONFIG --
    remove_non_ascii_files = True
    training_group = "T"
    shuffle = True
    max_files_per_torrent = 10
    min_file_name_size = 12

    # -- SCRIPT --
    L.info(f"Training group: {training_group}")
    rows = DB.get_file_names(training_group)
    if shuffle:
        random.shuffle(rows)

    combined = []
    start_time = time.time()
    for index, row in enumerate(rows):
        # Extract only filenames from paths
        filenames = [os.path.basename(f) for f in row["file_names"].split("\n")]

        # Remove extensions
        filenames = [os.path.splitext(f)[0].strip() for f in filenames]

        # Ignore filenames less than min_file_name_size chars, to avoid files like Sample.mkv that don't have enough data to be annotated.
        filenames = [name for name in filenames if len(name) >= min_file_name_size]

        # Remove files with non-ascii character
        if remove_non_ascii_files:
            filenames = [name for name in filenames if _contains_non_ascii_characters(name) is False]

        # only take up to max_files_per_torrent filenames per group
        if len(filenames) > max_files_per_torrent:
            filenames = random.sample(filenames, 10)

        processed_filenames = []
        for filename in filenames:
            processed = preprocessor(filename)
            # L.info(f"{filename} ->")
            # L.info(f"{processed}")
            processed_filenames.append(processed)

        combined.extend(processed_filenames)

    L.info(f"Writing to db")
    annotation_rows_count = DB.bulk_insert_files_to_annotate(combined)

    # Summary
    L.info(f"---- Script has finished. ----")
    L.info(f"Run time: {format_time(time.time() - start_time)}")
    L.info(f"Results: ")
    L.info(f"{len(rows)} Rows Processed.")
    L.info(f"{len(combined)} File Names added")
    L.info(f"{annotation_rows_count} Actual rows added to annotations database")
    L.info(f'{L.num_errors} errors occurred')
    L.print_error_messages()
