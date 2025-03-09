import random
import time

from common.database import Database as DB
from common.logger import Logger as L
from src.tfr_ner_trainer.common.time_helper import format_time

if __name__ == "__main__":
    # -- CONFIG --
    training_groups = ['T', 'E']
    weights = [0.5, 0.5]  # 50% 'T', 50% 'E'

    # -- SCRIPT --
    start_time = time.time()
    rows = DB.get_rows_with_file_names()
    L.info(f'Found {len(rows)} rows to process')
    assigned_t = 0
    assigned_e = 0
    for row in rows:
        random_pool = random.choices(training_groups, weights=weights)[0]
        DB.set_training_group(row['id'], random_pool)

        if random_pool == 'T':
            assigned_t += 1
        else:
            assigned_e += 1

    # Summary
    L.info(f"---- Script has finished. ----")
    L.info(f"Run time: {format_time(time.time()-start_time)}")
    L.info(f"Results: ")
    L.info(f"{len(rows)} Rows Processed.")
    L.info(f"{assigned_t} assigned T and {assigned_e} assigned E")
    L.info(f'{L.num_errors} errors occurred')
    L.print_error_messages()
