from pathlib import Path


class Constants:
    BASE_DIR_PATH = Path(__file__).parent.parent.resolve()  # Project directory path
    TRAINING_DATA_FOLDER_PATH = (BASE_DIR_PATH / "training_data").resolve()  # Training Data path:  <project dir>/training_data
    MODEL_FOLDER_PATH = (BASE_DIR_PATH / "model").resolve()  # Model path:  <project dir>/model
    DB_FILE_PATH = (BASE_DIR_PATH / "../../../tfr-data-scraper/src/tfr_data_scraper/data/database.db").resolve()  # <project dir>/data/database.db
