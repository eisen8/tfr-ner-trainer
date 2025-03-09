import os
import sqlite3
from constants import Constants as C


class Database:
    """
    A facade class for interacting with the Database.
    """

    @staticmethod
    def _connect():
        conn = sqlite3.connect(C.DB_FILE_PATH)
        conn.row_factory = sqlite3.Row  # Treat rows as dictionaries rather than tuples
        return conn

    @staticmethod
    def _close(conn):
        if conn:
            conn.commit()
            conn.close()


    @staticmethod
    def set_training_group(id: int, training_group: str) -> None:
        """ Assigns a training group to a specific record.
        :param id: The database record ID.
        :param training_group: The training group to assign ('T' for training or 'E' for evaluating).
        :return: None
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("UPDATE links SET training_group = ? WHERE id = ?", (training_group, id))
            conn.commit()
        finally:
            Database._close(conn)

    @staticmethod
    def get_rows_with_file_names() -> list[sqlite3.Row]:
        """ Retrieves all rows that have file names but no training group assigned.
        :return: A list of rows containing ids that need training group assignment.
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("SELECT id FROM links WHERE file_names IS NOT NULL and training_group IS NULL")
            return conn.cursor().fetchall()
        finally:
            Database._close(conn)


    @staticmethod
    def get_file_names(training_group: str) -> list[sqlite3.Row]:
        """Retrieves file names for a specific training group.
        :param training_group: The training group to retrieve files for ('T' or 'E').
        :return: A list of rows containing file_names for the specified training group.
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("SELECT file_names FROM links WHERE training_group = ? and file_names IS NOT NULL", (training_group,))
            return conn.cursor().fetchall()
        finally:
            Database._close(conn)

    @staticmethod
    def get_files_to_annotate(n: int) -> list[str]:
        """ Retrieves a limited number of files that need annotation.
        :param n: The maximum number of files to retrieve.
        :return: A list of filenames that need annotation.
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("SELECT filename FROM annotations WHERE annotation_json IS NULL LIMIT ?", (n,))
            rows = [row[0] for row in conn.cursor().fetchall()]
            return rows
        finally:
            Database._close(conn)

    @staticmethod
    def get_count_of_files_to_annotate() -> int:
        """ Gets the count of files that still need annotation.
        :return: The number of files that need annotation.
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("SELECT COUNT(filename) FROM annotations WHERE annotation_json IS NULL")
            count = conn.cursor().fetchone()[0]
            return count
        finally:
            Database._close(conn)

    @staticmethod
    def get_annotated_files() -> list[str]:
        """ Returns all files annotation_json
        :return: List of annotation_json
        """
        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("SELECT annotation_json FROM annotations WHERE annotation_json IS NOT NULL")
            return [row[0] for row in conn.cursor().fetchall()]
        finally:
            Database._close(conn)

    @staticmethod
    def update_indiced_annotation(filename: str, indiced_annotation: str):
        """ Returns all files annotation_json
        :return: None
        """
        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("UPDATE annotations SET annotation_json_indiced = ? WHERE filename = ?", (indiced_annotation, filename))
            conn.commit()
        finally:
            Database._close(conn)


    @staticmethod
    def bulk_insert_files_to_annotate(filenames: list[str]) -> int:
        """Bulk inserts filenames into the annotations table for annotation.
        :param filenames: List of filenames to be annotated.
        :return: The number of filenames successfully inserted.
        """

        conn = None
        try:
            conn = Database._connect()
            filenames = [(filename,) for filename in filenames]  # ExecuteMany expects a list of tuples
            conn.cursor().executemany("INSERT OR IGNORE INTO annotations (filename) VALUES (?)", filenames)
            conn.commit()
            return conn.cursor().rowcount
        finally:
            Database._close(conn)

    @staticmethod
    def add_annotation(filename: str, annotation: str) -> None:
        """Adds an annotation to a specific file.
        :param filename: The filename to annotate.
        :param annotation: The annotation JSON string.
        :return: None
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("UPDATE annotations SET annotation_json = ? WHERE filename = ?", (annotation, filename))
            conn.commit()
        finally:
            Database._close(conn)

    @staticmethod
    def clear_all_annotations() -> None:
        """ Clears all annotations from the database.
        :return: None
        """

        conn = None
        try:
            conn = Database._connect()
            conn.cursor().execute("UPDATE annotations SET annotation_json = NULL WHERE annotation_json IS NOT NULL")
            conn.commit()
        finally:
            Database._close(conn)