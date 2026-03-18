import sqlite3
import logging
import os

# Set up logging for the database operations
logger = logging.getLogger("Database")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

class DatabaseManager:
    def __init__(self, path):
        self.data_base_path = path
        # In SQLite, checking the same thread often causes issues with async/UDP handling unless set to False.
        # Since this UDP server processes one request at a time, this is safe.
        try:
            # We initialize connection to None to establish it per-request or globally safely
            if not os.path.exists(self.data_base_path):
                logger.warning(f"Database file {self.data_base_path} not found. A new one will be created upon connection.")
            
            self.connection = sqlite3.connect(self.data_base_path, check_same_thread=False)
            self.cursor = self.connection.cursor()
            logger.info(f"Connected to database at {self.data_base_path}")
            
            # Ensure tables exist
            self._create_tables_if_not_exists()
        except sqlite3.Error as error:
            logger.error(f"Error while connecting to sqlite: {error}")

    def _create_tables_if_not_exists(self):
        try:
            self.cursor.execute("CREATE TABLE IF NOT EXISTS Cars (num TEXT NOT NULL)")
            self.cursor.execute("CREATE TABLE IF NOT EXISTS Cops (username TEXT NOT NULL, password TEXT NOT NULL)")
            self.connection.commit()
            logger.info("Checked/Created default database tables.")
        except sqlite3.Error as error:
            logger.error(f"Error while ensuring tables exist: {error}")

    def do_query(self, query, params=()):
        """Executes a custom query safely using parameters."""
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
            return True
        except sqlite3.Error as error:
            logger.error(f"Error while executing query '{query}': {error}")
            return False

    def insert_into_cars_table(self, num_to_insert):
        try:
            # Parameterized query to prevent SQL injection
            self.cursor.execute("INSERT INTO Cars (num) VALUES (?)", (num_to_insert,))
            self.connection.commit()
            logger.info(f"Inserted license plate '{num_to_insert}' into Cars table.")
            return True
        except sqlite3.Error as error:
            logger.error(f"Error while inserting '{num_to_insert}' to sqlite table: {error}")
            return False

    def check_num_in_cars_db(self, request_dict):
        try:
            plate_number = request_dict.get("Data", "")
            # Parameterized query
            self.cursor.execute("SELECT num FROM Cars WHERE num = ?", (plate_number,))
            self.connection.commit()
            num = self.cursor.fetchone()
            
            if num is None:
                return "No detection in the database"
            else:
                return f"STOLEN: {num[0]}"
        except sqlite3.Error as error:
            logger.error(f"Error while checking data from sqlite table: {error}")
            return "DATABASE ERROR"
    
    def delete_num_from_cars(self, num):
        try:
            self.cursor.execute("DELETE FROM Cars WHERE num = ?", (num,))
            self.connection.commit()
            logger.info(f"Deleted license plate '{num}' from Cars table.")
            return True
        except sqlite3.Error as error:
            logger.error(f"Error while deleting '{num}' from sqlite table: {error}")
            return False

    def check_all_from_cops_table(self, request_dict):
        try:
            username = request_dict.get("UserName", "")
            password = request_dict.get("PassWord", "")
            
            self.cursor.execute("SELECT * FROM Cops WHERE username = ? AND password = ?", (username, password))
            self.connection.commit()
            result = self.cursor.fetchall()

            if result:
                logger.info(f"Successful login for user '{username}'.")
                return "OK"
            else:
                logger.warning(f"Failed login attempt for user '{username}'.")
                return "Login Bad"
        except sqlite3.Error as error:
            logger.error(f"Error while reading from sqlite Cops table: {error}")
            return "Login Bad"
