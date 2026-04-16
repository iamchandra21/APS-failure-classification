import pymongo
from sensor.constants.database import DATABASE_NAME
import os
import certifi
ca = certifi.where()

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

class MongoDBClient:
    client = None

    def __init__(self, database_name = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                # mongo_db_url = "USE your OWN URL"
                # MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
                MongoDBClient.client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"), tlsCAFile = ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise e