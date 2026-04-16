import os

import certifi
import pymongo
from dotenv import find_dotenv, load_dotenv

from sensor.constants.database import DATABASE_NAME

ca = certifi.where()

_ = load_dotenv(find_dotenv())


class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                MongoDBClient.client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"), tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise e
