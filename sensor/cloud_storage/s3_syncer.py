import subprocess
import time

from sensor.exception import SensorException
from sensor.logger import logging


class S3Sync:

    def sync_folder_to_s3(self, folder: str, aws_bucket_url: str) -> None:
        try:
            command = ["aws", "s3", "sync", folder, aws_bucket_url, "--no-progress"]
            logging.info(f"Starting S3 sync: {folder} → {aws_bucket_url}")
            start = time.time()
            subprocess.run(command, check=True, capture_output=True, text=True)
            duration = round(time.time() - start, 2)
            logging.info(f"S3 sync completed successfully in {duration}s")
        except subprocess.CalledProcessError as e:
            logging.error(f"S3 sync failed: {e.stderr}")
            raise SensorException(str(e))

    def sync_folder_from_s3(self, folder: str, aws_bucket_url: str) -> None:
        try:
            command = ["aws", "s3", "sync", aws_bucket_url, folder, "--no-progress"]
            logging.info(f"Starting S3 download: {aws_bucket_url} → {folder}")
            start = time.time()
            subprocess.run(command, check=True, capture_output=True, text=True)
            duration = round(time.time() - start, 2)
            logging.info(f"S3 download completed successfully in {duration}s")
        except subprocess.CalledProcessError as e:
            logging.error(f"S3 download failed: {e.stderr}")
            raise SensorException(str(e))
