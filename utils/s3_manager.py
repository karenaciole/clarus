import boto3
import os
from botocore.exceptions import ClientError
from config.settings import Settings as ConfigSettings
from config.logging_config import app_logger

class S3Manager:
    def __init__(self):
        self.bucket_name = ConfigSettings.s3_bucket_name
        self.region = ConfigSettings.bedrock_region
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.logger = app_logger

    def upload_file(self, file_path, object_name=None):
        """
        Upload a file to an S3 bucket.
        :param file_path: File to upload
        :param object_name: S3 object name. If not specified, file_path basename is used
        :return: True if file was uploaded, else False
        """
        if object_name is None:
            object_name = os.path.basename(file_path)

        if ConfigSettings.s3_prefix:
            object_name = f"{ConfigSettings.s3_prefix.strip('/')}/{object_name}"

        try:
            self.logger.info(f"Uploading {file_path} to s3://{self.bucket_name}/{object_name}")
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
        except ClientError as e:
            self.logger.error(f"Error uploading to S3: {e}")
            return False
        return True

    def upload_bytes(self, file_bytes, object_name):
        """
        Upload bytes to an S3 bucket.
        :param file_bytes: Bytes to upload
        :param object_name: S3 object name
        :return: True if file was uploaded, else False
        """
        if ConfigSettings.s3_prefix:
            object_name = f"{ConfigSettings.s3_prefix.strip('/')}/{object_name}"

        try:
            self.logger.info(f"Uploading bytes to s3://{self.bucket_name}/{object_name}")
            self.s3_client.put_object(Body=file_bytes, Bucket=self.bucket_name, Key=object_name)
        except ClientError as e:
            self.logger.error(f"Error uploading bytes to S3: {e}")
            return False
        return True

    def list_documents(self):
        """List documents in the configured S3 prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, 
                Prefix=ConfigSettings.s3_prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except ClientError as e:
            self.logger.error(f"Error listing S3 objects: {e}")
            return []
