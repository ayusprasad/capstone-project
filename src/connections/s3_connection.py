# -*- coding: utf-8 -*-
"""S3 connection module for fetching data from AWS S3 buckets."""

from io import StringIO
import boto3
import pandas as pd
from src.logger import logging


class s3_operations:
    """Handle S3 operations for data ingestion."""

    def __init__(
        self,
        bucket_name,
        aws_access_key,
        aws_secret_key,
        region_name="us-east-1"
    ):
        """
        Initialize S3 operations with AWS credentials.

        Args:
            bucket_name: S3 bucket name
            aws_access_key: AWS access key ID
            aws_secret_key: AWS secret access key
            region_name: AWS region (default: us-east-1)
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("S3 data ingestion initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetch a CSV file from S3 and return as DataFrame.

        Args:
            file_key: S3 file path (e.g., 'data/data.csv')

        Returns:
            pd.DataFrame or None: DataFrame if successful, None otherwise
        """
        try:
            logging.info(
                "Fetching file '%s' from S3 bucket '%s'...",
                file_key,
                self.bucket_name
            )
            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(
                "Successfully fetched '%s' from S3 with %d records",
                file_key,
                len(df)
            )
            return df
        except Exception as e:
            logging.error(
                "Failed to fetch '%s' from S3: %s",
                file_key,
                e,
                exc_info=True
            )
            return None
