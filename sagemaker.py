import logging
import os
import traceback
import boto3
import datetime
import json
import botocore
from src.utils.mongo_connection import connection

logger = logging.getLogger(__name__)

class Sagemaker:
    def __init__(self):
        self.region_name = ""
        self.assume_role = ""
        self.content_type = ""
        self.llm_endpoint_name = ""
        self.reinit_client = ""
        self.max_new_tokens = 512
        self.top_p = 0.9
        self.temperature = 0.1
        self.smr_client = None
        self.sequence_max_length = 512

        if os.environ.get('release_tag_name'):
            self.__initialise_in_eks()

    def __initialise_in_eks(self):
        session = boto3.session.Session()
        sts_client = session.client('sts')
        try:
            assumed_sts = sts_client.assume_role(RoleArn=self.assume_role,
                                                     RoleSessionName=f"Sess_{datetime.datetime.now().date()}")

            credentials = assumed_sts['Credentials']
            self.smr_client = session.client(
                'sagemaker-runtime',
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
                region_name=self.region_name
            )
        except Exception as e:
            print(
                f"Exception : Not able to assume role- region_name: {self.region_name}, assume_role: {self.assume_role}")
            print(str(traceback.format_exc()))

    def llm_predict(self, sys_msg, context, question):
        try:
            if self.reinit_client:
                self.__initialise_in_eks()
            prompt_template = f'''
    
                    Context : {context}
    
                    Question : {question}
    
                    '''

            payload = {
                "inputs": [
                    [
                        {
                            "role": "system",
                            "content": sys_msg
                        },
                        {
                            "role": "user",
                            "content": prompt_template,
                        },
                    ]
                ],
                "parameters": {"max_new_tokens": self.max_new_tokens, "top_p": self.top_p,
                               "temperature": self.temperature},
            }
            try:
                response = self.smr_client.invoke_endpoint(EndpointName=self.llm_endpoint_name, Body=json.dumps(payload),
                                                           ContentType=self.content_type,
                                                           CustomAttributes="accept_eula=true")
            except botocore.exceptions.ClientError as client_error:
                print(''.join(traceback.TracebackException.from_exception(client_error).format()))
                logger.error(''.join(traceback.TracebackException.from_exception(client_error).format()))
                if "ExpiredTokenException" in str(client_error):
                    print("ExpiredTokenException, re __initialise_in_eks")
                    self.__initialise_in_eks()
                    response = self.smr_client.invoke_endpoint(EndpointName=self.llm_endpoint_name,
                                                               Body=json.dumps(payload),
                                                               ContentType=self.content_type,
                                                               CustomAttributes="accept_eula=true")
                else:
                    raise client_error

            return json.loads(response['Body'].read().decode("utf-8"))
        except Exception as e:
            print(
                f"Exception : Failed to invoke sagemaker llm endpoint: endpoint: {self.llm_endpoint_name}, sys_msg: {sys_msg}, "
                f"context: {context}, question: {question}")
            print(str(traceback.format_exc()))
            raise

    