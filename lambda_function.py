import boto3, cv2, time, numpy as np, matplotlib.pyplot as plt, random
import base64, json


def lambda_handler(event, context):
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
