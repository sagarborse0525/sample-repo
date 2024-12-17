import boto3, cv2, time, numpy as np, matplotlib.pyplot as plt, random
import base64, json


def lambda_handler(event, context):
    # TODO implement
    sm_client = boto3.client(service_name="sagemaker")
    s3_client = boto3.client('s3')

    # Restore the endpoint name stored in the 2_DeployEndpoint.ipynb notebook
    # %store -r ENDPOINT_NAME
    ENDPOINT_NAME = "yolo8-aws-endpoint-1"
    print(f'Endpoint Name: {ENDPOINT_NAME}')

    endpoint_created = False
    while True:
        response = sm_client.list_endpoints()
        for ep in response['Endpoints']:
            print(f"Endpoint Status = {ep['EndpointStatus']}")
            if ep['EndpointName']==ENDPOINT_NAME and ep['EndpointStatus']=='InService':
                endpoint_created = True
                break
        if endpoint_created:
            break
        time.sleep(5)
    infer_start_time = time.time()
    # Read the image into a numpy  array
    orig_image = cv2.imread('images-5.jpg')

    # Calculate the parameters for image resizing
    image_height, image_width, _ = orig_image.shape
    model_height, model_width = 640, 640
    x_ratio = image_width/model_width
    y_ratio = image_height/model_height

    # Resize the image as numpy array
    resized_image = cv2.resize(orig_image, (model_height, model_width))
    # Conver the array into jpeg
    resized_jpeg = cv2.imencode('.jpg', resized_image)[1]
    # Serialize the jpg using base 64
    payload = base64.b64encode(resized_jpeg).decode('utf-8')

    runtime= boto3.client('runtime.sagemaker')
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                            ContentType='text/csv',
                                            Body=payload)
    response_body = response['Body'].read()
    result = json.loads(response_body.decode('ascii'))

    infer_end_time = time.time()


    print(f"Inference Time = {infer_end_time - infer_start_time:0.4f} seconds")

    if 'boxes' in result:
        for idx,(x1,y1,x2,y2,conf,lbl) in enumerate(result['boxes']):
            # Draw Bounding Boxes
            x1, x2 = int(x_ratio*x1), int(x_ratio*x2)
            y1, y2 = int(y_ratio*y1), int(y_ratio*y2)
            color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
            cv2.rectangle(orig_image, (x1,y1), (x2,y2), color, 4)
            cv2.putText(orig_image, f"Class: {int(lbl)}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(orig_image, f"Conf: {int(conf*100)}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            if 'masks' in result:
                # Draw Masks
                mask = cv2.resize(np.asarray(result['masks'][idx]), dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
                for c in range(3):
                    orig_image[:,:,c] = np.where(mask>0.5, orig_image[:,:,c]*(0.5)+0.5*color[c], orig_image[:,:,c])

    if 'probs' in result:
        # Find Class
        lbl = result['probs'].index(max(result['probs']))
        color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        cv2.putText(orig_image, f"Class: {int(lbl)}", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
    if 'keypoints' in result:
        # Define the colors for the keypoints and lines
        keypoint_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))
        line_color = (random.randint(10,255), random.randint(10,255), random.randint(10,255))

        # Define the keypoints and the lines to draw
        # keypoints = keypoints_array[:, :, :2]  # Ignore the visibility values
        lines = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Torso
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw the keypoints and the lines on the image
        for keypoints_instance in result['keypoints']:
            # Draw the keypoints
            for keypoint in keypoints_instance:
                if keypoint[2] == 0:  # If the keypoint is not visible, skip it
                    continue
                cv2.circle(orig_image, (int(x_ratio*keypoint[:2][0]),int(y_ratio*keypoint[:2][1])), radius=5, color=keypoint_color, thickness=-1)

            # Draw the lines
            for line in lines:
                start_keypoint = keypoints_instance[line[0]]
                end_keypoint = keypoints_instance[line[1]]
                if start_keypoint[2] == 0 or end_keypoint[2] == 0:  # If any of the keypoints is not visible, skip the line
                    continue
                cv2.line(orig_image, (int(x_ratio*start_keypoint[:2][0]),int(y_ratio*start_keypoint[:2][1])),(int(x_ratio*end_keypoint[:2][0]),int(y_ratio*end_keypoint[:2][1])), color=line_color, thickness=2)

    # Save the processed image to a temporary location
    processed_image_path = '/tmp/processed_image.jpg'
    cv2.imwrite(processed_image_path, orig_image)

    # Upload the image to S3
    bucket_name = 'your-s3-bucket-name'
    s3_key = 'image-segmentation/output_image/processed_image.jpg'
    s3_client.upload_file(processed_image_path, bucket_name, s3_key)
    print(f"Image uploaded to S3 bucket {bucket_name} with key {s3_key}")
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    plt.show()
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
