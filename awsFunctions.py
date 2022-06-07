import boto3
from PIL import Image
import numpy as np
import cv2
import time


bucket_name = "etis3bucket"
names = ["SusieRowe",
         "JackCarpenter",
         "HelenBerry",
         "KimberlyCarson",
         "FaithWagner",
         "ClintonPeters",
         "WillieFisher",
         "ErikFoster",
         "TrevorSingleton",
         "MaggieMoody"]


def get_response(result):
    res = result.get('ResponseMetadata')
    status = res.get('HTTPStatusCode')
    print("~" * 80)
    if status == 200:
        print(f"Status: {status}".center(80))
    else:
        print("Failure:")
    print("~" * 80)

def deleteFiles():
    s3 = boto3.client('s3')
    directory_name = "databases"
    bucket_name = "etis3bucket"
    for name in names:
        result = s3.delete_object(Bucket=bucket_name, Key=f"{directory_name}/{name}")
        get_response(result)

def deleteDirectory():
    s3 = boto3.client('s3')
    directory_name = "/home/sean/Development/depthai-experiments/gen2-face-recognition/databases/ElsaFrozen.npz"
    bucket_name = "etis3bucket"
    result = s3.delete_object(Bucket=bucket_name, Key=f"{directory_name}")
    get_response(result)

def saveFile():
    s3 = boto3.resource('s3')
    result = s3.meta.client.put_object(Body='Text Contents', Bucket='etis3bucket', Key='filename.txt')
    get_response(result)

def listFiles():
    s3 = boto3.client('s3')
    bucket_name = "etis3bucket"
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")
    print("~" * 80)
    get_response(response)
    print("~" * 80)
    for file in files:
        for key, value in file.items():
            print(f"{key}: {value}")
        print("*"*80)
    return files

def getFiles():
    s3 = boto3.client('s3')
    bucket_name = "etis3bucket"
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")
    get_response(response)
    return files


def createFolders():
    # import boto3
    # s3 = boto3.client("s3")
    # BucketName = "mybucket"
    # myfilename = "myfile.dat"
    # KeyFileName = "/a/b/c/d/{fname}".format(fname=myfilename)
    # with open(myfilename) as f:
    #     object_data = f.read()
    #     client.put_object(Body=object_data, Bucket=BucketName, Key=KeyFileName)

    s3 = boto3.client('s3')
    bucket_name = "etis3bucket"
    # source_file_path = "test"
    directory_name = "databases"
    for name in names:
        key = f"{directory_name}/{name}"
        result = s3.put_object(Bucket=bucket_name, Body='Text Contents', Key=key)
        print(key)
        get_response(result)

def createFolder():
    s3 = boto3.client('s3')
    bucket_name = "etis3bucket"
    directory_name = "databases"
    result = s3.put_object(Bucket=bucket_name, Body='Text Contents', Key=f"{directory_name}")
    get_response(result)

def directory_exists(directory):
    s3 = boto3.client('s3')
    bucket_name = "etis3bucket"
    response = s3.list_objects_v2(Bucket=bucket_name)
    files = response.get("Contents")
    for file in files:
        filename = file.get("Key")
        if filename == directory:
            print(f"Found: {filename}")
            break

def save_new_model(file):
    directory = f"/databases/{file}"
    print(directory)
    # s3 = boto3.client('s3')
    # result = s3.put_object(Bucket=bucket_name, Body=b'bytes', Key=directory)
    # # get_response(result)
    # listFiles()

def saving_file(file):
    print(file)
    s3 = boto3.client("s3")
    s3filename = "databases/ElsaFrozen.npz"
    with open(file, "rb") as f:
        object_data = f.read()
        s3.put_object(Body=object_data, Bucket=bucket_name, Key=s3filename)
    listFiles()

def startOver():
    s3 = boto3.client('s3')
    files = getFiles()
    try:
        for i in files:
            key = i["Key"]
            bucket_name = "etis3bucket"
            result = s3.delete_object(Bucket=bucket_name, Key=key)
            get_response(result)
    except TypeError:
        print("No files to delete")

def view_image(key):
    bucket_name = "etis3bucket"
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    pix = np.array(im)
    cv2.imshow("Face Detection", cv2.resize(pix, (800,800)))
    cv2.waitKey(1)
    return np.array(pix)

def showCamera():
    files = getFiles()
    try:
        for i in files:
            key = i["Key"]
            if "jpg" in key:
                view_image(key)
    except TypeError:
        print("No files to delete")


# importing the module
import timeit


# sample function that returns
# square of the value passed
def print_square(x):
    return (x ** 2)


# importing the module
from datetime import datetime


# sample function that returns square
# of the value passed
def print_square(x):
    return (x ** 2)


# records the time at this instant of
# the program in HH:MM:SS format
start = datetime.now()

# calls the function
time.sleep(.5)

# records the time at this instant of the
# program in HH:MM:SS format
end = datetime.now()

# printing the execution time by subtracting
# the time before the function from
# the time after the function
timeTook = end - start
if timeTook.microseconds > 500000:
    print("Longer than one second")
else:
    print("Less than one second")
print(timeTook)