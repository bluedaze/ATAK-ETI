import base64
from pymongo import MongoClient
import face_recognition
import os
from pprint import pprint
import gridfs
import copy
import uuid
import datetime
from datetime import datetime
from datetime import date
from pprint import pprint
from PIL import Image, UnidentifiedImageError
import io

class DB():
    def __init__(self):
        self.client = MongoClient("mongodb://127.0.0.1:27017")
        self.registered_users = self.get_registrants()
        self.imgdb = self.client.gridfs_images
        # self.fs = gridfs.GridFS(db)

    def send_to_log(self, user_request, database_name):
        registration = self.client[database_name]
        collection = registration['faces']
        collection.insert_one(user_request)

    def compare_images(self, user_request, user):
        try:
            unknown_image = face_recognition.load_image_file(f".{user_request['image']}")
            known_image = face_recognition.load_image_file(f".{user['image']}")
            # Assume the whole image is the location of the face
            height, width, _ = known_image.shape
            # location is in css order - top, right, bottom, left
            face_location = (0, width, height, 0)
            known_user = face_recognition.face_encodings(known_image, known_face_locations=[face_location])[0]
            unknown_user = face_recognition.face_encodings(unknown_image)[0]
            results = face_recognition.compare_faces([known_user], unknown_user)
            return results[0]
        except UnidentifiedImageError:
            print("Image sucks, or is broken. Most likely broken")
        except IndexError:
            print("Image is missing a face.")
        return None

    def save_image(self, user_request):
        directory = "static/images"
        cwd = os.getcwd()
        base64_string = user_request["image"]
        unique_id = uuid.uuid1()
        file = f"{unique_id}.png"
        location = f"/{directory}/{file}"
        working_directory = f"{cwd}/{directory}/"
        completeName = os.path.join(working_directory, file)
        b64 = base64.b64decode(base64_string)
        base64_bytes = base64.b64encode(b64)
        with open(completeName, "wb") as file:
            file.write(base64.decodebytes(base64_bytes))
        user_request["image"] = f"{location}"
        return user_request

    def recognize_images(self, user_request):
        self.save_image(user_request)
        for user in self.registered_users:
            match = self.compare_images(user_request, user)
            if match:
                user_request["Original"] = user["image"]
                user_request["Name"] = user["Name"]
                user_request["ID Number"] = user["ID Number"]
                self.send_to_log(user_request, 'recognized_faces')
                self.send_to_log(user_request, 'detected_faces')
                data = {"name": user_request["Name"], "ID Number": user_request["ID Number"]}
                return data
        print("UNKNOWN")
        self.send_to_log(user_request, 'detected_faces')
        return False

    def get_registrants(self):
        registration = self.client['registered_users']
        collection = registration['faces']
        items = collection.find()
        return items

    def get_logs(self, databaseName):
        rows = []
        client = self.client[databaseName]
        database = client['faces']
        collection = database.find()
        for count, item in enumerate(collection):
            columns = {key: value for (key, value) in item.items() if key != "_id"}
            rows.append(columns)
        header = [item for item in rows[0]]

        return header, rows[::-1]

if __name__ == "__main__":
    mydb = DB()
    items = mydb.get_logs("recognized_faces")
