import base64
from pymongo import MongoClient
import face_recognition
import gridfs
import datetime
from datetime import datetime
from datetime import date
from PIL import Image, UnidentifiedImageError
import io

class DB():
    def __init__(self):
        self.client = MongoClient("mongodb://127.0.0.1:27017")
        self.registered_users = self.get_registrants()
        self.imgdb = self.client.gridfs_images
        # self.fs = gridfs.GridFS(db)

    def decode_base64(self, base64_string):
        ''' Decodes a base 64 image, and saves it as unknown_image.png '''
        b = io.BytesIO(base64.b64decode(base64_string))
        im = Image.open(b)
        image_bytes = io.BytesIO()
        im.save(image_bytes, format='PNG')
        im.save("./unknown_image.png")
        return image_bytes.getvalue()

    def send_to_log(self, json, database_name):
        b64 = base64.b64decode(json["image"])
        fs = gridfs.GridFS(self.imgdb)
        put_image = fs.put(b64)
        post = {"image": put_image,
                "date": datetime.now(),
                "name": json["name"],
                "id_num": json["id_num"]}
        registration = self.client[database_name]
        collection = registration['faces']
        collection.insert_one(post)

    def get_unknown_image(self, user_request):
        self.decode_image(user_request["image"], "unknown_user.png")
        unknown_image = face_recognition.load_image_file("unknown_user.png")
        return unknown_image

    def get_user_image(self, user):
        self.fs = gridfs.GridFS(self.imgdb)
        user_image = self.fs.get(user["image"]).read()
        user_image = Image.open(io.BytesIO(user_image))
        user_image.save("known_user.png")
        known_user = face_recognition.load_image_file("known_user.png")
        return known_user

    def compare_images(self, user_request, user):
        try:
            unknown_image = self.get_unknown_image(user_request)
            known_image = self.get_user_image(user)
            # Assume the whole image is the location of the face
            height, width, _ = known_image.shape
            # location is in css order - top, right, bottom, left
            face_location = (0, width, height, 0)
            known_user = face_recognition.face_encodings(known_image, known_face_locations=[face_location])[0]
            unknown_user = face_recognition.face_encodings(unknown_image)[0]
            results = face_recognition.compare_faces([known_user], unknown_user)
            print(results)
            return results[0]
        except UnidentifiedImageError:
            print("Image sucks, or is broken. Most likely broken")
        except IndexError:
            print("Image is missing a face.")
        return None

    def process_request(self, user_request):
        for user in self.registered_users:
            known_user = self.get_user_image(user)
            unknown_user = self.get_unknown_image(user_request)
            match = self.compare_images(known_user, unknown_user)
            if match:
                user_request["name"] = user["name"]
                user_request["id_num"] = user["id_num"]
                self.send_to_log(user_request, 'recognized_faces')
                break
        # We send all faces to detected faces, whether it is a match or not.
        # Then we delete the image, so that it can't be reused again.
        self.send_to_log(user_request, 'detected_faces')
        return user_request


    def recognize_images(self, user_request):
        sent_image = self.decode_base64(user_request["image"])
        for user in self.registered_users:
            match = self.compare_images(user_request, user)
            if match:
                user_request["image"] = sent_image
                user_request["name"] = user["name"]
                user_request["id_num"] = user["id_num"]
                self.send_to_log(user_request, 'recognized_faces')
                data = {"name": user_request["name"], "id_num": user_request["id_num"]}
                return data
        print("UNKNOWN")
        user_request["name"] = "UNKNOWN"
        user_request["id_num"] = "UNKNOWN"
        user_request["image"] = sent_image
        self.send_to_log(user_request, 'detected_faces')
        return False

    def decode_image(self, base64_string, name):
        # Convert from base64 string to base64 bytes
        b64 = base64.b64decode(base64_string)
        # Convert to b64 binary object.
        base64_bytes = base64.b64encode(b64)
        with open(name, "wb") as fh:
            fh.write(base64.decodebytes(base64_bytes))
            return fh

    def get_registrants(self):
        registration = self.client['registered_users']
        collection = registration['faces']
        items = collection.find()
        return items

    def get_logs(self, databaseName):
        rows = []
        database = self.client[databaseName]
        collection = database['faces']
        items = collection.find()
        for item in items:
            user = {'Name': item["name"], 'ID': item["id_num"], 'Date': item["date"].strftime("%Y-%m-%d"), 'Time': item["date"].strftime("%H:%M:%S")}
            rows.append(user)
        return rows[::-1]

if __name__ == "__main__":
    mydb = DB()
    items = mydb.get_logs("detected_faces")
