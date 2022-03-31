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

    def send_to_log(self, json, database_name):
        db = self.client.gridfs_images
        fs = gridfs.GridFS(db)
        put_image = fs.put(json["image"])
        post = {"image": put_image,
                "date": datetime.now(),
                "name": json["name"],
                "id_num": json["id_num"]}
        registration = self.client[database_name]
        collection = registration['faces']
        collection.insert_one(post)

    def recognize_images(self, sent_image):
        # This is the sloppiest code I've ever written.
        # There is a better way to do this, but I'm in a rush.
        time = datetime.now().strftime("%H:%M:%S")
        current_date = date.today().strftime("%Y-%m-%d")
        items = self.get_registrants()
        db = self.client.gridfs_images
        fs = gridfs.GridFS(db)
        for count, item in enumerate(items):
            results = [None]
            item["time"] = time
            item["date"] = current_date
            img = fs.get(item["image"]).read()
            img = Image.open(io.BytesIO(img))
            img.save("testImage.png")
            known_image = face_recognition.load_image_file("testImage.png")
            # Assume the whole image is the location of the face
            height, width, _ = known_image.shape
            # location is in css order - top, right, bottom, left
            face_location = (0, width, height, 0)
            try:
                unknown_image = face_recognition.load_image_file("temp.png")
                known_encoding = face_recognition.face_encodings(known_image, known_face_locations=[face_location])[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            except UnidentifiedImageError:
                print("Image sucks, or is broken. Most likely broken")
            except IndexError:
                print("Image is missing a face.")
            if results[0]:
                item["image"] = sent_image
                self.send_to_log(item, 'recognized_faces')
                self.send_to_log(item, 'detected_faces')
                print(f"Name: {item['name']}\nID: {item['id_num']}")
                data = {"Name": item["name"], "ID_NUM": item["id_num"]}
                return data
            else:
                print("UNKNOWN")
                item["name"] = "UNKNOWN"
                item["id_num"] = "UNKNOWN"
                item["image"] = sent_image
                self.send_to_log(item, 'detected_faces')
                return False

    def get_registrants(self):
        registration = self.client['registered_users']
        collection = registration['faces']
        items = collection.find()
        return items

    def get_logs(self, databaseName):
        rows = []
        registration = self.client[databaseName]
        collection = registration['faces']
        items = collection.find()
        for item in items:
            registrant = {'Name': item["name"], 'ID': item["id_num"], 'Date': item["date"].strftime("%Y-%m-%d"), 'Time': item["date"].strftime("%H:%M:%S")}
            rows.append(registrant)
        return rows

if __name__ == "__main__":
    mydb = DB()
    items = mydb.get_registrants()
