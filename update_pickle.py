import json
import requests
import os
import pickle5 as pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as applications
import random
import sqlite3
from PIL import Image
import cv2
import prof_timer

class CardUpdater:
    def __init__(self):
        '''
        Copied from open_sorts project by Kennet Belenky, https://github.com/kbelenky/open_sorts
        Class initializer copied from Recognizer class located card_recognizer.py
        '''
        print('Loading Updater.')
        # The embedding model turns an image of a card into an embedding vector.
        self.embedding_interpreter = tf.lite.Interpreter(
            model_path='embedding_model.tflite')
        self.embedding_interpreter.allocate_tensors()
        self.embedding_input_details = self.embedding_interpreter.get_input_details(
        )[0]
        self.embedding_output_details = self.embedding_interpreter.get_output_details(
        )[0]
        self.image_dimensions = (self.embedding_input_details['shape'][1],
                                 self.embedding_input_details['shape'][2])
        print(f'Model image dimensions: {self.image_dimensions}')

        # The embedding dictionary maps embedding vectors to card ids.
        with open('embedding_dictionary.pickle', 'rb') as handle:
            embedding_dictionary = pickle.load(handle)
        # Turn the embedding dictionary into a numpy matrix for efficiency.
        self.card_ids = []
        embedding_list = []
        for card_id, embedding in embedding_dictionary.items():
            self.card_ids.append(card_id)
            embedding_list.append(embedding)
        self.embedding_matrix = np.array(embedding_list)


    def vectorize_image(self, image):
        '''
        Function Copied from open_sorts project by Kennet Belenky, https://github.com/kbelenky/open_sorts
        Modified from function called recogize_by_embedding() in class Recognizer, located in card_recognizer.py
        Adjusted function to return the vector output from the ml model.
        '''
        # Scale the image values to what the network expects.
        small_image = tf.image.resize(image,
                                      self.image_dimensions,
                                      antialias=True)
        with prof_timer.PerfTimer('preprocess'):
            image = applications.mobilenet_v2.preprocess_input(image * 255.0)

        # Generate the embedding from the image.
        with prof_timer.PerfTimer('predict embedding'):
            image = np.expand_dims(image, axis=0).astype(np.single)
            self.embedding_interpreter.set_tensor(
                self.embedding_input_details['index'], image)
            self.embedding_interpreter.invoke()
            target_embedding = self.embedding_interpreter.get_tensor(
                self.embedding_output_details["index"])[0]
        image_vector = np.squeeze(target_embedding)
        return image_vector


    def automatic_brightness_and_contrast(self, image, clip_hist_percent=1):
        '''
        Function Copied from open_sorts project by Kennet Belenky, https://github.com/kbelenky/open_sorts
        Originally located in transform.py, and Lifted from: https://stackoverflow.com/questions/57030125 
        per comment there. Returns adjusted image array.
        '''

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        '''
        # Calculate new histogram with desired range and show histogram 
        new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        plt.plot(hist)
        plt.plot(new_hist)
        plt.xlim([0,256])
        plt.show()
        '''

        auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return (auto_result, alpha, beta)


    def open_sql_connection(self):
        '''
        Opens card database
        '''
        try:
            conn = None
            dbPath = './card_db.sqlite'
            conn = sqlite3.connect(dbPath)
        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
        return conn


    def getImage(self, url, id):
        '''
        Downloads and saves image to vec.png. Opens image with opencv. Resizes image to size expected by model.
        Image array values are normalized between -1 to 1. Returns the image array.
        '''
        r = requests.get(url)
        fileName = "vec.png"
        open(fileName, 'wb').write(r.content)
        img = cv2.imread(fileName)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        down_width = 192
        down_height = 256
        down_points = (down_width, down_height)
        img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
        img, _, _ =  self.automatic_brightness_and_contrast(
                    img)
        #img = tf.image.resize(img,
        #                              self.image_dimensions,
        #                              antialias=True)
        img = ((img / 255.0) * 2) - 1
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img


    def deleteImage(self):
        '''
        Deletes vec.png
        '''
        if os.path.exists("vec.png"):
            os.remove("vec.png")
        else:
            print("The file does not exist") 


    def update_databases(self):
        '''
        Iterates over each entry of the sql and checks if the card id is already stored in the .pickle file.
        If the card is not already in the .pickle, checks if Scryfall has a .png link for the card.
        If there is a link to the image the image is converted to a vector via the ml model and stored in the .pickle
        '''
        with open('embedding_dictionary.pickle', 'rb') as p:
            embedding_dictionary = pickle.load(p)
        # Connect to sql db and setup sql query
        connection = self.open_sql_connection()
        cursor = connection.cursor()
        query = '''SELECT * FROM main.mtg'''
        cursor.execute(query)

        for row in cursor:
            card_id = row[0] + "_0"
            if card_id not in self.card_ids and row[16] != '':
                url = row[16]
                try:
                    image = self.getImage(url, card_id) # Download the image
                    embedding = np.array(self.vectorize_image(image), dtype=np.float32) # Vectorize image
                    embedding_dictionary[card_id] = embedding # Add entry to the unpickled object
                except:
                    print(card_id + ": Failed to add image.")

        with open('embedding_dictionary.pickle', 'wb') as p:
            pickle.dump(embedding_dictionary, p)
        connection.close()
        self.deleteImage()


def main():
    '''
    Updates .pickle file containing vectors charactarizing every magic .png available from Scryfall.com
    '''
    updater = CardUpdater()

    updater.update_databases()


if __name__ == "__main__":
    main()