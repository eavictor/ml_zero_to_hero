import os
import zipfile
import requests
import tensorflow as tf
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"
TEST_DATA_URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip"


def download():
    urls = [TRAIN_DATA_URL, TEST_DATA_URL]
    files = []
    for url in urls:
        filename = url.split("/")[-1]
        files.append(filename)
        if not os.path.exists(os.path.join(BASE_DIR, filename)):
            with requests.get(url, stream=True) as req:
                req.raise_for_status()
                # write chunks
                with open(os.path.join(BASE_DIR, filename), "wb") as file:
                    for chunk in tqdm(req.iter_content(chunk_size=8192)):
                        file.write(chunk)
                    file.flush()

    for filename in files:
        with zipfile.ZipFile(os.path.join(BASE_DIR, filename), mode="r", allowZip64=True) as zipped:
            zipped.extractall(BASE_DIR)


def main():
    training_dir = os.path.join(BASE_DIR, "rps")
    validation_dir = os.path.join(BASE_DIR, "rps-test-set")

    # ImageDataGenerator requires package pillow and scipy
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(training_dir,
                                                  target_size=(150, 150),
                                                  class_mode="categorical"
                                                  )
    validation_generator = datagen.flow_from_directory(validation_dir,
                                                       target_size=(150, 150),
                                                       class_mode="categorical"
                                                       )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu",
                               input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="rmsprop", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])

    model.fit_generator(train_generator, epochs=25,
                        validation_data=validation_generator,
                        verbose=1, use_multiprocessing=True, workers=16)
    model.save("04_build_an_image_classifier.h5")


if __name__ == "__main__":
    download()
    main()
