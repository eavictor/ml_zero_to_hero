import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),

    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer='sgd', loss=tf.keras.losses.sparse_categorical_crossentropy)

model.fit(train_images, train_labels, batch_size=16, epochs=5, use_multiprocessing=True, workers=8)

print("Evaluate")
model.evaluate(test_images, test_labels)