import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()


train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
train_images = train_images/255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
test_images = test_images/255


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.1),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(rate=0.2),

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),

    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)

model.fit(train_images, train_labels, batch_size=16, epochs=3, use_multiprocessing=True, workers=8,
          validation_data=(test_images, test_labels))

print("Evaluate")
model.evaluate(test_images, test_labels)
