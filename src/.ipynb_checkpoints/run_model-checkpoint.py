import tensorflow as tf

weights_dict = {}
weight_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: weights_dict.update({epoch: model.get_weights()}))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, validation_data=(
    x_test, y_test), callbacks=[weight_callback])

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")

for epoch, weights in weights_dict.items():
    print("Weights for 1st Layer of epoch #", epoch+1)
    print(weights[2])
