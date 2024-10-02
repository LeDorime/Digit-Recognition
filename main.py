import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# importing a dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing the data to a number between 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"The loss for this model is {loss}")
print(f"The accuracy for this model is {accuracy}")

for i in range(1, 10):
    img = cv2.imread(f"{i}.png")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("--------------")
    print(f"The model predicts that the number is: {np.argmax(prediction)}")
    print(f"The correct number is {i}")
    print("--------------\n")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()



