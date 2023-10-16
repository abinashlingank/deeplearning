‘1’
import numpy as np
from matplotlib import pyplot as plt
def sigmoid(z):
 return 1 / (1 + np.exp(-z))
def initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures):
 W1 = np.random.randn(neuronsInHiddenLayers, inputFeatures)
 W2 = np.random.randn(outputFeatures, neuronsInHiddenLayers)
 b1 = np.zeros((neuronsInHiddenLayers, 1))
 b2 = np.zeros((outputFeatures, 1))
 parameters = {"W1" : W1, "b1": b1,
 "W2" : W2, "b2": b2}
 return parameters
def forwardPropagation(X, Y, parameters):
 m = X.shape[1]
 W1 = parameters["W1"]
 W2 = parameters["W2"]
 b1 = parameters["b1"]
 b2 = parameters["b2"]
 Z1 = np.dot(W1, X) + b1
 A1 = sigmoid(Z1)
 Z2 = np.dot(W2, A1) + b2
 A2 = sigmoid(Z2)
 cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
 logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
 cost = -np.sum(logprobs) / m
 return cost, cache, A2
def backwardPropagation(X, Y, cache):
 m = X.shape[1]
 (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
 dZ2 = A2 - Y
 dW2 = np.dot(dZ2, A1.T) / m
 db2 = np.sum(dZ2, axis = 1, keepdims = True)
 dA1 = np.dot(W2.T, dZ2)
 dZ1 = np.multiply(dA1, A1 * (1- A1))
 dW1 = np.dot(dZ1, X.T) / m
 db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
 gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
 "dZ1": dZ1, "dW1": dW1, "db1": db1}
 return gradients
def updateParameters(parameters, gradients, learningRate):
 parameters["W1"] = parameters["W1"] - learningRate * gradients["dW1"]
 parameters["W2"] = parameters["W2"] - learningRate * gradients["dW2"]
 parameters["b1"] = parameters["b1"] - learningRate * gradients["db1"]
 parameters["b2"] = parameters["b2"] - learningRate * gradients["db2"]
 return parameters
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # XOR input
Y = np.array([[0, 1, 1, 0]]) # XOR output
neuronsInHiddenLayers = 2
inputFeatures = X.shape[0] # number of input features (2) 
outputFeatures = Y.shape[0] # number of output features (1)
parameters = initializeParameters (inputFeatures, neuronsInHiddenLayers, outputFeatures)
epoch = 100000
learningRate = 0.01
losses = np.zeros((epoch, 1))
for i in range(epoch):
 losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
 gradients = backwardPropagation(X, Y, cache)
 parameters = updateParameters(parameters, gradients, learningRate)
plt.figure()
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()
X = np.array([[1, 1, 0, 0], [0, 1, 0, 1]]) # XOR input
cost, _, A2 = forwardPropagation(X, Y, parameters)
prediction = (A2 > 0.5) * 1.0
print(prediction)
‘2’
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
 loss='categorical_crossentropy',
 metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}') 
‘3’
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,
Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # Binary classification (face or not face)
model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])
datagen = ImageDataGenerator(
 rescale=1.0/255.0,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True
)
train_data = datagen.flow_from_directory('data/train', target_size=(64, 64),
batch_size=32, class_mode='binary')
test_data = datagen.flow_from_directory('data/test', target_size=(64, 64),
batch_size=32, class_mode='binary')
model.fit(train_data, epochs=10, validation_data=test_data)
model.save('face_recognition_model.h5')
loaded_model = tf.keras.models.load_model('face_recognition_model.h5')
def recognize_face(image_path):
 image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
 image = tf.keras.preprocessing.image.img_to_array(image)
 image = np.expand_dims(image, axis=0)
 prediction = loaded_model.predict(image)
 if prediction[0][0] >= 0.5:
 return "Face"
 else:
 return "Not Face"
result = recognize_face('new_image.jpg')
print(f'Recognition Result: {result}') 
‘4’
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
text = """Your input text goes here..."""
chars = sorted(list(set(text)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}
sequence_length = 100 # Length of input sequences
step = 1 # Step size for moving the sliding window
X_data = [] # Input sequences
y_data = [] # Corresponding target characters
for i in range(0, len(text) - sequence_length, step):
 sequence_in = text[i:i + sequence_length]
 sequence_out = text[i + sequence_length]
 X_data.append([char_to_int[char] for char in sequence_in])
 y_data.append(char_to_int[sequence_out])
X = np.reshape(X_data, (len(X_data), sequence_length, 1))
X = X / float(len(chars))
y = to_categorical(y_data)
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=20, batch_size=128)
seed_text = "Your seed text goes here..."
generated_text = seed_text
for _ in range(200):
 x = np.array([char_to_int[char] for char in seed_text[-sequence_length:]]).reshape(1,
sequence_length, 1) / float(len(chars))
 prediction = model.predict(x, verbose=0)[0]
 next_char = int_to_char[np.argmax(prediction)]
 generated_text += next_char
 seed_text += next_char
print(generated_text) 
‘5’
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 10000 # Vocabulary size (use the most frequent words)
max_length = 250 # Sequence length (truncate or pad reviews to this length)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128,
input_length=max_length))
model.add(SimpleRNN(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])
batch_size = 64
epochs = 5
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.4f}, accuracy: {accuracy:.4f}")
def predict_sentiment(text):
 text_sequence = imdb.get_word_index()[text.lower()]
 text_sequence = pad_sequences([text_sequence], maxlen=max_length)
 prediction = model.predict(text_sequence)
 if prediction >= 0.5:
 return "Positive"
 else:
 return "Negative"
sample_review = "This movie was great! I loved it."
sentiment = predict_sentiment(sample_review)
print(f"Sentiment: {sentiment}") 
‘6’
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, RepeatVector,
TimeDistributed
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = ["The quick brown fox jumps over the lazy dog",
 "She sells seashells by the seashore"]
pos_tags = ["DT JJ NN VBZ IN DT JJ NN", "PRP VBZ NNS IN DT NN"]
tokenizer_sent = Tokenizer()
tokenizer_sent.fit_on_texts(sentences)
sent_seq = tokenizer_sent.texts_to_sequences(sentences)
tokenizer_tags = Tokenizer()
tokenizer_tags.fit_on_texts(pos_tags)
tags_seq = tokenizer_tags.texts_to_sequences(pos_tags)
# Vocabulary size
vocab_size = len(tokenizer_sent.word_index) + 1
num_tags = len(tokenizer_tags.word_index) + 1
max_sent_length = max(len(seq) for seq in sent_seq)
max_tag_length = max(len(seq) for seq in tags_seq)
sent_seq = pad_sequences(sent_seq, maxlen=max_sent_length, padding='post')
tags_seq = pad_sequences(tags_seq, maxlen=max_tag_length, padding='post')
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sent_length))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
tags_seq_onehot = tf.keras.utils.to_categorical(tags_seq, num_tags)
model.fit(sent_seq, tags_seq_onehot, epochs=10, batch_size=32)
new_sentence = "A quick brown dog jumps over the lazy fox"
new_sentence_seq = tokenizer_sent.texts_to_sequences([new_sentence])
new_sentence_seq = pad_sequences(new_sentence_seq, maxlen=max_sent_length,
padding='post')
predicted_tags = model.predict(new_sentence_seq)
predicted_tags = np.argmax(predicted_tags, axis=-1)
predicted_tags = [tokenizer_tags.index_word[tag] for tag in predicted_tags[0] if tag
!= 0]
print(predicted_tags) 
‘7’
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
encoder_inputs = Input(shape=(encoder_input_seq_length,))
decoder_inputs = Input(shape=(decoder_input_seq_length,))
encoder_embedding = Embedding(input_dim=num_encoder_tokens,
output_dim=embedding_dim)(encoder_inputs)
decoder_embedding = Embedding(input_dim=num_decoder_tokens,
output_dim=embedding_dim)(decoder_inputs)
encoder_lstm = LSTM(units=latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h,
state_c])
output_layer = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = output_layer(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
batch_size=batch_size, epochs=epochs, validation_split=0.2)
‘8’
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
generator = Sequential()
generator.add(Dense(128, input_shape=(100,), activation='relu'))
generator.add(Dense(784, activation='sigmoid'))
generator.add(Reshape((28, 28, 1)))
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(28, 28, 1)))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5),
metrics=['accuracy'])
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = tf.keras.models.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5)) 
