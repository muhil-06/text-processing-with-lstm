import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


sentences = [
    "I love this movie",
    "This is a terrible film, I hate it.",
    "The acting was great and the story was engaging.",
    "What a bore, absolutely awful.",
    "Highly recommend, a truly amazing experience.",
    "Never again, a complete waste of time."
]


labels = np.array([1, 0, 1, 0, 1, 0])


vocab_size = 1000  
embedding_dim = 16 
max_length = 10    
trunc_type = 'post'
padding_type = 'post' 

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(f"Found {len(word_index)} unique tokens.")

sequences = tokenizer.texts_to_sequences(sentences)

padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("Original sentences:", sentences)
print("Padded sequences shape:", padded_sequences.shape)
print("First padded sequence:", padded_sequences[0])


model = Sequential([
    Embedding(vocab_size, embedding_dim),
    LSTM(32), 
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


num_epochs = 10

history = model.fit(padded_sequences, labels, epochs=num_epochs, verbose=2)

print("Training finished.")

test_sentence = ["I love this movie"]

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)

prediction = model.predict(test_padded)

if prediction[0][0] > 0.5:
    sentiment = "Positive"
else:
    sentiment = "Negative"

print(f"Test Sentence: '{test_sentence[0]}'")
print(f"Prediction score: {prediction[0][0]:.4f} (closer to 1 is positive, closer to 0 is negative)")
print(f"Predicted Sentiment: {sentiment}")