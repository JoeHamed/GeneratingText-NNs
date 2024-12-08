import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Building the word vocabulary
# dataset is the lyrics of Langigan's Ball
data = ("In the town of Athy one Jeremy Lanigan \n"
        " Battered away til he hadnt a pound. \n"
        "His father died and made him a man again \n"
        " Left him a farm and ten acres of ground. \n"
        "He gave a grand party for friends and relations \n"
        "Who didnt forget him when come to the wall, \n"
        "And if youll but listen Ill make your eyes glisten \n"
        "Of the rows and the ructions of Lanigans Ball. \n"
        "Myself to be sure got free invitation, \n"
        "For all the nice girls and boys I might ask, \n"
        "And just in a minute both friends and relations \n"
        "Were dancing round merry as bees round a cask. \n"
        "Judy ODaly, that nice little milliner, \n"
        "She tipped me a wink for to give her a call, \n"
        "And I soon arrived with Peggy McGilligan \n"
        "Just in time for Lanigans Ball. \n"
        "There were lashings of punch and wine for the ladies, \n"
        "Potatoes and cakes; there was bacon and tea, \n"
        "There were the Nolans, Dolans, OGradys \n"
        "Courting the girls and dancing away. \n"
        "Songs they went round as plenty as water, \n"
        "The harp that once sounded in Taras old hall,\n"
        "Sweet Nelly Gray and The Rat Catchers Daughter,\n"
        "All singing together at Lanigans Ball. \n"
        "They were doing all kinds of nonsensical polkas \n"
        "All round the room in a whirligig. \n"
        "Julia and I, we banished their nonsense \n"
        "And tipped them the twist of a reel and a jig. \n"
        "Ach mavrone, how the girls got all mad at me \n"
        "Danced til youd think the ceiling would fall. \n"
        "For I spent three weeks at Brooks Academy \n"
        "Learning new steps for Lanigans Ball. \n"
        "Three long weeks I spent up in Dublin, \n"
        "Three long weeks to learn nothing at all,\n"
        " Three long weeks I spent up in Dublin, \n"
        "Learning new steps for Lanigans Ball. \n"
        "She stepped out and I stepped in again, \n"
        "I stepped out and she stepped in again, \n"
        "She stepped out and I stepped in again, \n"
        "Learning new steps for Lanigans Ball. \n"
        "Boys were all merry and the girls they were hearty \n"
        "And danced all around in couples and groups, \n"
        "Til an accident happened, young Terrance McCarthy \n"
        "Put his right leg through miss Finnertys hoops. \n"
        "Poor creature fainted and cried Meelia murther, \n"
        "Called for her brothers and gathered them all. \n"
        "Carmody swore that hed go no further \n"
        "Til he had satisfaction at Lanigans Ball. \n"
        "In the midst of the row miss Kerrigan fainted, \n"
        "Her cheeks at the same time as red as a rose. \n"
        "Some of the lads declared she was painted, \n"
        "She took a small drop too much, I suppose. \n"
        "Her sweetheart, Ned Morgan, so powerful and able, \n"
        "When he saw his fair colleen stretched out by the wall, \n"
        "Tore the left leg from under the table \n"
        "And smashed all the Chaneys at Lanigans Ball. \n"
        "Boys, oh boys, twas then there were runctions. \n"
        "Myself got a lick from big Phelim McHugh. \n"
        "I soon replied to his introduction \n"
        "And kicked up a terrible hullabaloo. \n"
        "Old Casey, the piper, was near being strangled. \n"
        "They squeezed up his pipes, bellows, chanters and all. \n"
        "The girls, in their ribbons, they got all entangled \n"
        "And that put an end to Lanigans Ball.")

# Corpus List
corpus = data.lower().split("\n")
# Preview the results
print(corpus)

# Initialize the Vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization()
# Build the vocabulary
vectorize_layer.adapt(corpus)

# Get the vocab
vocabulary = vectorize_layer.get_vocabulary()
# Get the size of the vocab
vocab_size = len(vocabulary)

print(f'{vocabulary}')
print(f'{vocab_size}')

# Initialize the sequences list
input_sequences = []

# Loop over every line
for line in corpus:
    # Generate the integer sequence of the current line
    sequence = vectorize_layer(line).numpy()
    # Loop over the line several times to generate the subphrases
    for i in range(1, len(sequence)):
        # Generate the subphrase
        n_gram_sequence = sequence[:i+1]
        # Append the subphrase to the sequence list
        input_sequences.append(n_gram_sequence)

# Get the length of the longest line
max_sequence_length = max([len(sequence) for sequence in input_sequences])

# Pad all sequences
input_sequences = np.array(tf.keras.utils.pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

# Create inputs and label by splitting the last taken in the subphrases
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

# Convert the label into one-hot arrays
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size) # Convert a class vector(int) to binary class matrix


# Checking the result of the first line of the song
sentence = corpus[0].split()
print(f'sample sentence: {sentence}')
# Initialize token list
token_list=[]
# Look up the indices of each word and append to the list
for word in sentence:
        token_list.append(vocabulary.index(word))
# Print the token list
print(token_list)

def sequence_to_text(sequence, vocabulary):
        # Loop through the integer sequence and look up the word from the vocab
        words = [vocabulary[index] for index in sequence]
        text = tf.strings.reduce_join(inputs=words, separator=' ').numpy().decode('utf-8')
        return text

print(f'token list: {xs[6]}')
print(f'decoded to text: {sequence_to_text(xs[6], vocabulary)}')
# Print label
print(f'one-hot label: {ys[6]}')
print(f'index of label: {np.argmax(ys[6])}')



# Building the model
model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(max_sequence_length,)),
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_sequence_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Printing the model summary
model.summary()


# Train the model
history = model.fit(x=xs, y=ys, epochs=600)

# Plot utility
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

# Visualize the accuracy
plot_graphs(history, 'accuracy')


# With the model trained, you can now use it to make its own song! The process would look like:
# 1. Feed a seed text to initiate the process.
# 2. Model predicts the index of the most probable next word. <----|
# 3. Look up the index in the reverse word index dictionary.       |---- Repeat till the
# 4. Append the next word to the seed text.                        |     desired length is reached
# 5. Feed the result to the model again.   <-----------------------|



# Define the seed
seed_text = 'Laurence went to Dublin'
# Define total words to predict
next_words = 100

# Loop till desired length is reached
for _ in range(next_words):
    # Convert the seed into an integer sequence
    sequence = vectorize_layer(seed_text)
    # Pad the sequences
    sequence = tf.keras.utils.pad_sequences([sequence],maxlen=max_sequence_length-1, padding='pre')
    # Feed to the model and get the probabilities for each index
    probabilities = model.predict(sequence)
    # Get the index with the highest probability
    predicted = np.argmax(probabilities, axis=-1)[0]
    # OR
    # # Pick a random number from [1,2,3]
    # choice = np.random.choice([1, 2, 3])
    # # Sort the probabilities in ascending order
    # # and get the random choice from the end of the array
    # predicted = np.argsort(probabilities)[0][-choice]
    
    # Ignore if index is 0 because that's the padding
    if predicted != 0:
        # Look up the word associated with the index
        output_word = vocabulary[predicted]
        # Combine with the seed text
        seed_text += ' ' + output_word

print(seed_text)
