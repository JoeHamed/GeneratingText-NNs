# Langigan's Ball Song Generation using TensorFlow

This project demonstrates how to train a model to generate lyrics similar to a given song, "Lanigan's Ball", using TensorFlow. The model learns from the lyrics and generates new sequences of text by predicting the next word based on a given seed text.

## Requirements

To run the code, you'll need the following libraries:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

You can install the required dependencies by running the following:

```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

- **Data**: The lyrics of the song "Lanigan's Ball" are provided as a string.
- **Text Preprocessing**: The lyrics are tokenized, and integer sequences are generated for training the model.
- **Model**: A bidirectional LSTM model is built using TensorFlow, which learns to predict the next word in a sequence.
- **Training**: The model is trained using the processed sequences of text.
- **Text Generation**: After training, the model can generate new song lyrics based on a seed phrase.

## Key Concepts

1. **Text Vectorization**: The song lyrics are tokenized and converted into integer sequences.
2. **N-Grams**: Sub-phrases of the song are generated to train the model.
3. **Neural Network**: A bidirectional LSTM network is used for text generation.
4. **One-Hot Encoding**: Labels (the next word) are one-hot encoded for classification.
5. **Prediction**: The model predicts the next word based on the current sequence.

## How It Works

### Step 1: Prepare the Data
The song lyrics are split into lines, and each line is tokenized into words. An integer sequence is generated for each line, and n-grams (subsequences) are created to train the model.

### Step 2: Build the Model
A Sequential model is built with the following layers:
- **Embedding Layer**: Converts integer sequences into dense vectors.
- **Bidirectional LSTM**: Captures both past and future context of the sequence.
- **Dense Layer**: Outputs a probability distribution over the vocabulary for the next word.

### Step 3: Train the Model
The model is trained on the sequences of text, learning to predict the next word based on the current sequence.

### Step 4: Generate New Text
Once trained, the model can generate new lyrics by feeding in a seed phrase. The model predicts the next word, appends it to the seed, and repeats this process until a desired length is reached.

## Usage

### Generate New Song Lyrics

After training, you can use the model to generate new song lyrics based on a seed phrase. Here’s an example of how to generate new lyrics:

```python
seed_text = 'Laurence went to Dublin'
next_words = 100

for _ in range(next_words):
    sequence = vectorize_layer(seed_text)
    sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_sequence_length-1, padding='pre')
    probabilities = model.predict(sequence)
    predicted = np.argmax(probabilities, axis=-1)[0]
    
    if predicted != 0:
        output_word = vocabulary[predicted]
        seed_text += ' ' + output_word

print(seed_text)
```

### Plot Training Accuracy
After training, you can plot the model's accuracy over epochs:

```python
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')
```

## Example Output

After training the model and providing a seed text, you might get the following output:

```
Laurence went to Dublin and learned new steps for Lanigans Ball. There were lashings of punch and wine for the ladies, potatoes and cakes, there was bacon and tea...
```
