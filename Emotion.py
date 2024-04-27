import os
import tkinter as tk
import speech_recognition as sr
import pygame
import pyttsx3
import numpy as np
import pandas as pd  # Import Pandas for handling CSV files
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize pygame for audio playback
pygame.mixer.init()

# Placeholder database of songs with associated moods
song_database = {
        'happy': ['Happy1.mp3', 'Happy2.mp3', 'Happy3.mp3', 'Happy4.mp3', 'Happy5.mp3', 'Happy6.mp3', 'Happy7.mp3', 'Happy8.mp3', 'Happy9.mp3', 'Happy10.mp3', 'Happy11.mp3', 'Happy12.mp3', 'Happy13.mp3', 'Happy14.mp3', 'Happy15.mp3', 'Happy16.mp3', 'Happy17.mp3', 'Happy18.mp3', 'Happy19.mp3', 'Happy20.mp3'],
        'sad': ['Sad1.mp3', 'Sad2.mp3', 'Sad3.mp3', 'Sad4.mp3', 'Sad5.mp3', 'Sad6.mp3', 'Sad7.mp3', 'Sad8.mp3', 'Sad9.mp3', 'Sad10.mp3', 'Sad11.mp3', 'Sad12.mp3', 'Sad13.mp3', 'Sad14.mp3', 'Sad15.mp3', 'Sad16.mp3', 'Sad17.mp3', 'Sad18.mp3'],
        'chill': ['Chill1.mp3', 'Chill2.mp3', 'Chill3.mp3', 'Chill4.mp3', 'Chill5.mp3', 'Chill6.mp3', 'Chill7.mp3', 'Chill8.mp3', 'Chill9.mp3', 'Chill10.mp3', 'Chill11.mp3', 'Chill12.mp3', 'Chill13.mp3', 'Chill14.mp3', 'Chill15.mp3', 'Chill16.mp3', 'Chill17.mp3', 'Chill18.mp3', 'Chill19.mp3', 'Chill20.mp3'],
        'lovemood': ['Lovemood1.mp3', 'Lovemood2.mp3', 'Lovemood3.mp3', 'Lovemood4.mp3', 'Lovemood5.mp3', 'Lovemood6.mp3', 'Lovemood7.mp3', 'Lovemood8.mp3', 'Lovemood9.mp3', 'Lovemood10.mp3', 'Lovemood11.mp3', 'Lovemood12.mp3', 'Lovemood13.mp3', 'Lovemood14.mp3', 'Lovemood15.mp3', 'Lovemood16.mp3', 'Lovemood17.mp3', 'Lovemood18.mp3', 'Lovemood19.mp3', 'Lovemood20.mp3'],
        'dance': ['Dance1.mp3', 'Dance2.mp3', 'Dance3.mp3', 'Dance4.mp3', 'Dance5.mp3', 'Dance6.mp3', 'Dance7.mp3', 'Dance8.mp3', 'Dance9.mp3', 'Dance10.mp3', 'Dance11.mp3', 'Dance12.mp3', 'Dance13.mp3', 'Dance14.mp3', 'Dance15.mp3', 'Dance16.mp3', 'Dance17.mp3', 'Dance18.mp3', 'Dance19.mp3', 'Dance20.mp3'],
        'partymood': ['Partymood1.mp3', 'Partymood2.mp3', 'Partymood3.mp3', 'Partymood4.mp3', 'Partymood5.mp3', 'Partymood6.mp3', 'Partymood7.mp3', 'Partymood8.mp3', 'Partymood9.mp3', 'Partymood10.mp3', 'Partymood11.mp3', 'Partymood12.mp3', 'Partymood13.mp3', 'Partymood14.mp3', 'Partymood15.mp3', 'Partymood16.mp3', 'Partymood17.mp3', 'Partymood18.mp3', 'Partymood19.mp3', 'Partymood20.mp3'],
        'relax': ['Relax1.mp3', 'Relax2.mp3', 'Relax3.mp3', 'Relax4.mp3', 'Relax5.mp3', 'Relax6.mp3', 'Relax7.mp3', 'Relax8.mp3', 'Relax9.mp3', 'Relax10.mp3', 'Relax11.mp3', 'Relax12.mp3', 'Relax13.mp3', 'Relax14.mp3', 'Relax15.mp3', 'Relax16.mp3', 'Relax17.mp3', 'Relax18.mp3', 'Relax19.mp3', 'Relax20.mp3'],
        'sleepy': ['Sleepy1.mp3', 'Sleepy2.mp3', 'Sleepy3.mp3', 'Sleepy4.mp3', 'Sleepy5.mp3', 'Sleepy6.mp3', 'Sleepy7.mp3', 'Sleepy8.mp3', 'Sleepy9.mp3', 'Sleepy10.mp3', 'Sleepy11.mp3', 'Sleepy12.mp3', 'Sleepy13.mp3', 'Sleepy14.mp3', 'Sleepy15.mp3', 'Sleepy16.mp3', 'Sleepy17.mp3', 'Sleepy18.mp3', 'Sleepy19.mp3', 'Sleepy20.mp3'],
        'romantic': ['Romantic1.mp3', 'Romantic2.mp3', 'Romantic3.mp3', 'Romantic4.mp3', 'Romantic5.mp3', 'Romantic6.mp3', 'Romantic7.mp3', 'Romantic8.mp3', 'Romantic9.mp3', 'Romantic10.mp3', 'Romantic11.mp3', 'Romantic12.mp3', 'Romantic13.mp3', 'Romantic14.mp3', 'Romantic15.mp3', 'Romantic16.mp3', 'Romantic17.mp3', 'Romantic18.mp3', 'Romantic19.mp3', 'Romantic20.mp3'],
        'devotional': ['Devotional1.mp3', 'Devotional2.mp3', 'Devotional3.mp3', 'Devotional4.mp3', 'Devotional5.mp3', 'Devotional6.mp3', 'Devotional7.mp3', 'Devotional8.mp3', 'Devotional9.mp3', 'Devotional10.mp3', 'Devotional11.mp3', 'Devotional12.mp3', 'Devotional13.mp3', 'Devotional14.mp3', 'Devotional15.mp3', 'Devotional16.mp3', 'Devotional17.mp3', 'Devotional18.mp3', 'Devotional19.mp3', 'Devotional20.mp3']
}

# Load the trained LSTM model
model_path = "C:/Users/NAVYANAMMU/OneDrive/Major project IV/voice-to-text based MUSIC/mood_prediction_model.h5"
model = load_model(model_path)

# Load the tokenizer
tokenizer_path = "C:/Users/NAVYANAMMU/OneDrive/Major project IV/voice-to-text based MUSIC/tokenizer.pkl"
tokenizer = Tokenizer()
tokenizer.fit_on_texts([""])  # Dummy fit to prevent 'fit_on_texts' called on empty list error
tokenizer = pd.read_pickle(tokenizer_path)

recommended_songs = []
current_song_index = 0

# Function to convert audio to text...
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError:
        return "Could not request results"

MAX_SEQUENCE_LENGTH = 100000  # Example value, you should adjust this based on your training data

# Function to predict mood using LSTM model...
# Function to predict mood using LSTM model...
def predict_mood(text):
    global tokenizer
    global model
    try:
        # Tokenize the text
        sequence = tokenizer.texts_to_sequences([text])
        # Pad sequences
        sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        # Predict mood
        predicted_mood_index = np.argmax(model.predict(sequence), axis=-1)
        predicted_mood_index = int(predicted_mood_index)  # Ensure it's a scalar value
        moods = ['happy', 'sad', 'chill', 'lovemood', 'dance', 'partymood', 'relax', 'sleepy', 'romantic', 'devotional']
        predicted_mood = moods[predicted_mood_index]
        return predicted_mood
    except Exception as e:
        print("Error predicting mood:", str(e))
        return "unknown"


# Function to recommend songs based on mood...
def recommend_songs(mood):
    # Retrieve songs from the database based on the predicted mood
    recommended_songs = song_database.get(mood, [])
    return recommended_songs

# Function to play the song using pygame...
def play_song(song_file):
    try:
        pygame.mixer.music.load(song_file)
        pygame.mixer.music.play()
    except pygame.error as e:
        print("Error during playback:", str(e))

# Function to stop the music
def stop_music():
    pygame.mixer.music.stop()

# Function to handle the entire process
def process_audio_and_recommend(recognized_text):
    global recommended_songs  # Update the global variable
    try:
        display_text.delete('1.0', tk.END)
        display_text.insert(tk.END, f"Recognized Text:\n{recognized_text}")

        mood = predict_mood(recognized_text)
        display_text.insert(tk.END, f"\n\nPredicted Mood: {mood}")

        recommended_songs = recommend_songs(mood)
        display_text.insert(tk.END, f"\n\nRecommended Songs:")
        for song in recommended_songs:
            display_text.insert(tk.END, f"\n- {song}")

            song_path = os.path.join("C:/Users/NAVYANAMMU/OneDrive/Major project IV/voice-to-text based MUSIC/songss", song)
            play_song(song_path)

    except Exception as e:
        display_text.insert(tk.END, f"\n\nError: {str(e)}")

# Function to play the next song
def play_next_song():
    global current_song_index
    global recommended_songs  # Access the global variable
    try:
        current_song_index += 1
        if current_song_index < len(recommended_songs):
            song = recommended_songs[current_song_index]
            print("Playing Next Song:", song)  # Debug print statement
            # Play the recommended song
            song_path = os.path.join("C:/Users/NAVYANAMMU/OneDrive/Major project IV/voice-to-text based MUSIC/songss", song)
            play_song(song_path)
        else:
            print("No more songs in the list.")  # Debug print statement
    except Exception as e:
        print("Error:", str(e))  # Debug print statement

# Function to convert voice to text...
def convert_voice_to_text():
    try:
        recognizer = sr.Recognizer()

        with sr.Microphone() as microphone:
            print('Listening...')
            audio = recognizer.listen(microphone, timeout=5)
            recognized_text = recognizer.recognize_google(audio)
            print('Recognized text:', recognized_text)

            # Convert the recognized text back to speech
            speech_engine.say(recognized_text)
            speech_engine.runAndWait()

            # Process recognized text and recommend songs
            process_audio_and_recommend(recognized_text)

    except sr.UnknownValueError:
        print('Speech Recognition could not understand the audio')
        speech_engine.say('Sorry, I could not understand you.')
        speech_engine.runAndWait()

    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        speech_engine.say('Sorry, I am having trouble connecting to the internet.')
        speech_engine.runAndWait()

# Create GUI
root = tk.Tk()
root.title("VOICE-TO-TEXT-BASED MUSIC RECOMMENDATION SYSTEM USING MACHINE LEARNING")

# Button to convert voice to text...
convert_button = tk.Button(root, text="Convert Voice to Text", command=convert_voice_to_text)
convert_button.pack()

# Button to play next song
play_next_button = tk.Button(root, text="Play Next", command=play_next_song)
play_next_button.pack()

stop_button = tk.Button(root, text="Stop Music", command=stop_music)
stop_button.pack()

display_text = tk.Text(root, wrap=tk.WORD, width=40, height=15)
display_text.pack()

# Initialize text-to-speech engine...
speech_engine = pyttsx3.init()

root.mainloop()


'''
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from nltk.tokenize import sent_tokenize
import os
import numpy as np
import pygame

def summarize_text(text):
    sentences = sent_tokenize(text)
    summary = ' '.join(sentences[:3])  # Take the first three sentences as summary
    return summary

# Function to predict mood (dummy placeholder)
def predict_mood(summary):
    # Use your LSTM model to predict mood here
    # This is just a placeholder
    moods = ['happy', 'sad', 'chill', 'lovemood', 'dance', 'partymood', 'relax', 'sleepy', 'romantic', 'devotional']
    predicted_mood = np.random.choice(moods)
    return predicted_mood

# Function to recommend music based on mood
def recommend_music(mood):
    # Replace 'music_folder' with the absolute path to your music folder
    music_folder = 'C:/Users/NAVYANAMMU/OneDrive/Major project IV/voice-to-text based MUSIC/songss'
    # Map mood to respective music files
    mood_to_music = {
        'happy': ['Happy1.mp3', 'Happy2.mp3', 'Happy3.mp3', 'Happy4.mp3', 'Happy5.mp3', 'Happy6.mp3', 'Happy7.mp3', 'Happy8.mp3', 'Happy9.mp3', 'Happy10.mp3', 'Happy11.mp3', 'Happy12.mp3', 'Happy13.mp3', 'Happy14.mp3', 'Happy15.mp3', 'Happy16.mp3', 'Happy17.mp3', 'Happy18.mp3', 'Happy19.mp3', 'Happy20.mp3'],
        'sad': ['Sad1.mp3', 'Sad2.mp3', 'Sad3.mp3', 'Sad4.mp3', 'Sad5.mp3', 'Sad6.mp3', 'Sad7.mp3', 'Sad8.mp3', 'Sad9.mp3', 'Sad10.mp3', 'Sad11.mp3', 'Sad12.mp3', 'Sad13.mp3', 'Sad14.mp3', 'Sad15.mp3', 'Sad16.mp3', 'Sad17.mp3', 'Sad18.mp3'],
        'chill': ['Chill1.mp3', 'Chill2.mp3', 'Chill3.mp3', 'Chill4.mp3', 'Chill5.mp3', 'Chill6.mp3', 'Chill7.mp3', 'Chill8.mp3', 'Chill9.mp3', 'Chill10.mp3', 'Chill11.mp3', 'Chill12.mp3', 'Chill13.mp3', 'Chill14.mp3', 'Chill15.mp3', 'Chill16.mp3', 'Chill17.mp3', 'Chill18.mp3', 'Chill19.mp3', 'Chill20.mp3'],
        'lovemood': ['Lovemood1.mp3', 'Lovemood2.mp3', 'Lovemood3.mp3', 'Lovemood4.mp3', 'Lovemood5.mp3', 'Lovemood6.mp3', 'Lovemood7.mp3', 'Lovemood8.mp3', 'Lovemood9.mp3', 'Lovemood10.mp3', 'Lovemood11.mp3', 'Lovemood12.mp3', 'Lovemood13.mp3', 'Lovemood14.mp3', 'Lovemood15.mp3', 'Lovemood16.mp3', 'Lovemood17.mp3', 'Lovemood18.mp3', 'Lovemood19.mp3', 'Lovemood20.mp3'],
        'dance': ['Dance1.mp3', 'Dance2.mp3', 'Dance3.mp3', 'Dance4.mp3', 'Dance5.mp3', 'Dance6.mp3', 'Dance7.mp3', 'Dance8.mp3', 'Dance9.mp3', 'Dance10.mp3', 'Dance11.mp3', 'Dance12.mp3', 'Dance13.mp3', 'Dance14.mp3', 'Dance15.mp3', 'Dance16.mp3', 'Dance17.mp3', 'Dance18.mp3', 'Dance19.mp3', 'Dance20.mp3'],
        'partymood': ['Partymood1.mp3', 'Partymood2.mp3', 'Partymood3.mp3', 'Partymood4.mp3', 'Partymood5.mp3', 'Partymood6.mp3', 'Partymood7.mp3', 'Partymood8.mp3', 'Partymood9.mp3', 'Partymood10.mp3', 'Partymood11.mp3', 'Partymood12.mp3', 'Partymood13.mp3', 'Partymood14.mp3', 'Partymood15.mp3', 'Partymood16.mp3', 'Partymood17.mp3', 'Partymood18.mp3', 'Partymood19.mp3', 'Partymood20.mp3'],
        'relax': ['Relax1.mp3', 'Relax2.mp3', 'Relax3.mp3', 'Relax4.mp3', 'Relax5.mp3', 'Relax6.mp3', 'Relax7.mp3', 'Relax8.mp3', 'Relax9.mp3', 'Relax10.mp3', 'Relax11.mp3', 'Relax12.mp3', 'Relax13.mp3', 'Relax14.mp3', 'Relax15.mp3', 'Relax16.mp3', 'Relax17.mp3', 'Relax18.mp3', 'Relax19.mp3', 'Relax20.mp3'],
        'sleepy': ['Sleepy1.mp3', 'Sleepy2.mp3', 'Sleepy3.mp3', 'Sleepy4.mp3', 'Sleepy5.mp3', 'Sleepy6.mp3', 'Sleepy7.mp3', 'Sleepy8.mp3', 'Sleepy9.mp3', 'Sleepy10.mp3', 'Sleepy11.mp3', 'Sleepy12.mp3', 'Sleepy13.mp3', 'Sleepy14.mp3', 'Sleepy15.mp3', 'Sleepy16.mp3', 'Sleepy17.mp3', 'Sleepy18.mp3', 'Sleepy19.mp3', 'Sleepy20.mp3'],
        'romantic': ['Romantic1.mp3', 'Romantic2.mp3', 'Romantic3.mp3', 'Romantic4.mp3', 'Romantic5.mp3', 'Romantic6.mp3', 'Romantic7.mp3', 'Romantic8.mp3', 'Romantic9.mp3', 'Romantic10.mp3', 'Romantic11.mp3', 'Romantic12.mp3', 'Romantic13.mp3', 'Romantic14.mp3', 'Romantic15.mp3', 'Romantic16.mp3', 'Romantic17.mp3', 'Romantic18.mp3', 'Romantic19.mp3', 'Romantic20.mp3'],
        'devotional': ['Devotional1.mp3', 'Devotional2.mp3', 'Devotional3.mp3', 'Devotional4.mp3', 'Devotional5.mp3', 'Devotional6.mp3', 'Devotional7.mp3', 'Devotional8.mp3', 'Devotional9.mp3', 'Devotional10.mp3', 'Devotional11.mp3', 'Devotional12.mp3', 'Devotional13.mp3', 'Devotional14.mp3', 'Devotional15.mp3', 'Devotional16.mp3', 'Devotional17.mp3', 'Devotional18.mp3', 'Devotional19.mp3', 'Devotional20.mp3']
    }
    if mood in mood_to_music:
        return [os.path.join(music_folder, f) for f in mood_to_music[mood]]
    else:
        return []


# Function to convert speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        messagebox.showerror("Error", "Sorry, I couldn't understand the audio.")
        return ""
    except sr.RequestError as e:
        messagebox.showerror("Error", "Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

# Function to play music
def play_music(music_files):
    pygame.init()
    for music_file in music_files:
        absolute_path = os.path.abspath(music_file)
        print("Playing:", absolute_path)  # Print the absolute path
        pygame.mixer.music.load(absolute_path)  # Use the absolute path
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue


def get_recommendation():
    text = speech_to_text()
    if not text:
        return

    summary = summarize_text(text)
    mood = predict_mood(summary)
    recommended_music = recommend_music(mood)

    if recommended_music:
        play_music([os.path.join('music', f) for f in recommended_music])
        messagebox.showinfo("Recommendation", "Recommended Music: {}".format(recommended_music))
    else:
        messagebox.showinfo("Recommendation", "No music recommended for this mood.")

def main():
    root = tk.Tk()
    root.title("Music Recommendation System")

    button = tk.Button(root, text="SPEAK NOW....", command=get_recommendation)
    button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()'''

