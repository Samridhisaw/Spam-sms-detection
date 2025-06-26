import tkinter as tk
from tkinter import messagebox
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the trained model and vectorizer
with open('trained_modle.sav', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.sav', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Function to clean and preprocess the input message
def preprocess_message(message):
    message = re.sub('[^a-zA-Z]', ' ', message)
    message = message.lower().split()
    return ' '.join(message)

# Function to predict if the input is ham or spam
def classify_message():
    message = message_entry.get()
    if not message.strip():
        messagebox.showwarning("Input Error", "Please enter a message to classify.")
        return

    cleaned_message = preprocess_message(message)
    transformed_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "Ham"
    else:
        result = "Spam"

    result_label.config(text=f"Prediction: {result}", fg="green" if result == "Ham" else "red")

# Create the GUI window
root = tk.Tk()
root.title("Spam Message Classifier")
root.geometry("400x300")

# Title Label
title_label = tk.Label(root, text="Spam Message Classifier", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

# Input Label and Entry
input_label = tk.Label(root, text="Enter your message:", font=("Arial", 12))
input_label.pack(pady=5)

message_entry = tk.Entry(root, font=("Arial", 12), width=50)
message_entry.pack(pady=5)

# Classify Button
classify_button = tk.Button(root, text="Classify", font=("Arial", 12), command=classify_message)
classify_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=20)

# Run the GUI
root.mainloop()
