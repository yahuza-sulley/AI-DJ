from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
from music21 import *
from model_inference import generate_music

#input_arr = np.random.randn(len(unique_x), 100)
app = Flask(__name__, template_folder="templates")

# Load the best model after training
model = load_model('best_model.h5')

# ... (code for converting MIDI to music21 format and generating music)



@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        #seed_note = request.form["seed_note"]
        #print(seed_note)
        note_name = "..\music.mp3"
        context = {"music_data": note_name}
    return render_template('index.html' )


app.route('/generate_music', methods=['POST, GET'])
def generate():
    # Get user input from the web form
    seed_note = request.form.get('seed_note')

    # ... (code to process user input and generate music using your model)
    
    # Convert the generated music to MIDI format
    # For example, use the function convert_to_midi() defined in your original code
    generate_music()
    
    #Return the URL of the generated MIDI file to the web app
    return jsonify({"music_data": "music.mid"})


if __name__ == '__main__':
    app.run(debug=True)
