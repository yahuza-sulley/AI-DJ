import midi2audio

def convert_midi_to_wav(midi_file):
    midi2audio.FluidSynth().midi_to_audio(midi_file, "static/gen_music.wav")
