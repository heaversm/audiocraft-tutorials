from flask import Flask, render_template, jsonify, request, send_from_directory
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import argparse
import os


def generate_audio(descriptions):
  if not os.path.exists('audio_files'):
    os.mkdir('audio_files')
  model = AudioGen.get_pretrained('facebook/audiogen-medium')
  model.set_generation_params(duration=5)  # generate [duration] seconds.
  wav = model.generate(descriptions)  # generates samples for all descriptions in array.
  results = []
  
  for idx, one_wav in enumerate(wav):
      filename = f'{idx}'
      file_path = os.path.join('audio_files', filename)  # 'audio_files' is the directory to save the files
      # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
      audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
      print(f'Generated {idx}th sample.')
      results.append(f'{idx}.wav')
  
  return results

app = Flask(__name__)

@app.route("/download_audio/<int:file_id>")
def download_audio(file_id):
    directory = 'audio_files'
    filename = f'{file_id}.wav'
    return send_from_directory(directory, filename, as_attachment=True)

@app.route("/generate_audio", methods=['POST'])
def generate_audio_route():
    data = request.get_json()
    descriptions = data.get("descriptions")
    if descriptions:
        results = generate_audio(descriptions)
        return jsonify({"results": results})
    else:
        return jsonify({"error": "No descriptions provided"})

@app.route("/")
def generate_home_route():
    return render_template('index.html')