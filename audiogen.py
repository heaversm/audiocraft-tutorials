import os
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import argparse
from flask import Flask, render_template, jsonify, request, send_from_directory


model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate [duration] seconds.

def generate_audio(descriptions):
  if not os.path.exists('audio_files'):
    os.mkdir('audio_files')

  wav = model.generate(descriptions)  # generates samples for all descriptions in array.
  results = []

  for idx, one_wav in enumerate(wav):
        print(descriptions[idx])
        filename = f'{idx}'
        file_path = os.path.join('audio_files', filename)  # 'audio_files' is the directory to save the files
        audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        results.append(f'Sample {idx}')

  return results

app = Flask(__name__)

@app.route("/generate_audio", methods=['POST'])
def generate_audio_route():
    print("Generating audio")
    data = request.get_json()
    descriptions = data.get("descriptions")
    if descriptions:
        results = generate_audio(descriptions)
        return jsonify({"results": results})
    else:
        return jsonify({"error": "No descriptions provided"})
    
@app.route("/download_audio/<int:file_id>")
def download_audio(file_id):
    directory = 'audio_files'
    filename = f'{file_id}.wav'
    return send_from_directory(directory, filename, as_attachment=True)

@app.route("/")
def generate_home_route():
    return render_template('index.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio based on descriptions.")
    parser.add_argument("descriptions", nargs='+', help="List of descriptions for audio generation")
    args = parser.parse_args()
    
    # generate_audio(args.descriptions)