import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import argparse
from flask import Flask, render_template, jsonify, request


model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate [duration] seconds.

def generate_audio(descriptions):
  wav = model.generate(descriptions)  # generates samples for all descriptions in array.
  results = []

  for idx, one_wav in enumerate(wav):
      # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
      audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
      print(f'Generated {idx}.wav.')
      results.append(f'{idx}.wav')

  return results

app = Flask(__name__)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio based on descriptions.")
    parser.add_argument("descriptions", nargs='+', help="List of descriptions for audio generation")
    args = parser.parse_args()
    
    # generate_audio(args.descriptions)