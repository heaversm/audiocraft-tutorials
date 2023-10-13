import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import argparse

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # generate [duration] seconds.

def generate_audio(descriptions):
  wav = model.generate(descriptions)  # generates samples for all descriptions in array.
  
  for idx, one_wav in enumerate(wav):
      # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
      audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
      print(f'Generated {idx}th sample.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio based on descriptions.")
    parser.add_argument("descriptions", nargs='+', help="List of descriptions for audio generation")
    args = parser.parse_args()
    
    generate_audio(args.descriptions)