import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import os
import gradio as gr

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

def ui_full():
   with gr.Blocks() as interface:
      gr.Markdown(
            """
            # Audiogen
            
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
      interface.queue().launch()
      
ui_full()