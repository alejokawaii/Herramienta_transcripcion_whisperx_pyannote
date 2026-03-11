#Librerias:
import os
import socket
#Ver si hay internet, si no hay modo Offline
def comprobar_internet():
    try:
        # Intenta conectarse a los servidores de Hugging Face de forma rápida
        socket.create_connection(("huggingface.co", 443), timeout=3)
        return True
    except OSError:
        return False
    
if comprobar_internet():
    print("🌐 Conexión a internet detectada. Modo Online activado")
    os.environ["HF_HUB_OFFLINE"] = "0" 
else:
    print("⚠️ No hay internet. Modo Offline activado (solo funciona si anteriormente se ha usado con idiomas base o ya descargadas)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    
import whisperx
import gradio as gr
import gc 
import datetime
import subprocess
import torch
import pyannote.audio
import sys
import subprocess
import shutil

#Leer token_hf (Permiso Read para el modelo de diarización)
def obtener_token():
    try:
        with open("token.txt", "r") as archivo:
            return archivo.read().strip()
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo token.txt")
        return None
    
token_hf = obtener_token()

#Configuración y carga de modelos:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")
batch_size = 1 # reducir si baja memoria GPU 
compute_type = "float16" if device == "cuda" else "int8"

os.environ['GRADIO_SERVER_TIMEOUT'] = '400000'
os.environ['GRADIO_NO_SERVER_CHUNK'] = '1'

# 1. Transcripción

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
print(f"Modelo Whisper large-v2 cargado exitosamente")
diarize_model = whisperx.DiarizationPipeline(use_auth_token=token_hf, device=device)
print("Modelo de Diarización cargado con éxito.")


def transcribir_audio(audio_file_path):
    if audio_file_path is None:
        return None, None
    
    # 1. Cargar el contenido del audio (array)
    try:
        audio = whisperx.load_audio(audio_file_path)
    except Exception as e:
        error_msg = f"❌ Error al cargar el archivo de audio. Asegúrate de que el formato sea compatible (e.g., MP3, WAV): {e}"
        return error_msg, None

    # 2. Transcribir
    try:
        result = model.transcribe(audio, batch_size=batch_size)
        language_code = result["language"] 
        print(f"Transcripción completada. Idioma detectado: {language_code}")
    except Exception as e:
        error_msg = f"❌ Error durante la transcripción con WhisperX: {e}"
        return error_msg, None


    # 3. Alineación
    try:
        # Cargar modelo de alineación (se hace aquí para usar language_code)
        model_a, metadata = whisperx.load_align_model(language_code, device=device)
        print("Alineando transcripción...")
        
        # La alineación usa el contenido del audio (array)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        print(f"Alineación completada.")
    except Exception as e:
        error_msg = f"❌ Error durante la alineación (Asegúrate de que el idioma es correcto o inténtalo sin diarización): {e}"
        return error_msg, None
        
    # 4. Diarización (Usando la RUTA del archivo)
    try:
        print("-> Realizando Diarización (pyannote.audio)...")
        diarize_segments = diarize_model(audio_file_path) 
        
        # 5. Asignación de hablantes
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("-> Diarización finalizada.")
    except Exception as e:
        transcription_only = "".join(seg['text'] for seg in result['segments']) if 'segments' in result else "Error de segmentación."
        error_msg = f"⚠️ La diarización falló con error: {e}.\n\nTranscripción sin separación de hablantes:\n{transcription_only}"
        return error_msg, None
    
    
    # 6. Formateo y Limpieza
    formatted_text = ""
    current_speaker = None
        
    for segment in result["segments"]:
        speaker = segment["speaker"] if "speaker" in segment else "Hablante Desconocido"
        start_time_sec = segment["start"]
        start_time = str(datetime.timedelta(seconds=int(start_time_sec)))
            
        if speaker != current_speaker:
            formatted_text += f"\n\n[{start_time} - **{speaker}**]\n"
            current_speaker = speaker
        
        formatted_text += segment["text"] + " "
        
    # Dejar CPU vacia 
    del model_a, metadata, audio
    gc.collect()
    torch.cuda.empty_cache()
    
    # 1. Crear una ruta temporal para el archivo de salida txt
    formatted_text_str = formatted_text.strip()
    output_filename = os.path.join(os.getcwd(), "transcripcion_resultado.txt")
    
    # 2. Guardar el texto formateado en archivo
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(formatted_text)
        
    # 3. Devolver la RUTA de archivo al componente gr.File de Gradio
    return formatted_text_str, output_filename

#Interfaz de Gradio:
def create_app():
    return gr.Interface(
        fn=transcribir_audio,
        inputs=gr.Audio(type="filepath", label="Sube tu archivo de audio", streaming=False),
        outputs=[
            gr.Textbox(label="📝 Transcripción (Para Visualización)", lines=15),
            gr.File(label="⬇️ Descargar Transcripción", file_types=[".txt"]),
        ],
        title="🎙️ Transcriptor WhisperX (Local)",
        description="...",
        theme="soft",
        allow_flagging="never"
    )
    
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        inbrowser=True,
        share=False
    )

#El presente código lo ha hecho Daniel Alejandro Salcedo Perez.
#Se ha realizado una modificación y redistribución del código original a la Clínica Jurídica por la Justicia Social de la EHU de acuerdo a
#la clausula de Copyright siguiente:

#BSD 2-Clause License

#Copyright (c) 2024, Max Bain

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:

#1. Redistributions of source code must retain the above copyright notice, this
#list of conditions and the following disclaimer.

#2. Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation
#and/or other materials provided with the distribution.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#https://github.com/m-bain/whisperX

#En cumplimiento del acuerdo de copyright del modelo de diarización de Pyannote, se reproduce dicho acuerdo:
#MIT License
#Copyright (c) 2020 CNRS
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#https://huggingface.co/pyannote/speaker-diarization

#La presente distribución del software mencionado ha sido creado por Daniel Alejandro Salcedo Pérez a lo largo de los años de 2025 y 2026 para su uso por parte de los alumnos y alumnas de la Clínica Jurídica para la Justicia Social de la EHU.