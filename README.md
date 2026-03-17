# 🎙️ Transcriptor Local con Diarización (WhisperX & Pyannote)

Este script está pensado para ejecutarse en **Windows 11** utilizando Miniconda o Anaconda con un entorno de Python (en este caso, 'whisperx.yaml'). La aplicación abrirá una interfaz gráfica en tu navegador web predeterminado, pero **todo el procesamiento de audio se ejecutará en local** en tu propio ordenador (por eso, pondrá local en la dirección). 

Hay dos modelos de Inteligencia Artificial trabajando en conjunto en este código:
* [**WhisperX:**](https://github.com/m-bain/whisperx) Para la transcripción del texto.
* [**Pyannote:**](https://huggingface.co/pyannote/speaker-diarization-3.1) Para la diarización (identificación de los diferentes hablantes).

---

## 🔒 Privacidad y Seguridad de Datos

La única acción que requiere conexión a internet al iniciar el programa es la verificación inicial en Hugging Face y descarga del modelo de diarización. 

**El token de Hugging Face que se incluya en el código no afecta la privacidad de tus audios ni de tus transcripciones.** * Su función es estrictamente de autenticación; actúa como una clave personal para demostrar que eres un usuario registrado que ha aceptado los términos de uso de `pyannote/speaker-diarization`.
* El token te da permiso para obtener la herramienta, pero **el análisis se realiza íntegramente en el interior de tu ordenador**. Ningún dato de audio, ni la transcripción generada, sale a internet.
* *(Nota: A futuro, el código en Python puede modificarse para cifrar los datos resultantes si se requiere un mayor nivel de seguridad y privacidad).*

---

## Guía:
1. Al descargar la carpeta, deberás descomprimir los archivos. Una vez dentro de la carpeta descomprimida instala Miniconda con el archivo Miniconda3-latest-Windows-x86_64.exe. Este archivo se encuentra dentro de la sub-carpeta 1. Instalar programas, tal cual te aparezca la configuración (solo darle "OK" y "NEXT" a todo).
2. Selecciona la carpeta "Transcriptor Local", dale click derecho y copia la ruta de acceso.
3. Una vez instalada Miniconda, busca en el menú de Inicio de Windows "Anaconda prompt", "Conda prompt" o "Miniconda prompt" para abrir este terminal.
4. Introduce el siguiente comando en la terminal: escribe "cd" y copia y pega la ruta de acceso de la carpeta "Transcriptor Local" con click derecho (Ejemplo: cd "C:\Users\Usuario1\Desktop\Transcriptor Local").
5. Una vez en dicha carpeta dentro del terminal (deberá poner algo similar a "(base) C:\Users\Usuario1\Desktop\Transcriptor Local"), copia y pega el siguiente comando: conda env create -f env_anaconda_whisperx.yaml (decir que sí a los términos de servicio de miniconda si pregunta).
6. Este proceso tardará varios minutos (ya que está descargando PyTorch y otros modelos grandes). Cuando pregunte "Proceed ([y]/n)?", escribe "y" y presiona Enter.
7. Una vez terminada la instalación, debes entrar al entorno creado. Introduce el siguiente comando en el terminal: conda activate whisperx
8.1 Por si acaso la herramienta FFmpeg (que lee y reconoce los archivos de audio) no se abre bien, instala en el entorno esta librería. Introduce en la terminal el siguiente comando: conda install -c conda-forge ffmpeg -y
8.2 Si tampoco funciona la página en la web (Error Windows 32: No se encuentra el archivo...), entra en la carpeta "Transcriptor Local", busca la subcarpeta llamada "ffmpeg-2026-02-18-git-52b676bb29-full_build" y entra después en la subcarpeta "bin".
Asegúrate de que dentro de bin existan tres archivos (ffmpeg.exe,ffplay.exe y ffprobe.exe), y copia la ruta completa de esa subcarpeta bin. Una vez hecho esto, pulsa la tecla Windows, escribe "Variables de entorno" y selecciona "Editar las variables de entorno del sistema". En la lista "Variables del sistema", busca la que se llama Path y hazle doble clic, haz clic en Nuevo, pega la ruta que copiaste y pulsa Aceptar en todas las ventanas. Reinicia el ordenador
9. Si se ha hecho bien, el nombre (whisperx) aparece al inicio de la línea de comandos. Ya está listo. Por último, introduce en la terminal: python app_transcriptor.py
10. Empezara a ejecutarse el código y el programa cargará los modelos. Si se está ejecutando con "cuda" será más rápido pero también está pensado para funcionar con la "cpu". Sin embargo, es recomendable que ante audios muy grandes se cierren los demás programas. 

Inmediatamente después, el navegador predeterminado (Google Chrome, Firefox, Edge, Safari, etc.) se abrirá automáticamente con la interfaz de la aplicación. Se podrá seleccionar los audios a transcribir y diarizar, en caso de error, es recomendable que si es un audio largo se corte en trozos más pequeños y manejables.

Suele tardar el triple de la duración del audio a transcribir (p.ej. alrededor de 1h de audio tardará 3h).
