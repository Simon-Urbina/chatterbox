import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo...")
modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)
texto = "Hola, bienvenidos a Inteligencia Artificial Generativa, vamos a crear audios, imagenes, texto, y si el pc aguanta... videos"

print("Generando audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7, # Expresión o emoción
    cfg_weight=0.4, # Ajusta el ritmo, si la voz va demasiado rápido
    temperature=0.8, # Controla variabilidad (más alta: más variabilidad = menos predecible)
    repetition_penalty=1.2, # Penaliza repeticiones extrañas
    min_p=0.05, # Filtra opciones improbables durante la generación
    top_p=1.0 # Controla el muestreo probabilístico, 1.0 lo deja abierto (básicamente)
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado")
