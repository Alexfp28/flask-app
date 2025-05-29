import telebot
from supabase import create_client, Client
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os

bot = telebot.TeleBot(os.getenv("TELEGRAM_TOKEN"))
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

# url = "https://tu-proyecto.supabase.co"
# key = "tu-clave-secreta"
# Telegram Bot
# bot = telebot.TeleBot('8075764581:AAFh9tt_wKeM1NiuZ9ciTrdDblLw6FU0kow')

# Supabase

supabase: Client = create_client(url, key)

# Modelo de embeddings y pregunta-respuesta
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=100)

# Obtener contexto relevante desde Supabase
def buscar_en_chroma(pregunta):
    # Paso 1: Embedding de la pregunta
    pregunta_emb = embedding_model.encode(pregunta, convert_to_tensor=True)

    # Paso 2: Obtener documentos desde Supabase
    data = supabase.table('documentos').select("contenido").execute()
    documentos = [doc['contenido'] for doc in data.data]

    if not documentos:
        return []

    # Paso 3: Calcular similitud
    docs_emb = embedding_model.encode(documentos, convert_to_tensor=True)
    similitudes = util.cos_sim(pregunta_emb, docs_emb)[0]

    top_indices = similitudes.argsort(descending=True)[:3]  # Top 3 más relevantes
    contextos_relevantes = [documentos[i] for i in top_indices]

    return "\n".join(contextos_relevantes)

# Generar respuesta
def responder_pregunta(pregunta):
    contextos = buscar_en_chroma(pregunta)

    if not contextos:
        return "No encontré información relevante en el documento."

    prompt = f"""
    Eres un asistente de IA experto en análisis de documentos.
    Responde de manera clara y amable. Evita caracteres especiales.

    Contexto:
    {contextos}

    Pregunta: {pregunta}

    Respuesta:
    """

    respuesta = qa_model(prompt)[0]['generated_text'].split("Respuesta:")[-1].strip()
    return respuesta

# Manejadores de mensajes
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "¡Hola! Soy un bot de consulta educativa. Hazme una pregunta.")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    pregunta = message.text
    respuesta = responder_pregunta(pregunta)
    bot.reply_to(message, respuesta)

if __name__ == '__main__':
    bot.infinity_polling()
