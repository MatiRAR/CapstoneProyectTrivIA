import os, json, io, base64, random, platform, asyncio, logging
from pathlib import Path
import requests
from dotenv import load_dotenv
from PIL import Image
from groq import AsyncGroq
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---- LOGGING ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tribgo")

# ---- FIX EVENT LOOP WINDOWS ----
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- RUTAS Y ENV ----
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not TG_TOKEN:
    raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env (junto a bot.py).")
if not GROQ_API_KEY:
    log.warning("Falta GROQ_API_KEY en .env (visión/STT podrían fallar).")

# ---- ARCHIVOS ----
with open(BASE / "config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)
with open(BASE / "preguntas.json", "r", encoding="utf-8") as f:
    PREG = json.load(f)

# ---- CHEQUEO TOKEN (getMe) ----
try:
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getMe", timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data)
    log.info(f"Token OK para: @{data['result']['username']} (id={data['result']['id']})")
except Exception as e:
    raise SystemExit(f"Token inválido o red bloqueada: {e}")

# ---- CLIENTE GROQ ----
client = AsyncGroq(api_key=GROQ_API_KEY)

# ---- ADAPTADORES (plug-ins opcionales) ----
try:
    from modules.adapters_sentiment import analyze_sentiment
except Exception:
    analyze_sentiment = None
try:
    from modules.adapters_voice import tts_synthesize
except Exception:
    tts_synthesize = None

# ---- Moderación + Prompt central

BANNED = ["insulto1", "insulto2"]
MAX_CHARS = 2000

def _is_allowed(text: str):
    """True/False + mensaje de aviso si corresponde."""
    t = (text or "").strip()
    if not t:
        return False, "No recibí contenido útil. ¿Podés reenviar tu idea?"
    if any(b in t.lower() for b in BANNED):
        return False, "Mantengamos un tono respetuoso 🙏. ¿Podés reformular?"
    if len(t) > MAX_CHARS:
        return False, f"Tu mensaje es muy largo ({len(t)}). ¿Podés resumirlo?"
    return True, None

DEBATE_SYSTEM = (
    "Eres DebateSensei: un asistente conversacional breve, natural y respetuoso. "
    "Da respuestas de 2 a 3 frases, sin listas, sin títulos ni negritas. "
    "Usa humor ligero SOLO si el tema no es sensible; si el tema es serio, prioriza empatía y claridad. "
    "Cierra con UNA única pregunta breve que invite a reflexionar, SOLO si corresponde "
    "(si el tema es muy delicado o la respuesta ya incluye una pregunta del usuario, podés omitirla). "
    "Si el usuario pide profundidad explícita, podés extenderte a 4–6 frases. "
    "Siempre responde en español y en un único párrafo."
)

def build_user_prompt_from_text(user_text: str) -> str:
    return (
        "El usuario compartió una idea u opinión. "
        "Respondé en 2–3 frases, tono natural, sin listas ni negritas. "
        "Aporta un matiz o perspectiva alternativa sin confrontar. "
        "Si el tema es liviano, humor leve es bienvenido; si es sensible o profundo, sé empático. "
        "Cerrá con UNA sola pregunta breve que invite a pensar, solo si corresponde.\n\n"
        f"Mensaje del usuario: {user_text}\n\n"
        "Respuesta:"
    )

def build_user_prompt_from_image(description: str) -> str:
    return (
        "El usuario envió una imagen; abajo está su descripción. "
        "Respondé en 2–3 frases, tono natural, sin listas ni negritas. "
        "Comentá la idea que sugiere y sumá un matiz. "
        "Cerrá con UNA sola pregunta breve que invite a pensar, solo si corresponde.\n\n"
        f"Descripción de la imagen: {description}\n\n"
        "Respuesta:"
    )

# ---- HELPERS GROQ ----
async def groq_chat(prompt: str, system: str = DEBATE_SYSTEM) -> str:
    chat = await client.chat.completions.create(
        model=CFG["models"]["chat"],
        messages=[
            {"role":"system","content":"Eres un asistente técnico, conciso y útil."},
            {"role":"user","content":prompt}
        ],
        max_completion_tokens=220,
        temperature=0.4,
        timeout=30
    )
    return chat.choices[0].message.content


def _img_to_b64(pil_img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

async def groq_vision(pil_img: Image.Image, question: str="Describe la imagen y da 3 etiquetas.") -> str:
    b64 = _img_to_b64(pil_img, "JPEG")
    chat = await client.chat.completions.create(
        model=CFG["models"]["vision"],
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text": question},
                {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        max_completion_tokens=384
    )
    return chat.choices[0].message.content

async def groq_transcribe(file_bytes: bytes, filename: str="audio.ogg", language: str="es") -> str:
    tr = await client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model=CFG["models"]["stt"],
        language=language,
        response_format="json",
        temperature=0.0
    )
    return tr.text

# ---- PREGUNTAS ----
def all_categories():
    return [t["id"] for t in PREG.get("topics", [])]

def pick_question(category: str | None):
    cats = {t["id"]: t["preguntas"] for t in PREG.get("topics", [])}
    if not cats:
        return None, None
    if not category or category not in cats:
        category = random.choice(list(cats.keys()))
    preguntas = cats.get(category, [])
    return category, (random.choice(preguntas) if preguntas else None)

# ---- HANDLERS ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topics = ", ".join(all_categories())
    await update.message.reply_text(
        "¡Hola! Envíame texto, una foto o una nota de voz.\n"
        "Comandos:\n"
        "  /ayuda\n"
        "  /pregunta <categoria>  (o sin categoría para aleatoria)\n"
        f"Categorías: {topics}"
    )

async def ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topics = ", ".join(all_categories())
    await update.message.reply_text(
        "• Texto: respondo con Groq\n"
        "• Imagen: describo y etiqueto la foto\n"
        "• Audio/voz: transcribo con Whisper y respondo\n"
        "• /pregunta <categoria>  (o sin categoría para aleatoria)\n"
        f"Categorías: {topics}"
    )

async def cmd_pregunta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    cat = (args[0].strip().lower() if args else None)
    sel_cat, q = pick_question(cat)
    if not q:
        await update.message.reply_text("No encontré preguntas. Revisá preguntas.json.")
        return
    await update.message.reply_text(f"🗂️ {sel_cat}\n💬 {q}")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    reply = await groq_chat(update.message.text)
    await update.message.reply_text(reply)

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    ph = update.message.photo[-1]
    file = await context.bot.get_file(ph.file_id)
    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)
    img = Image.open(bio).convert("RGB")
    desc = await groq_vision(img, "Describe en español y da 3 etiquetas útiles.")
    await update.message.reply_text(desc)

async def on_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    audio = update.message.audio
    doc = update.message.document

    file = None
    filename = "audio.ogg"
    if voice:
        file = await context.bot.get_file(voice.file_id)
        filename = "voice.ogg"
    elif audio:
        file = await context.bot.get_file(audio.file_id)
        filename = audio.file_name or "audio.bin"
    elif doc and (doc.mime_type or "").startswith(("audio/", "video/")):
        file = await context.bot.get_file(doc.file_id)
        filename = doc.file_name or "audio.bin"
    else:
        await update.message.reply_text("Envíame una nota de voz o archivo de audio.")
        return

    bio = io.BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)

    text = await groq_transcribe(bio.getvalue(), filename=filename, language="es")
    reply = await groq_chat(f"Transcripción del usuario: {text}\nRespondé breve en español.")
    await update.message.reply_text(f"📝 {text}\n\n🤖 {reply}")

def main():
    log.info("Inicializando aplicación Telegram…")
    app = ApplicationBuilder().token(TG_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", ayuda))
    app.add_handler(CommandHandler("pregunta", cmd_pregunta))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))

    log.info("Iniciando polling… (si queda aquí, el bot YA está escuchando)")
    app.run_polling(allowed_updates=None, drop_pending_updates=True)

if __name__ == "__main__":
    main()