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
log = logging.getLogger("debatesensei")

# ---- FIX LOOP WINDOWS ----
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- ENV ----
BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not TG_TOKEN: raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env")

# ---- ARCHIVOS ----
with open(BASE / "config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)
with open(BASE / "preguntas.json", "r", encoding="utf-8") as f:
    PREG = json.load(f)

# ---- TOKEN TEST ----
try:
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getMe", timeout=10)
    r.raise_for_status()
    data = r.json()
    log.info(f"OK: @{data['result']['username']}")
except Exception as e:
    raise SystemExit(f"Token inválido: {e}")

# ---- CLIENT GROQ ----
client = AsyncGroq(api_key=GROQ_API_KEY)

# ---- OPTIONAL ADAPTERS ----
try: from modules.adapters_sentiment import analyze_sentiment
except: analyze_sentiment = None

# ---- MEMORY ----
from modules.debate import ConversationMemory, debate_reply
memory = ConversationMemory(max_turns=5)

# ---- HELPERS ----
async def groq_vision(pil_img: Image.Image, question="Describe esta imagen y lo que intenta comunicar."):
    import base64, io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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

async def groq_transcribe(file_bytes: bytes, filename="audio.ogg", language="es"):
    tr = await client.audio.transcriptions.create(
        file=(filename, file_bytes),
        model=CFG["models"]["stt"],
        language=language,
        response_format="json",
        temperature=0.0
    )
    return tr.text

# ---- COMMANDS ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Envíame texto, memes o notas de voz: yo respondo con debate respetuoso 🙂")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.chat.send_action(ChatAction.TYPING)
    reply = await debate_reply(client, CFG["models"]["chat"], update.message.text, memory, analyze_sentiment)
    await update.message.reply_text(reply)

async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ph = update.message.photo[-1]
    file = await context.bot.get_file(ph.file_id)
    bio = io.BytesIO(); await file.download_to_memory(out=bio); bio.seek(0)
    img = Image.open(bio).convert("RGB")
    vision_text = await groq_vision(img)
    reply = await debate_reply(client, CFG["models"]["chat"], vision_text, memory, analyze_sentiment)
    await update.message.reply_text(reply)

async def on_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice or update.message.audio or update.message.document
    if not voice:
        return await update.message.reply_text("Mandame una nota de voz 🙂")

    file = await context.bot.get_file(voice.file_id)
    bio = io.BytesIO(); await file.download_to_memory(out=bio); bio.seek(0)
    text = await groq_transcribe(bio.getvalue())
    reply = await debate_reply(client, CFG["models"]["chat"], text, memory, analyze_sentiment)
    await update.message.reply_text(f"📝 {text}\n\n🤖 {reply}")

# ---- MAIN ----
def main():
    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.Document.AUDIO, on_audio))
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
