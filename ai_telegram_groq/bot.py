# bot.py — versión POO con memoria de conversación + teclado
import os, io, json, base64, random, platform, asyncio, logging, requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from groq import AsyncGroq
from collections import defaultdict, deque

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tribgo.oop")

# Fix event loop en Windows
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


class TelegramGroqBot:
    """Bot POO con memoria de conversación y UX con teclado."""
    def __init__(self, base: Path):
        self.base = base

        # .env, config y preguntas
        load_dotenv(self.base / ".env")
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.tg_token:
            raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env")
        if not self.groq_key:
            log.warning("Falta GROQ_API_KEY en .env (visión/STT podrían fallar).")

        with open(self.base / "config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(self.base / "preguntas.json", "r", encoding="utf-8") as f:
            self.preg = json.load(f)

        # Cliente Groq
        self.client = AsyncGroq(api_key=self.groq_key)

        # (Opcional) adaptadores
        try:
            from modules.adapters_sentiment import analyze_sentiment  # noqa: F401
            self.analyze_sentiment = analyze_sentiment
        except Exception:
            self.analyze_sentiment = None
        try:
            from modules.adapters_voice import tts_synthesize  # noqa: F401
            self.tts_synthesize = tts_synthesize
        except Exception:
            self.tts_synthesize = None

        # 🧠 Memoria de conversación (por chat)
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=12))  # ~6 turnos (user+assistant)
        self.max_turns = 6
        self.system_prompt = (
            "Eres un asistente técnico y claro. "
            "Responde en español y recuerda el contexto de la conversación."
        )

    # ---------- utilitarios ----------
    @staticmethod
    def _img_to_b64(pil_img: Image.Image, fmt="JPEG") -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _all_categories(self):
        return [t["id"] for t in self.preg.get("topics", [])]

    def _pick_question(self, category: str | None):
        cats = {t["id"]: t["preguntas"] for t in self.preg.get("topics", [])}
        if not cats:
            return None, None
        if not category or category not in cats:
            category = random.choice(list(cats.keys()))
        qs = cats.get(category, [])
        return category, (random.choice(qs) if qs else None)

    # ---------- memoria ----------
    def _remember(self, chat_id: int, role: str, content: str):
        self.history[chat_id].append({"role": role, "content": content})

    def _reset_history(self, chat_id: int):
        self.history.pop(chat_id, None)

    def _build_messages(self, chat_id: int, user_text: str) -> list[dict]:
        hist = list(self.history[chat_id])[-(self.max_turns * 2):]
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(hist)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    # ---------- llamadas Groq ----------
    async def groq_chat(self, chat_id: int, user_text: str) -> str:
        """Chat con contexto de conversación."""
        messages = self._build_messages(chat_id, user_text)
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["chat"],
            messages=messages,
            max_completion_tokens=512,
        )
        reply = chat.choices[0].message.content
        # actualizar memoria
        self._remember(chat_id, "user", user_text)
        self._remember(chat_id, "assistant", reply)
        return reply

    async def groq_vision(self, img: Image.Image, question: str) -> str:
        b64 = self._img_to_b64(img, "JPEG")
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["vision"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_completion_tokens=384,
        )
        return chat.choices[0].message.content

    async def groq_transcribe(self, file_bytes: bytes, filename: str, language: str = "es") -> str:
        tr = await self.client.audio.transcriptions.create(
            file=(filename, file_bytes),
            model=self.cfg["models"]["stt"],
            language=language,
            response_format="json",
            temperature=0.0,
        )
        return tr.text

    # ---------- handlers ----------
    async def h_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Teclado dinámico según categorías
        cats = self._all_categories()
        buttons = [[KeyboardButton(f"/pregunta {c}")] for c in cats]
        buttons.append([KeyboardButton("/contexto"), KeyboardButton("/reset")])
        buttons.append([KeyboardButton("/ayuda")])
        reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)

        await update.message.reply_text(
            "👋 ¡Hola! Soy tu bot IA.\n"
            "Escribime texto libre o usá los botones.\n"
            "También analizo imágenes y audios 🎙️🖼️",
            reply_markup=reply_markup
        )

    async def h_ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cats = self._all_categories()
        buttons = [[KeyboardButton(f"/pregunta {c}")] for c in cats]
        buttons.append([KeyboardButton("/contexto"), KeyboardButton("/reset")])
        reply_markup = ReplyKeyboardMarkup(buttons, resize_keyboard=True)

        await update.message.reply_text(
            "ℹ️ Puedo hacer lo siguiente:\n"
            "• Texto: respondo con Groq (recuerdo contexto)\n"
            "• Imagen: describo y etiqueto\n"
            "• Audio/voz: transcribo y respondo\n"
            "• /pregunta <categoria>  ·  /contexto  ·  /reset",
            reply_markup=reply_markup
        )

    async def h_pregunta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        cat = (args[0].strip().lower() if args else None)
        sel_cat, q = self._pick_question(cat)
        if not q:
            await update.message.reply_text("No encontré preguntas. Revisá preguntas.json.")
            return
        await update.message.reply_text(f"🗂️ {sel_cat}\n💬 {q}")

    async def h_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        await update.message.chat.send_action(ChatAction.TYPING)
        reply = await self.groq_chat(chat_id, update.message.text)
        # enviar en chunks largos (UX)
        for i in range(0, len(reply), 1500):
            await update.message.reply_text(reply[i:i+1500])

    async def h_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.photo:
            return
        chat_id = update.effective_chat.id
        await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
        ph = update.message.photo[-1]
        file = await context.bot.get_file(ph.file_id)
        bio = io.BytesIO()
        await file.download_to_memory(out=bio)
        bio.seek(0)
        img = Image.open(bio).convert("RGB")
        desc = await self.groq_vision(img, "Describe en español y agrega 3 etiquetas útiles.")
        # memoria del evento (para coherencia del hilo)
        user_note = "Imagen enviada" + (f" | {update.message.caption}" if update.message.caption else "")
        self._remember(chat_id, "user", user_note)
        self._remember(chat_id, "assistant", desc)
        await update.message.reply_text(desc)

    async def h_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        voice = update.message.voice
        audio = update.message.audio
        doc = update.message.document
        chat_id = update.effective_chat.id

        file = None
        filename = "audio.ogg"
        if voice:
            file = await context.bot.get_file(voice.file_id)
            filename = "voice.ogg"
        elif audio:
            file = await context.bot.get_file(audio.file_id)
            filename = audio.file_name or "audio.bin"
        elif doc and ((doc.mime_type or "").startswith("audio/") or (doc.mime_type or "").startswith("video/")):
            # En PTB v21 no existe filters.Document.AUDIO → usamos mime types
            file = await context.bot.get_file(doc.file_id)
            filename = doc.file_name or "audio.bin"
        else:
            await update.message.reply_text("Envíame una nota de voz o archivo de audio.")
            return

        bio = io.BytesIO()
        await file.download_to_memory(out=bio)
        bio.seek(0)

        text = await self.groq_transcribe(bio.getvalue(), filename=filename, language="es")
        reply = await self.groq_chat(chat_id, text)
        await update.message.reply_text(f"📝 {text}\n\n🤖 {reply}")

    async def h_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self._reset_history(chat_id)
        await update.message.reply_text("🧹 Memoria de conversación borrada.")

    async def h_contexto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        hist = list(self.history.get(chat_id, []))
        if not hist:
            await update.message.reply_text("Sin contexto guardado.")
            return
        preview = []
        for m in hist[-10:]:
            role = "👤" if m["role"] == "user" else "🤖"
            text = m["content"]
            if len(text) > 120:
                text = text[:120] + "…"
            preview.append(f"{role} {text}")
        await update.message.reply_text("📚 Contexto reciente:\n" + "\n".join(preview))

    # ---------- wiring / arranque ----------
    def _check_token(self):
        r = requests.get(f"https://api.telegram.org/bot{self.tg_token}/getMe", timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"getMe ok=false: {data}")
        log.info(f"Token OK para: @{data['result']['username']} (id={data['result']['id']})")

    def build_app(self):
        self._check_token()
        app = ApplicationBuilder().token(self.tg_token).build()
        app.add_handler(CommandHandler("start", self.h_start))
        app.add_handler(CommandHandler("ayuda", self.h_ayuda))
        app.add_handler(CommandHandler("pregunta", self.h_pregunta))
        app.add_handler(CommandHandler("contexto", self.h_contexto))
        app.add_handler(CommandHandler("reset", self.h_reset))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.h_text))
        app.add_handler(MessageHandler(filters.PHOTO, self.h_photo))
        # v21: no hay filters.Document.AUDIO → usamos MimeType
        doc_audio_or_video = filters.Document.MimeType("audio/") | filters.Document.MimeType("video/")
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | doc_audio_or_video, self.h_audio))
        return app

    def run(self):
        log.info("Inicializando aplicación Telegram (POO)…")
        app = self.build_app()
        log.info("Iniciando polling… (si queda aquí, ya está escuchando)")
        app.run_polling(allowed_updates=None, drop_pending_updates=True)


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    TelegramGroqBot(BASE).run()
