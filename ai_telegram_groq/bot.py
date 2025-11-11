# bot.py — POO con memoria + JSON + evaluación + MODO DEBATE (comando único /debate)

import os, io, json, base64, random, platform, asyncio, logging, requests, re
from pathlib import Path
from collections import defaultdict, deque
from functools import wraps
from difflib import SequenceMatcher

from dotenv import load_dotenv
from PIL import Image
from groq import AsyncGroq
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ---------- CONFIG ----------
USE_GROQ_FALLBACK = True
LOW_CONFIDENCE_THRESH = 0.45
MAX_HISTORY_TURNS = 6

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tribgo.debate")

# ---------- FIX EVENT LOOP EN WINDOWS ----------
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---------- DECORADOR SEGURO ----------
def safe_handler(fn):
    @wraps(fn)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await fn(self, update, context, *args, **kwargs)
        except Exception:
            log.exception("Error en handler %s", fn.__name__)
            if update and update.message:
                await update.message.reply_text("⚠️ Ocurrió un error. Intentá de nuevo.")
    return wrapper


class TelegramGroqBot:
    """Bot local+Groq con modo debate toggle (/debate)."""

    def __init__(self, base: Path):
        self.base = base

        load_dotenv(self.base / ".env")
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if not self.tg_token:
            raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env")

        with open(self.base / "preguntas.json", "r", encoding="utf-8") as f:
            self.preg = json.load(f)

        self.cfg = {
            "models": {
                "chat": "llama-3.3-70b-versatile",
                "vision": "meta-llama/llama-4-scout-17b-16e-instruct",
                "stt": "whisper-large-v3-turbo",
            }
        }

        self.client = AsyncGroq(api_key=self.groq_key) if (USE_GROQ_FALLBACK and self.groq_key) else None
        if not self.client:
            log.warning("Groq no configurado, el modo debate no funcionará.")

        # Estado y memoria
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))
        self.system_prompt = "Eres un asistente técnico, claro y amable en español."
        self.debate_mode: dict[int, bool] = defaultdict(lambda: False)
        self.debate_system = (
            "Actúa como un crítico lógico. Refuta cortés y rigurosamente la afirmación del usuario. "
            "Aclara términos, detecta falacias, pide evidencia, ofrece contraejemplos y cierra con síntesis. "
            "Sé breve, razonado y en español neutral."
        )

    # ---------- GROQ ----------
    async def groq_chat(self, cid: int, text: str) -> str:
        if not self.client:
            return "🤖 (modo local) Puedo ayudarte con preguntas del JSON."
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["chat"],
            messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            max_completion_tokens=512,
        )
        reply = chat.choices[0].message.content
        self._remember(cid, "user", text)
        self._remember(cid, "assistant", reply)
        return reply

    async def groq_debate(self, cid: int, text: str) -> str:
        if not self.client:
            return "🧩 (modo local) El modo debate requiere GROQ_API_KEY configurada."
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["chat"],
            messages=[{"role": "system", "content": self.debate_system}, {"role": "user", "content": text}],
            max_completion_tokens=512,
            temperature=0.25,
        )
        reply = chat.choices[0].message.content
        self._remember(cid, "user", text)
        self._remember(cid, "assistant", reply)
        return reply

    # ---------- MEMORIA ----------
    def _remember(self, cid: int, role: str, content: str):
        self.history[cid].append({"role": role, "content": content})

    def _reset_history(self, cid: int):
        self.history.pop(cid, None)

    # ---------- HANDLERS ----------
    @safe_handler
    async def h_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        topics = ", ".join([t["id"] for t in self.preg.get("topics", []) if "id" in t])
        msg = (
            "👋 *¡Hola! Soy tu asistente interactivo.*\n"
            "_Podés enviarme texto, fotos o notas de voz._\n\n"
            "📋 *Comandos principales:*\n"
            "• `/ayuda` — Ver funciones y modos\n"
            "• `/pregunta <categoria>` — Recibir una pregunta aleatoria\n"
            "• `/contexto` — Mostrar tu historial reciente\n"
            "• `/reset` — Borrar memoria de conversación\n"
            "• `/debate` — Alternar modo de *refutación lógica*\n\n"
            f"📚 *Categorías disponibles:* {topics}\n\n"
            "💡 _Usá `/debate` para cambiar entre modo normal y modo debate._"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    @safe_handler
    async def h_ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        mode = "Local + Groq" if self.client else "Solo Local"
        await update.message.reply_text(
            f"Modo: {mode}\n"
            "• /pregunta <categoria> → pregunta local\n"
            "• Si decís 'no sé', te explico\n"
            "• /debate → alterna ON/OFF modo refutación lógica\n"
            "• /contexto  ·  /reset\n"
        )

    @safe_handler
    async def h_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        text = update.message.text.strip()
        await self._action(update, ChatAction.TYPING)

        # MODO DEBATE
        if self.debate_mode.get(cid, False):
            reply = await self.groq_debate(cid, text)
            await update.message.reply_text(f"🧠 {reply}")
            return

        reply = await self.groq_chat(cid, text)
        await update.message.reply_text(f"✅ {reply}")

    # ---------- /debate toggle ----------
    @safe_handler
    async def h_debate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        args = (context.args or [])

        if args:
            opt = args[0].strip().lower()
            if opt in ("on", "true", "1", "si", "sí"):
                self.debate_mode[cid] = True
            elif opt in ("off", "false", "0", "no"):
                self.debate_mode[cid] = False
            else:
                self.debate_mode[cid] = not self.debate_mode.get(cid, False)
        else:
            self.debate_mode[cid] = not self.debate_mode.get(cid, False)

        st = "ON" if self.debate_mode[cid] else "OFF"
        emoji = "🧠" if self.debate_mode[cid] else "💬"
        await update.message.reply_text(f"{emoji} Debate {st}: modo {'refutación' if st=='ON' else 'normal'} activado.")

    @safe_handler
    async def h_contexto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        hist = list(self.history.get(cid, []))
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
        await update.message.reply_text("📚 Contexto:\n" + "\n".join(preview))

    @safe_handler
    async def h_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        self._reset_history(cid)
        await update.message.reply_text("🧹 Memoria borrada.")

    async def _action(self, update: Update, action: ChatAction):
        try:
            if update and update.message:
                await update.message.chat.send_action(action)
        except Exception:
            pass

    # ---------- ARRANQUE ----------
    def _check_token(self):
        r = requests.get(f"https://api.telegram.org/bot{self.tg_token}/getMe", timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Token inválido: {data}")
        log.info(f"Token OK para @{data['result']['username']}")

    def build_app(self):
        self._check_token()
        app = ApplicationBuilder().token(self.tg_token).build()

        app.add_handler(CommandHandler("start", self.h_start))
        app.add_handler(CommandHandler("ayuda", self.h_ayuda))
        app.add_handler(CommandHandler("contexto", self.h_contexto))
        app.add_handler(CommandHandler("reset", self.h_reset))
        app.add_handler(CommandHandler("debate", self.h_debate))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.h_text))
        return app

    def run(self):
        log.info("Iniciando bot...")
        app = self.build_app()
        app.run_polling(allowed_updates=None, drop_pending_updates=True)


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    TelegramGroqBot(BASE).run()
