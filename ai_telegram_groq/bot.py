# bot.py — Bot general + debate + recordatorios reales + gastos + explicador de texto
# Responde SIEMPRE en ESPAÑOL

import os
import json
import platform
import asyncio
import logging
import requests
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from functools import wraps

from dotenv import load_dotenv
from groq import AsyncGroq

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------- CONFIG ----------
USE_GROQ = True
MAX_HISTORY_TURNS = 6

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("tribot")

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
    """Bot general con memoria, modo debate y mini-módulos:
    - Recordatorios reales (/recordar)
    - Gastos (/gasto, /gastos)
    - Explicador de texto (/doc)
    - Preguntas por categoría (/pregunta <categoría>)
    """

    def __init__(self, base: Path):
        self.base = base

        # ---- .env ----
        load_dotenv(self.base / ".env")
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.groq_key = os.getenv("GROQ_API_KEY")

        if not self.tg_token:
            raise SystemExit("❌ Falta TELEGRAM_BOT_TOKEN en .env")

        # ---- preguntas.json (para /pregunta y categorías) ----
        self.preg = {}
        try:
            with open(self.base / "preguntas.json", "r", encoding="utf-8") as f:
                self.preg = json.load(f)
        except Exception:
            log.warning("No pude leer preguntas.json (afecta solo a /pregunta y listado de categorías).")

        # ---- Cliente Groq ----
        self.client = AsyncGroq(api_key=self.groq_key) if (USE_GROQ and self.groq_key) else None
        if not self.client:
            log.warning("Groq no configurado: las respuestas serán limitadas.")

        self.model_chat = "llama-3.3-70b-versatile"

        # ---- Estado de conversación ----
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))
        self.debate_mode: dict[int, bool] = defaultdict(lambda: False)

        # ---- Módulos ----
        self.reminders: dict[int, list[tuple[datetime, str]]] = defaultdict(list)   # (datetime, texto)
        self.expenses: dict[int, list[tuple[float, str]]] = defaultdict(list)       # (monto, categoría)

        # Estado de diálogo para /recordar (por chat)
        # ej: { chat_id: {"step": "waiting_text"/"waiting_time", "text": "..." } }
        self.reminder_state: dict[int, dict] = {}

    # ---------- UTILIDADES BÁSICAS ----------
    async def _action(self, update: Update, action: ChatAction):
        try:
            if update and update.message:
                await update.message.chat.send_action(action)
        except Exception:
            pass

    def _remember(self, cid: int, role: str, content: str):
        self.history[cid].append({"role": role, "content": content})

    # ---------- PREGUNTAS JSON ----------
    def _all_categories(self) -> list[str]:
        try:
            return [t["id"] for t in self.preg.get("topics", []) if "id" in t]
        except Exception:
            return []

    def _pick_question(self, category: str | None):
        cats: dict[str, list[str]] = {}
        for topic in self.preg.get("topics", []):
            tid = topic.get("id")
            qs = topic.get("preguntas", [])
            if not tid:
                continue
            preguntas_cat = []
            for item in qs:
                if isinstance(item, str):
                    preguntas_cat.append(item)
                elif isinstance(item, dict) and "q" in item:
                    preguntas_cat.append(item["q"])
            if preguntas_cat:
                cats[tid] = preguntas_cat

        if not cats:
            return None, None

        if not category or category not in cats:
            category = random.choice(list(cats.keys()))
        qs_cat = cats.get(category, [])
        return category, (random.choice(qs_cat) if qs_cat else None)

    def _topics_str(self) -> str:
        cats = self._all_categories()
        return ", ".join(cats) if cats else "—"

    # ---------- GROQ ----------
    async def groq_chat(self, cid: int, text: str) -> str:
        """Chat normal: SIEMPRE en español."""
        if not self.client:
            return "🤖 (modo local) Sin conexión a Groq. Puedo ayudarte con /recordar, /gasto, /gastos, /doc o /pregunta."

        chat = await self.client.chat.completions.create(
            model=self.model_chat,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente útil que responde SIEMPRE en español. "
                        "Sé claro, concreto y amable. No uses inglés a menos que el usuario lo pida explícitamente."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_completion_tokens=512,
            temperature=0.4,
        )
        reply = chat.choices[0].message.content
        self._remember(cid, "user", text)
        self._remember(cid, "assistant", reply)
        return reply

    async def groq_debate(self, cid: int, text: str) -> str:
        """Modo debate: refuta en español."""
        if not self.client:
            return "🧩 (modo local) El modo debate requiere GROQ_API_KEY configurada."

        chat = await self.client.chat.completions.create(
            model=self.model_chat,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Actúas como un crítico lógico en ESPAÑOL. "
                        "Tu tarea es analizar y refutar de forma respetuosa la afirmación del usuario. "
                        "Detecta posibles falacias, pide evidencia, ofrece contraejemplos y termina con una síntesis breve. "
                        "Nunca respondas en inglés salvo que el usuario lo pida explícitamente."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_completion_tokens=512,
            temperature=0.25,
        )
        reply = chat.choices[0].message.content
        self._remember(cid, "user", text)
        self._remember(cid, "assistant", reply)
        return reply

    # ---------- PANEL /START ----------
    async def _panel_html(self, update: Update):
        """Panel con listado de comandos (formato HTML)."""
        topics_str = self._topics_str()
        msg = (
            "👋 <b>¡Hola! Soy tu asistente interactivo.</b>\n"
            "Podés escribirme directamente o usar comandos.\n\n"
            "📋 <b>Comandos principales:</b>\n"
            "• /ayuda — Ver funciones y modos\n"
            "• /contexto — Ver historial reciente\n"
            "• /reset — Borrar memoria\n"
            "• /debate — Activar/desactivar refutación lógica\n\n"
            "🗓️ <b>Organización personal:</b>\n"
            "• /recordar — Crear un recordatorio guiado\n"
            "• /gasto &lt;monto&gt; &lt;categoría&gt; — Registrar un gasto\n"
            "• /gastos — Ver el resumen de gastos del chat\n"
            "• /doc &lt;texto&gt; — Explicar un texto en lenguaje sencillo\n\n"
            f"❓ <b>Preguntas por categoría:</b>\n"
            f"• /pregunta &lt;categoría&gt; — Ej: /pregunta estudio\n"
            f"   Categorías disponibles: {topics_str}\n\n"
            "💡 Usá <b>/debate</b> para cambiar entre modo normal y modo debate."
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    # ---------- PARSER DE FECHA/HORA PARA RECORDATORIOS ----------
    def _parse_reminder_time(self, text: str) -> tuple[datetime | None, str | None]:
        """
        Acepta formas simples:
          - 'HH:MM'
          - 'hoy HH:MM'
          - 'mañana HH:MM'
        Devuelve (datetime, None) o (None, mensaje_error)
        """
        s = text.lower().strip()
        s = s.replace("hs", "").replace("hs.", "").replace(" horas", "").strip()
        s = s.replace("a las", "").strip()

        now = datetime.now()

        # mañana HH:MM
        if s.startswith("mañana"):
            resto = s.replace("mañana", "", 1).strip()
            m = re.match(r"^(\d{1,2}):(\d{2})$", resto)
            if not m:
                return None, "No entendí la hora. Usá algo como 'mañana 20:30'."
            h, mi = int(m.group(1)), int(m.group(2))
            dt = (now + timedelta(days=1)).replace(hour=h, minute=mi, second=0, microsecond=0)
            return dt, None

        # hoy HH:MM
        if s.startswith("hoy"):
            resto = s.replace("hoy", "", 1).strip()
            m = re.match(r"^(\d{1,2}):(\d{2})$", resto)
            if not m:
                return None, "No entendí la hora. Usá algo como 'hoy 20:30'."
            h, mi = int(m.group(1)), int(m.group(2))
            dt = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            if dt <= now:
                dt = dt + timedelta(days=1)
            return dt, None

        # Solo HH:MM → hoy (o mañana si ya pasó)
        m = re.match(r"^(\d{1,2}):(\d{2})$", s)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            dt = now.replace(hour=h, minute=mi, second=0, microsecond=0)
            if dt <= now:
                dt = dt + timedelta(days=1)
            return dt, None

        return None, "Formato no reconocido. Usá por ejemplo: '20:30', 'hoy 21:00' o 'mañana 09:15'."

    async def _schedule_reminder(self, cid: int, texto: str, when_dt: datetime,
                                 update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Programa un recordatorio real usando asyncio."""
        now = datetime.now()
        delay = max(0, (when_dt - now).total_seconds())

        # Guardar en memoria interna
        self.reminders[cid].append((when_dt, texto))

        # Aviso al usuario
        nice = when_dt.strftime("%d/%m %H:%M")
        await update.message.reply_text(f"⏰ Listo, te voy a recordar esto el {nice}:\n• {texto}")

        async def task():
            try:
                await asyncio.sleep(delay)
                await context.bot.send_message(
                    chat_id=cid,
                    text=f"🔔 Recordatorio:\n• {texto}\n({nice})"
                )
            except Exception:
                log.exception("Error enviando recordatorio programado")

        context.application.create_task(task())

    # ---------- HANDLERS ----------
    @safe_handler
    async def h_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._panel_html(update)

    @safe_handler
    async def h_ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        topics_str = self._topics_str()
        await update.message.reply_text(
            "📘 *Funciones disponibles:*\n\n"
            "🤖 Conversación general (siempre en español).\n"
            "🧠 `/debate` — alterna modo refutación lógica.\n"
            "🧹 `/reset` — limpia memoria, recordatorios y gastos.\n"
            "📚 `/contexto` — muestra el historial reciente.\n\n"
            "⏰ `/recordar` — inicia un diálogo para crear un recordatorio real.\n"
            "💸 `/gasto <monto> <categoría>` — registra un gasto (ej: `/gasto 1200 comida`).\n"
            "💰 `/gastos` — muestra el listado y total de gastos.\n"
            "📄 `/doc <texto>` — explica un texto difícil en lenguaje sencillo.\n"
            f"❓ `/pregunta <categoría>` — muestra una pregunta del JSON. Categorías: {topics_str}\n",
            parse_mode="Markdown",
        )

    @safe_handler
    async def h_pregunta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Devuelve una pregunta del JSON según categoría."""
        args = context.args or []
        cat = args[0].strip().lower() if args else None
        sel_cat, q = self._pick_question(cat)
        if not q:
            await update.message.reply_text("No encontré preguntas. Revisá preguntas.json.")
            return
        await update.message.reply_text(f"🗂️ Categoría: {sel_cat}\n💬 {q}")

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
        await update.message.reply_text("📚 Contexto reciente:\n" + "\n".join(preview))

    @safe_handler
    async def h_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        self.history.pop(cid, None)
        self.reminders.pop(cid, None)
        self.expenses.pop(cid, None)
        self.reminder_state.pop(cid, None)
        await update.message.reply_text("🧹 Memoria borrada. Te vuelvo a mostrar los comandos disponibles:")
        await self._panel_html(update)

    @safe_handler
    async def h_debate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        self.debate_mode[cid] = not self.debate_mode.get(cid, False)
        st = "ON" if self.debate_mode[cid] else "OFF"
        emoji = "🧠" if self.debate_mode[cid] else "💬"
        await update.message.reply_text(f"{emoji} Modo debate: {st}")

    # ----- Recordatorios (modo diálogo) -----
    @safe_handler
    async def h_recordar(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Inicia o acelera el flujo de creación de recordatorios."""
        cid = update.effective_chat.id
        rest = update.message.text.replace("/recordar", "", 1).strip()

        # Si viene texto junto, lo tomamos como contenido y pedimos hora
        if rest:
            self.reminder_state[cid] = {"step": "waiting_time", "text": rest}
            await update.message.reply_text(
                "⏰ ¿Para cuándo querés el recordatorio?\n"
                "Ejemplos: `20:30`, `hoy 21:00`, `mañana 09:15`",
                parse_mode="Markdown",
            )
            return

        # Si viene solo /recordar desde el botón azul
        self.reminder_state[cid] = {"step": "waiting_text"}
        await update.message.reply_text(
            "⏰ ¿Qué querés recordar?\n"
            "Ejemplo: `estudiar para el parcial`, `llevar documentos`"
        )

    # ----- Gastos -----
    @safe_handler
    async def h_gasto(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        rest = update.message.text.replace("/gasto", "", 1).strip()
        parts = rest.split()
        if len(parts) < 2:
            await update.message.reply_text(
                "💸 Uso: `/gasto <monto> <categoría>`\nEj: `/gasto 1200 comida`",
                parse_mode="Markdown",
            )
            return
        try:
            monto = float(parts[0].replace(",", "."))
        except ValueError:
            await update.message.reply_text(
                "💸 El monto debe ser un número. Ej: `1500` o `1500.50`.",
                parse_mode="Markdown",
            )
            return
        categoria = " ".join(parts[1:])
        self.expenses[cid].append((monto, categoria))
        await update.message.reply_text(f"💰 Gasto registrado: {monto} — {categoria}")

    @safe_handler
    async def h_gastos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        items = self.expenses.get(cid, [])
        if not items:
            await update.message.reply_text("No hay gastos registrados en este chat.")
            return
        total = sum(m for m, _ in items)
        lineas = [f"• {m} — {c}" for m, c in items]
        texto = "\n".join(lineas)
        await update.message.reply_text(
            f"💸 <b>Gastos registrados:</b>\n{texto}\n\n<b>Total:</b> {total}",
            parse_mode="HTML",
        )

    # ----- Explicador de texto (/doc) -----
    @safe_handler
    async def h_doc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        texto = update.message.text.replace("/doc", "", 1).strip()
        if not texto:
            await update.message.reply_text(
                "📄 Uso: `/doc <texto que no entiendas>`\n"
                "Ej: `/doc El presente contrato establece que...`",
                parse_mode="Markdown",
            )
            return

        await self._action(update, ChatAction.TYPING)
        if not self.client:
            await update.message.reply_text("📄 (modo local) Sin Groq, no puedo explicar el documento.")
            return

        chat = await self.client.chat.completions.create(
            model=self.model_chat,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Explica el siguiente texto legal/técnico en ESPAÑOL sencillo. "
                        "Usa viñetas si hace falta y resalta los puntos importantes."
                    ),
                },
                {"role": "user", "content": texto},
            ],
            max_completion_tokens=512,
            temperature=0.3,
        )
        reply = chat.choices[0].message.content
        self._remember(cid, "user", texto)
        self._remember(cid, "assistant", reply)
        await update.message.reply_text(f"📄 {reply}")

    # ----- Texto normal (sin comando) -----
    @safe_handler
    async def h_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cid = update.effective_chat.id
        text = (update.message.text or "").strip()

        # 1) Flujo de recordatorio en curso
        if cid in self.reminder_state:
            state = self.reminder_state[cid]
            step = state.get("step")

            if step == "waiting_text":
                # Guardamos el texto y pedimos la hora
                state["text"] = text
                state["step"] = "waiting_time"
                self.reminder_state[cid] = state
                await update.message.reply_text(
                    "⏰ Perfecto. ¿Para cuándo querés el recordatorio?\n"
                    "Ejemplos: `20:30`, `hoy 21:00`, `mañana 09:15`"
                )
                return

            if step == "waiting_time":
                texto = state.get("text", "(sin detalle)")
                when_dt, err = self._parse_reminder_time(text)
                if not when_dt:
                    await update.message.reply_text(f"⚠️ {err}")
                    return
                # cerramos estado
                self.reminder_state.pop(cid, None)
                await self._schedule_reminder(cid, texto, when_dt, update, context)
                return

        # 2) Saludos → muestran panel y NO llaman al modelo
        lower = text.lower()
        saludos = {"hola", "holaa", "holis", "buenas", "buen día", "buen dia", "hello", "hi"}
        if lower in saludos:
            await self._panel_html(update)
            return

        # 3) Conversación normal / modo debate
        await self._action(update, ChatAction.TYPING)

        if self.debate_mode.get(cid, False):
            reply = await self.groq_debate(cid, text)
            emoji = "🧠"
        else:
            reply = await self.groq_chat(cid, text)
            emoji = "💬"

        await update.message.reply_text(f"{emoji} {reply}")

    # ---------- ARRANQUE ----------
    def _check_token(self):
        r = requests.get(f"https://api.telegram.org/bot{self.tg_token}/getMe", timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Token inválido: {data}")
        log.info("✅ Bot OK: @%s", data["result"]["username"])

    def build_app(self):
        self._check_token()
        app = ApplicationBuilder().token(self.tg_token).build()

        app.add_handler(CommandHandler("start", self.h_start))
        app.add_handler(CommandHandler("ayuda", self.h_ayuda))
        app.add_handler(CommandHandler("contexto", self.h_contexto))
        app.add_handler(CommandHandler("reset", self.h_reset))
        app.add_handler(CommandHandler("debate", self.h_debate))
        app.add_handler(CommandHandler("pregunta", self.h_pregunta))

        # nuevos comandos
        app.add_handler(CommandHandler("recordar", self.h_recordar))
        app.add_handler(CommandHandler("gasto", self.h_gasto))
        app.add_handler(CommandHandler("gastos", self.h_gastos))
        app.add_handler(CommandHandler("doc", self.h_doc))

        # texto normal
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.h_text))

        return app

    def run(self):
        log.info("🚀 Iniciando bot…")
        app = self.build_app()
        app.run_polling(allowed_updates=None, drop_pending_updates=True)


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    TelegramGroqBot(BASE).run()
