# bot.py — POO con memoria + UX/acciones + handlers seguros
# Prioriza respuestas locales (JSON): auto-responde "no sé", evalúa y corrige la respuesta del usuario.
# Usa Groq SOLO como fallback si no hay respuesta local o si la confianza es baja.

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
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ---------- AJUSTES ----------
USE_GROQ_FALLBACK = True          # Si no hay respuesta local, o confianza baja, usa Groq como respaldo
LOW_CONFIDENCE_THRESH = 0.45      # Umbral de similitud para decidir si pedir ayuda a Groq
MAX_HISTORY_TURNS = 6             # Memoria corta por chat

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tribgo.localfirst")

# ---------- FIX EVENT LOOP EN WINDOWS ----------
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---------- DECORADOR: HANDLERS SEGUROS ----------
def safe_handler(fn):
    @wraps(fn)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await fn(self, update, context, *args, **kwargs)
        except Exception:
            log.exception("Error en handler %s", fn.__name__)
            try:
                if update and update.message:
                    await update.message.reply_text("⚠️ Ocurrió un error. Intentá de nuevo.")
            except Exception:
                pass
    return wrapper


class TelegramGroqBot:
    """Bot orientado a objetos con prioridad local (JSON) y fallback opcional a Groq."""

    def __init__(self, base: Path):
        self.base = base

        # ---- .env, config y preguntas ----
        load_dotenv(self.base / ".env")
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.tg_token:
            raise SystemExit("Falta TELEGRAM_BOT_TOKEN en .env")

        # config.json opcional (solo si querés usar Groq fallback)
        self.cfg = {"models": {"chat": "llama-3.3-70b-versatile", "vision": "meta-llama/llama-4-scout-17b-16e-instruct", "stt": "whisper-large-v3-turbo"}}
        cfg_path = self.base / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.cfg = json.load(f)
            except Exception:
                log.warning("No pude leer config.json; uso defaults internos.")

        with open(self.base / "preguntas.json", "r", encoding="utf-8") as f:
            self.preg = json.load(f)

        # ---- Cliente Groq (solo si hay API key y se permite fallback) ----
        self.client = AsyncGroq(api_key=self.groq_key) if (USE_GROQ_FALLBACK and self.groq_key) else None
        if USE_GROQ_FALLBACK and not self.client:
            log.warning("Groq fallback habilitado pero falta GROQ_API_KEY; seguiré solo con respuestas locales.")

        # ---- Memoria de conversación ----
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS * 2))
        self.system_prompt = "Eres un asistente técnico, claro y amable en español."

        # ---- Estado de tutoría ----
        self.last_question: dict[int, str] = {}
        self.awaiting_answer: dict[int, bool] = defaultdict(bool)

        # ---- Índice local de Q/A a partir del JSON ----
        # Mapa: pregunta_normalizada -> {"q":..., "a":..., "keywords":[...]}
        self.qa_index = self._build_local_index(self.preg)

        # Stopwords mínimas para similitud
        self.stop_es = {
            "el","la","los","las","un","una","unos","unas","de","del","al","y","o","u","es","son",
            "en","por","para","con","sin","a","que","se","lo","su","sus","mi","mis","tu","tus",
            "yo","vos","usted","ustedes","él","ella","ellos","ellas","nosotros","nosotras","me",
            "te","le","les","nos","como","sobre","entre","hasta","desde","ya","muy","mas","más",
            "si","sí","no","tambien","también","pero","porque","qué","que"
        }

    # ---------- UTILITARIOS ----------
    @staticmethod
    def _normalize(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"[^\wáéíóúñü\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _tokens(self, s: str) -> list[str]:
        norm = self._normalize(s)
        toks = [t for t in norm.split() if t not in self.stop_es]
        return toks

    @staticmethod
    def _img_to_b64(pil_img: Image.Image, fmt="JPEG") -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _all_categories(self):
        return [t["id"] for t in self.preg.get("topics", []) if "id" in t]

    def _pick_question(self, category: str | None):
        cats = {}
        for topic in self.preg.get("topics", []):
            tid = topic.get("id")
            qs = topic.get("preguntas", [])
            if not tid: 
                continue
            # Soportar strings o objetos con {q,a,keywords}
            cat_qs = []
            for item in qs:
                if isinstance(item, str):
                    cat_qs.append(item)
                elif isinstance(item, dict) and "q" in item:
                    cat_qs.append(item["q"])
            if cat_qs:
                cats[tid] = cat_qs

        if not cats:
            return None, None
        if not category or category not in cats:
            category = random.choice(list(cats.keys()))
        qs = cats.get(category, [])
        return category, (random.choice(qs) if qs else None)

    def _build_local_index(self, preg_json: dict) -> dict[str, dict]:
        """Construye un índice de preguntas → {q,a,keywords} para respuestas locales."""
        index = {}
        for topic in preg_json.get("topics", []):
            for item in topic.get("preguntas", []):
                if isinstance(item, str):
                    q = item
                    index[self._normalize(q)] = {"q": q, "a": None, "keywords": []}
                elif isinstance(item, dict) and "q" in item:
                    q = item["q"]
                    a = item.get("a")
                    kw = item.get("keywords", [])
                    index[self._normalize(q)] = {"q": q, "a": a, "keywords": kw}
        return index

    def _find_local_entry(self, question: str) -> dict | None:
        """Busca la pregunta en el índice local (normalizada)."""
        key = self._normalize(question)
        if key in self.qa_index:
            return self.qa_index[key]
        # Fallback: similitud aproximada por ratio
        best_key, best_ratio = None, 0.0
        for k in self.qa_index.keys():
            r = SequenceMatcher(None, key, k).ratio()
            if r > best_ratio:
                best_key, best_ratio = k, r
        if best_key and best_ratio >= 0.75:
            return self.qa_index[best_key]
        return None

    def _local_answer(self, question: str) -> str | None:
        """Devuelve respuesta local si existe en el JSON enriquecido."""
        entry = self._find_local_entry(question)
        if entry and entry.get("a"):
            return entry["a"]
        return None

    def _local_grade(self, question: str, user_answer: str) -> tuple[str, float]:
        """Evalúa localmente (sin Groq) usando keywords/semántica simple. Devuelve (feedback, score[0-1])."""
        entry = self._find_local_entry(question)
        ua_toks = set(self._tokens(user_answer))
        if not entry:
            # Sin referencia local: evaluar por longitud/claridad mínima
            score = min(1.0, len(ua_toks) / 12.0)
            verdict = "Correcta" if score >= 0.7 else ("Parcial" if score >= 0.4 else "Incorrecta")
            fb = (
                f"Veredicto: {verdict}\n"
                f"Por qué: respuesta {'completa' if score>=0.7 else 'parcial' if score>=0.4 else 'muy breve o poco específica'}.\n"
                "Corrección: —\n"
                "Puntos clave: • Define conceptos • Da un ejemplo\n"
                f"Puntaje: {int(score*100)}"
            )
            return fb, score

        # Hay entrada local: usar keywords y/o respuesta modelo
        kw = set(self._tokens(" ".join(entry.get("keywords", []))))
        a_model = entry.get("a") or ""
        a_toks = set(self._tokens(a_model))

        # similitudes básicas
        j_kw = len(ua_toks & kw) / max(1, len(kw)) if kw else 0.0
        j_ans = len(ua_toks & a_toks) / max(1, len(a_toks)) if a_toks else 0.0
        seq = SequenceMatcher(None, " ".join(sorted(ua_toks)), " ".join(sorted(a_toks))).ratio() if a_toks else 0.0

        score = max(j_kw*0.6 + j_ans*0.3 + seq*0.1, 0.0)
        verdict = "Correcta" if score >= 0.75 else ("Parcial" if score >= 0.45 else "Incorrecta")

        puntos_clave = []
        if kw:
            faltantes = list(kw - ua_toks)
            if faltantes:
                puntos_clave.append("• Menciona: " + ", ".join(faltantes[:4]))
        if a_model and verdict != "Correcta":
            puntos_clave.append("• Aclara ideas principales de la definición")

        corr = a_model if a_model else "Ampliá con definición breve y un ejemplo."
        fb = (
            f"Veredicto: {verdict}\n"
            f"Por qué: coincidencia con conceptos esperados ~{int(score*100)}%.\n"
            f"Corrección o mejor respuesta: {corr}\n"
            f"Puntos clave:\n" + ("\n".join(puntos_clave) if puntos_clave else "• —") + "\n"
            f"Puntaje: {int(score*100)}"
        )
        return fb, score

    async def _action(self, update: Update, action: ChatAction):
        try:
            if update and update.message:
                await update.message.chat.send_action(action)
        except Exception:
            pass

    @staticmethod
    def _is_no_se(text: str) -> bool:
        t = (text or "").strip().lower()
        gatillos = {
            "no se", "no sé", "nose", "ni idea", "no lo sé", "no lo se",
            "no sabría", "no sabria", "no estoy seguro", "no estoy segura",
            "paso", "no puedo", "no recuerdo"
        }
        t = " ".join(t.split())
        return t in gatillos or t.startswith("no se") or t.startswith("no sé")

    @staticmethod
    def _is_command(text: str) -> bool:
        return (text or "").strip().startswith("/")

    # ---------- MEMORIA ----------
    def _remember(self, chat_id: int, role: str, content: str):
        self.history[chat_id].append({"role": role, "content": content})

    def _reset_history(self, chat_id: int):
        self.history.pop(chat_id, None)
        self.last_question.pop(chat_id, None)
        self.awaiting_answer.pop(chat_id, None)

    def _build_messages(self, chat_id: int, user_text: str) -> list[dict]:
        hist = list(self.history[chat_id])[-(MAX_HISTORY_TURNS * 2):]
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(hist)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    # ---------- GROQ (fallback opcional) ----------
    async def groq_chat(self, chat_id: int, user_text: str) -> str:
        if not self.client:
            # Sin cliente Groq: responder simple
            return "🤖 (Modo local) Puedo ayudarte con preguntas del JSON y evaluar tus respuestas."
        messages = self._build_messages(chat_id, user_text)
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["chat"],
            messages=messages,
            max_completion_tokens=512,
        )
        reply = chat.choices[0].message.content
        self._remember(chat_id, "user", user_text)
        self._remember(chat_id, "assistant", reply)
        return reply

    async def groq_vision(self, img: Image.Image, question: str) -> str:
        if not self.client:
            return "🖼️ (Modo local) Recibí la imagen. La descripción avanzada requiere Groq habilitado."
        b64 = self._img_to_b64(img, "JPEG")
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["vision"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
            }],
            max_completion_tokens=384,
        )
        return chat.choices[0].message.content

    async def groq_transcribe(self, file_bytes: bytes, filename: str, language: str = "es") -> str:
        if not self.client:
            return "(modo local) Transcripción no disponible sin Groq."
        tr = await self.client.audio.transcriptions.create(
            file=(filename, file_bytes),
            model=self.cfg["models"]["stt"],
            language=language,
            response_format="json",
            temperature=0.0,
        )
        return tr.text

    async def groq_one_off(self, prompt: str) -> str:
        """Llamado breve sin contaminar la historia del chat."""
        if not self.client:
            return "(modo local) Explicación completa requiere Groq."
        chat = await self.client.chat.completions.create(
            model=self.cfg["models"]["chat"],
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            max_completion_tokens=400,
        )
        return chat.choices[0].message.content

    # ---------- HANDLERS ----------
    @safe_handler
    async def h_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        topics = ", ".join(self._all_categories())
        await update.message.reply_text(
            "¡Hola! Enviame texto, una foto o una nota de voz.\n"
            "Comandos:\n"
            "  /ayuda\n"
            "  /pregunta <categoria>  (o sin categoría para aleatoria)\n"
            "  /contexto  ·  /reset\n"
            f"Categorías: {topics}"
        )

    @safe_handler
    async def h_ayuda(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        topics = ", ".join(self._all_categories())
        mode = "Local + Groq fallback" if self.client else "Solo Local"
        await update.message.reply_text(
            f"Modo: {mode}\n"
            "• Texto: respondo y mantengo contexto\n"
            "• Preguntas del JSON: priorizo respuestas locales; si decís «no sé», te explico\n"
            "• Evalúo tu respuesta y corrijo si hace falta\n"
            "• Imagen/Audio requieren Groq habilitado\n"
            f"Categorías: {topics}"
        )

    @safe_handler
    async def h_pregunta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args or []
        cat = (args[0].strip().lower() if args else None)
        sel_cat, q = self._pick_question(cat)
        if not q:
            await update.message.reply_text("No encontré preguntas. Revisá preguntas.json.")
            return

        chat_id = update.effective_chat.id
        self.last_question[chat_id] = q
        self.awaiting_answer[chat_id] = True

        await update.message.reply_text(
            f"🗂️ {sel_cat}\n💬 {q}\n\n"
            "Respondé con tu idea. Si no sabés, decí «no sé» y te ayudo."
        )

    @safe_handler
    async def h_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        user_text = update.message.text.strip()

        # Comandos → flujo normal (podés personalizar)
        if self._is_command(user_text):
            await self._action(update, ChatAction.TYPING)
            reply = await self.groq_chat(chat_id, user_text) if self.client else "Comando recibido."
            await update.message.reply_text(f"✅ {reply}")
            return

        # "no sé" → responder local si hay, si no fallback a Groq
        if self._is_no_se(user_text) and chat_id in self.last_question:
            pregunta = self.last_question[chat_id]
            await self._action(update, ChatAction.TYPING)
            local = self._local_answer(pregunta)
            if local:
                self.awaiting_answer[chat_id] = False
                await update.message.reply_text(f"🧠 Te ayudo con eso (local):\n{local}")
                return
            # sin respuesta local → Groq fallback
            tutor = (
                "Responde de forma clara y breve en español a la siguiente pregunta. "
                "Luego da un ejemplo corto si aplica.\n\n"
                f"Pregunta: {pregunta}"
            )
            reply = await self.groq_one_off(tutor) if self.client else "(modo local) Sin respuesta definida."
            self.awaiting_answer[chat_id] = False
            await update.message.reply_text(f"🧠 Te ayudo con eso:\n{reply}")
            return

        # Si estamos esperando respuesta a una pregunta → evaluar localmente primero
        if self.awaiting_answer.get(chat_id) and chat_id in self.last_question:
            pregunta = self.last_question[chat_id]
            await self._action(update, ChatAction.TYPING)
            feedback, score = self._local_grade(pregunta, user_text)

            # Si confianza baja y está disponible Groq, mejorar explicación
            extra = ""
            if score < LOW_CONFIDENCE_THRESH and USE_GROQ_FALLBACK and self.client:
                prompt = (
                    "Mejora esta explicación de forma breve y clara en español. "
                    "Si la respuesta del estudiante es incorrecta, corrige con una definición correcta y un ejemplo.\n\n"
                    f"Pregunta: {pregunta}\n"
                    f"Respuesta del estudiante: {user_text}\n"
                    f"Feedback local: {feedback}\n"
                )
                extra = "\n\n🔎 Mejora:\n" + await self.groq_one_off(prompt)

            await update.message.reply_text(
                f"🗣️ Tu respuesta: {user_text}\n\n📋 Feedback:\n{feedback}{extra}"
            )
            self.awaiting_answer[chat_id] = False
            return

        # Flujo normal de conversación (si querés, podés mantenerlo local con reglas)
        await self._action(update, ChatAction.TYPING)
        if self.client:
            reply = await self.groq_chat(chat_id, user_text)
        else:
            reply = "🤖 (Modo local) Decime una categoría con /pregunta <categoria> o pedime evaluar tu respuesta."
        await update.message.reply_text(f"✅ {reply}")

    @safe_handler
    async def h_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.photo:
            return
        chat_id = update.effective_chat.id

        await self._action(update, ChatAction.UPLOAD_PHOTO)
        ph = update.message.photo[-1]
        file = await context.bot.get_file(ph.file_id)

        bio = io.BytesIO()
        await file.download_to_memory(out=bio)
        bio.seek(0)

        img = Image.open(bio).convert("RGB")
        await self._action(update, ChatAction.TYPING)
        desc = await self.groq_vision(img, "Describe en español y agrega 3 etiquetas útiles.")

        user_note = "Imagen enviada" + (f" | {update.message.caption}" if update.message.caption else "")
        self._remember(chat_id, "user", user_note)
        self._remember(chat_id, "assistant", desc)

        await update.message.reply_text(f"🖼️✅ {desc}")

    @safe_handler
    async def h_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        voice = update.message.voice
        audio = update.message.audio
        doc = update.message.document
        chat_id = update.effective_chat.id

        file = None
        filename = "audio.ogg"

        if voice:
            await self._action(update, ChatAction.RECORD_VOICE)
            file = await context.bot.get_file(voice.file_id)
            filename = "voice.ogg"
        elif audio:
            await self._action(update, ChatAction.UPLOAD_AUDIO)
            file = await context.bot.get_file(audio.file_id)
            filename = audio.file_name or "audio.bin"
        elif doc and (doc.mime_type or "").startswith(("audio/", "video/")):
            await self._action(update, ChatAction.UPLOAD_DOCUMENT)
            file = await context.bot.get_file(doc.file_id)
            filename = doc.file_name or "audio.bin"
        else:
            await update.message.reply_text("Enviame una nota de voz o archivo de audio.")
            return

        bio = io.BytesIO()
        await file.download_to_memory(out=bio)
        bio.seek(0)

        await self._action(update, ChatAction.TYPING)
        text = await self.groq_transcribe(bio.getvalue(), filename=filename, language="es")
        reply = await self.groq_chat(chat_id, text) if self.client else "(modo local) Transcripción no disponible."
        await update.message.reply_text(f"📝 {text}\n\n🤖✅ {reply}")

    @safe_handler
    async def h_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        self._reset_history(chat_id)
        await update.message.reply_text("🧹 Memoria de conversación borrada.")

    @safe_handler
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

    # ---------- WIRING / ARRANQUE ----------
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
