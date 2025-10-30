# CapstoneProyectTrivIA

# 🤖 DebateSensei — Telegram AI Debate Bot

**DebateSensei** es un bot de Telegram impulsado por IA que ayuda a desarrollar pensamiento crítico y fomentar debates respetuosos.

El usuario envía:
- ✅ Texto
- ✅ Imágenes (memes, noticias, frases, screenshots)

El bot analiza el contenido, detecta la postura y genera un **contraargumento respetuoso**, promoviendo una conversación constructiva y reflexiva.

> No busca ganar discusiones — busca ayudarte a pensar mejor.

---

## 🎯 Objetivo

Promover el pensamiento crítico y la conversación sana usando IA para analizar ideas, detectar posturas y ofrecer puntos de vista alternativos.

---

## ✨ Funcionalidades principales

| Función | Descripción |
|--------|-------------|
🧠 NLP | Detección de postura, tema y tono |
🖼️ OCR | Extrae texto desde imágenes (memes, noticias, carteles) |
⚖️ Contraargumentos | Respuestas razonadas y respetuosas |
🔍 Fact-check | Verificación rápida cuando aplica |
💬 Preguntas reflexivas | Estimula pensamiento crítico |
👍 Feedback | El usuario puede evaluar la respuesta |

---

## 🧩 Flujo general

## 📂 Estructura inicial del proyecto

```bash
DebateSensei/
│── src/
│   ├── bot.py
│   ├── ocr_service.py
│   ├── stt_service.py
│   ├── text_processor.py
│   ├── debate_engine.py
│   ├── fact_checker.py
│   ├── feedback.py
│   └── utils.py
│
├── tests/
│
├── README.md
└── requirements.txt
