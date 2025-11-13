# 🤖 CapstoneProyectTrivIA — Telegram + Groq AI

Proyecto desarrollado en **Python (POO)** que implementa un **bot inteligente de Telegram**, potenciado por la **API de Groq**, capaz de:

- 💬 Mantener **memoria contextual** en las conversaciones.  
- 🧠 Analizar **texto, imágenes y notas de voz**.  
- 📘 Gestionar **preguntas desde un archivo JSON**, con autoevaluación y feedback.  
- 🤓 Responder automáticamente cuando el usuario dice “no sé”.  
- ⚔️ Activar un **Modo Debate lógico** que refuta ideas con razonamiento.  
- 🤝 Proporcionar una experiencia fluida y segura con manejo de errores controlado (`@safe_handler`).

---

## ⚙️ Requisitos

- Python **3.10 o superior**  
- Token del bot de **Telegram**  
- API Key de **Groq**  
- Librerías definidas en `requirements.txt`

---

## 🧩 Instalación y Configuración

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/MatiRAR/CapstoneProyectTriviA.git
cd CapstoneProyectTriviA/ai_telegram_groq


# Configurar variables de entorno
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui
GROQ_API_KEY=tu_api_key_de_groq_aqui


#  Ejecucion del bot 
python bot.py


# Si todo está correcto verás
INFO - Bot OK: @Tribgo_bot (id=...)
INFO - Iniciando polling… (si queda aquí, ya está escuchando)


# Comandos disponibles
| Comando     | Descripción                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `/start`    | Muestra un mensaje de bienvenida con las categorías disponibles.       |
| `/ayuda`    | Explica las funciones y modos del bot.                                 |
| `/pregunta` | Envía una pregunta aleatoria del JSON (o según la categoría indicada). |
| `/contexto` | Muestra la memoria reciente de la conversación.                        |
| `/reset`    | Limpia toda la memoria del chat.                                       |
| `/debate`   | Alterna el modo debate (ON/OFF): refuta ideas con lógica y argumentos. |
