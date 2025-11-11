# 🤖 CapstoneProyectTriviA – Bot de Inteligencia Artificial (Telegram + Groq)

Este proyecto implementa un **bot de Telegram con inteligencia artificial**, desarrollado en **Python (POO)** e integrado con la **API de Groq**.  
El bot puede analizar texto, imágenes y audio, mantener el contexto de conversación y realizar análisis de sentimiento.

---

## ⚙️ Instalación y configuración completa

### 1️⃣ Clonar el repositorio
```bash
git clone https://github.com/MatiRAR/CapstoneProyectTriviA.git
cd CapstoneProyectTriviA/ai_telegram_groq

# Instalar dependencias
pip install -r requirements.txt


# Configurar variables de entorno
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui
GROQ_API_KEY=tu_api_key_de_groq_aqui


#  Ejecucion del bot 
python bot.py


# Si todo está correcto verás
INFO - Bot OK: @Tribgo_bot (id=...)
INFO - Iniciando polling… (si queda aquí, ya está escuchando)


# Comandos disponibles
| Comando     | Descripción                                                    |
| ----------- | -------------------------------------------------------------- |
| `/start`    | Muestra un mensaje de bienvenida y las categorías disponibles. |
| `/ayuda`    | Explica las funciones y el uso del bot.                        |
| `/pregunta` | Envía una pregunta aleatoria (o de una categoría específica).  |
| `/contexto` | Muestra la memoria reciente del chat.                          |
| `/reset`    | Limpia la memoria de conversación.                             | 