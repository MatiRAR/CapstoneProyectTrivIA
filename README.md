# 🤖 CapstoneProyectTrivIA — Telegram + Groq AI

Proyecto desarrollado en **Python (POO)** que implementa un **bot inteligente de Telegram**, potenciado por la **API de Groq**, orientado a resolver problemas de la vida diaria y apoyar el estudio.

El bot es capaz de:

- 💬 Mantener **memoria contextual** en las conversaciones.
- 🧠 Activar un **Modo Debate lógico** para refutar ideas con argumentos y detectar fallas en el razonamiento.
- ⏰ Crear **recordatorios reales** con fecha y hora usando `/recordar`.
- 💸 Registrar y listar **gastos personales por chat** (`/gasto`, `/gastos`).
- 📄 Explicar **textos difíciles** (técnicos/legales) en lenguaje sencillo con `/doc`.
- 📘 Gestionar **preguntas desde un archivo JSON**, por categoría, para practicar (/pregunta).
- 🧹 Limpiar memoria, recordatorios y gastos con `/reset`.
- 🔒 Manejar errores de forma segura mediante un decorador `@safe_handler`.

> 🔁 El bot responde **siempre en español**, excepto si el usuario pide explícitamente otro idioma.

---

## ⚙️ Requisitos

- Python **3.10 o superior**  
- Token del bot de **Telegram**  
- API Key de **Groq** (opcional pero recomendada)  
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
| Comando        | Descripción                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `/start`       | Muestra el panel de bienvenida con todos los comandos y categorías disponibles.                                           |
| `/ayuda`       | Explica de forma resumida las funciones del bot y cómo usarlas.                                                           |
| `/contexto`    | Muestra el historial reciente de mensajes (memoria del chat).                                                             |
| `/reset`       | Limpia memoria, recordatorios y gastos del chat, y vuelve a mostrar el panel de comandos.                                 |
| `/debate`      | Alterna el modo debate (ON/OFF). En modo debate el bot refuta tus ideas con lógica y pide evidencia.                      |
| `/pregunta`    | Envía una pregunta aleatoria del JSON. Podés usar `/pregunta <categoria>` para filtrar (ej: `/pregunta estudio`).         |
| `/recordar`    | Inicia un flujo guiado para crear un **recordatorio real** con texto y horario.                                           |
| `/gasto`       | Inicia un flujo guiado para registrar un gasto: primero pide el monto y luego la categoría.                               |
| `/gastos`      | Muestra todos los gastos registrados en ese chat y el total acumulado.                                                    |
| `/doc <texto>` | Explica un texto difícil (técnico, académico o legal) en lenguaje simple y en español, resaltando los puntos importantes. |
