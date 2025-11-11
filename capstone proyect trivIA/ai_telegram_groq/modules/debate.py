# debate.py
from collections import deque
from groq import AsyncGroq


def is_information_question(text: str) -> bool:
    interrogatives = [
        "qué", "como", "cuándo", "dónde", "quién",
        "por qué", "sabes", "podés explicarme", "me explicas"
    ]
    text = text.lower()
    return any(q in text for q in interrogatives) and "?" in text

def is_greeting(text: str) -> bool:
    greetings = ["hola", "holaa", "hey", "buenas", "eo", "ola"]
    return text.lower().strip() in greetings



class ConversationMemory:
    def __init__(self, max_turns=5):
        self.memory = deque(maxlen=max_turns)

    def add(self, role: str, content: str):
        self.memory.append({"role": role, "content": content})

    def get(self):
        return list(self.memory)


def build_debate_prompt(user_text: str, sentiment: str | None = None) -> str:
    tone = "calmado y empático" if sentiment == "negativo" else "conversacional pero firme"
    return f"""
Actúas como DebateSensei, un asistente orientado al pensamiento crítico y al debate constructivo.

Objetivo:
- No solo entender la postura del usuario, sino **desafiarla con argumentos sólidos**.
- Presenta **ejemplos, datos, o escenarios reales** para apoyar tu contraargumento.
- Evita sonar neutral o indeciso: **toma una posición clara**, pero sin agresividad.

Guía de respuesta:
1) Resume la postura del usuario con precisión.
2) Presenta una **posición contraria bien desarrollada**, explicando su lógica.
3) Incluye **al menos un ejemplo o consecuencia práctica**.
4) Termina con **una pregunta que confronte directamente** la suposición del usuario.

Tono:
- {tone}
- Natural, directo, sin frases de relleno.
- No pidas más contexto. No digas “parece que”.

Texto del usuario:
\"\"\"{user_text}\"\"\"
"""



async def debate_reply(client: AsyncGroq, model: str, text: str, memory: ConversationMemory, analyze_sentiment=None):

    # 1) Si solo saludan → responder simple + preguntar tema
    if is_greeting(text):
        reply = "Hola! ¿Sobre qué tema te gustaría debatir hoy?"
        memory.add("user", text)
        memory.add("assistant", reply)
        return reply

    # 2) Si es pregunta informativa → modo respuesta normal
    if is_information_question(text):

        prompt = f"""
Responde claramente, sin sonar robótico, como si estuvieras hablando con alguien en persona.
Explica lo necesario, pero no exageres.
Texto:
\"\"\"{text}\"\"\"
"""
        chat = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Respondes preguntas de forma clara y directa, con tono natural."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=500
        )
        reply = chat.choices[0].message.content
        memory.add("user", text)
        memory.add("assistant", reply)
        return reply

    # Modo debate
    sentiment = analyze_sentiment(text) if analyze_sentiment else None
    debate_prompt = build_debate_prompt(text, sentiment)

    full_memory_context = memory.get()
    messages = [{"role": "system", "content": "Eres un facilitador de conversación reflexiva."}] + full_memory_context + [
        {"role": "user", "content": debate_prompt}
    ]

    chat = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=500
    )

    reply = chat.choices[0].message.content

    memory.add("user", text)
    memory.add("assistant", reply)

    return reply
