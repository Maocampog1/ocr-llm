# Taller IA: OCR + LLM (solo GROQ como lo dijo el profe)


import os
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import easyocr


# Configuración de la página

st.set_page_config(page_title="Taller IA: OCR + LLM", page_icon=None, layout="centered")
st.title("Taller IA: OCR + LLM")
st.caption("OCR con EasyOCR y análisis con GROQ. Todo en español, excepto la tarea de traducir al inglés.")

# Clave GROQ cargada desde .env

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    st.error("Falta la clave GROQ_API_KEY en tu archivo .env.")
    st.stop()


# Estado de sesión

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""
if "llm_output" not in st.session_state:
    st.session_state["llm_output"] = ""


# OCR con EasyOCR 

@st.cache_resource
def cargar_lector():
    return easyocr.Reader(["es", "en"])

lector = cargar_lector()

def ocr_con_tu_metodo(archivo_imagen, min_conf: float) -> str:
    # Mostrar imagen
    try:
        st.image(Image.open(archivo_imagen), caption="Imagen cargada", use_column_width=True)
    except Exception:
        pass

    # Leer imagen con OpenCV desde el archivo subido por el usuario 
    bytes_data = archivo_imagen.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError("No se pudo leer la imagen.")

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

   
    resultados = lector.readtext(img_rgb, detail=1)  
    lineas = []
    for r in resultados:
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            _, txt, conf = r
            if isinstance(txt, str) and txt.strip():
                if conf is None or conf >= min_conf:
                    lineas.append(txt.strip())

    if not lineas:  
        lineas = [r[1] for r in resultados if isinstance(r, (list, tuple)) and len(r) >= 2 and isinstance(r[1], str)]

    return "\n".join(lineas).strip()


# Interfaz OCR boni

st.subheader("Módulo 1: Lector de Imágenes (OCR)")
archivo_imagen = st.file_uploader("Sube una imagen (PNG, JPG o JPEG)", type=["png", "jpg", "jpeg"])
col_a, col_b = st.columns(2)
with col_a:
    min_conf = st.slider("Confianza mínima del OCR", 0.0, 1.0, 0.5, 0.05)
with col_b:
    st.caption("Valores más altos filtran resultados inseguros.")

if archivo_imagen is not None:
    if st.button("Extraer texto (OCR)", use_container_width=True):
        with st.spinner("Extrayendo texto..."):
            try:
                texto = ocr_con_tu_metodo(archivo_imagen, min_conf)
                st.session_state["ocr_text"] = texto
                if texto:
                    st.success("Texto detectado.")
                else:
                    st.warning("No se detectó texto con suficiente confianza.")
            except Exception as e:
                st.error("No se pudo procesar la imagen.")
                st.exception(e)
else:
    st.info("Sube una imagen para comenzar.")

st.text_area("Texto detectado (editable)", key="ocr_text", height=220)


# GROQ LLM Analysis

st.subheader("Módulo 2: Análisis con GROQ")

groq_model = st.selectbox(
    "Modelo de GROQ",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    index=0,
)

tarea = st.selectbox(
    "Tarea",
    [
        "Resumir en 3 puntos clave",
        "Identificar entidades principales",
        "Traducir al inglés",
        "Analizar sentimiento (positivo/neutral/negativo)",
    ],
)

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
with col2:
    max_tokens = st.slider("Máx. tokens", 64, 2048, 512, 32)

texto = st.session_state.get("ocr_text", "").strip()
disabled_btn = not bool(texto)


# Prompts  para GROQ 
#puntos clave, entidades, traducción, sentimiento
def build_system(tarea_: str) -> str:
    if tarea_ == "Resumir en 3 puntos clave":
        return (
            "Eres un asistente que responde SIEMPRE en español. "
            "TAREA: Resume el texto en EXACTAMENTE TRES puntos clave, cada uno en su propia línea, con guion al inicio y etiqueta.\n"
            "- Punto clave 1: ...\n- Punto clave 2: ...\n- Punto clave 3: ...\n"
            "No incluyas títulos, introducciones ni texto adicional."
        )
    if tarea_ == "Identificar entidades principales":
        return (
            "Eres un asistente que responde SIEMPRE en español. "
            "TAREA: Identifica las identidades/entidades principales del texto y preséntalas en BLOQUES con este formato exacto:\n\n"
            "Nombre de la entidad\n"
            "→ Explicación breve en 1–2 líneas que diga quién/qué es y qué representa en el texto.\n\n"
            "Deja UNA línea en blanco entre bloques. No repitas entidades. No agregues encabezados."
        )
    if tarea_ == "Traducir al inglés":
        return (
            "You are a professional translator. "
            "Output ONLY the English translation of the user's text. "
            "No headers, no explanations, no quotes."
        )
   
    return (
        "Eres un asistente que responde SIEMPRE en español. "
        "TAREA: Analiza el sentimiento y responde en UNA sola línea con el formato:\n"
        "Positivo: ...   o   Neutral: ...   o   Negativo: ...\n"
        "Incluye una justificación breve. No añadas más texto."
    )

def build_user(tarea_: str, txt: str) -> str:
    if tarea_ == "Resumir en 3 puntos clave":
        return f"Resume el siguiente texto en exactamente tres puntos clave, siguiendo el formato con guiones:\n\n{txt}"
    if tarea_ == "Identificar entidades principales":
        return (
            "Identifica las identidades/entidades principales y devuélvelas en bloques con el formato indicado "
            "(Nombre en una línea; en la siguiente línea, una flecha '→' seguida de una explicación breve en español; "
            "una línea en blanco entre bloques; sin repeticiones):\n\n" + txt
        )
    if tarea_ == "Traducir al inglés":
        return f"Translate this text into natural, fluent English. Output only the translation:\n\n{txt}"
    return f"Analiza el sentimiento (positivo, neutral o negativo) del siguiente texto y justifica brevemente en una sola línea:\n\n{txt}"

def groq_chat_completion(model, system_msg, user_msg, temp, max_t, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": float(temp),
        "max_tokens": int(max_t),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

if st.button("Analizar texto con GROQ", type="primary", use_container_width=True, disabled=disabled_btn):
    if not texto:
        st.warning("Primero extrae o pega texto.")
        st.stop()

    with st.spinner("Consultando GROQ..."):
        try:
            out = groq_chat_completion(
                groq_model,
                build_system(tarea),
                build_user(tarea, texto),
                temperature,
                max_tokens,
                GROQ_API_KEY,
            )
            st.session_state["llm_output"] = out
        except Exception as e:
            st.error("Error al consultar GROQ.")
            st.exception(e)

if st.session_state.get("llm_output"):
    st.success("Respuesta de GROQ")
    st.markdown(st.session_state["llm_output"])
