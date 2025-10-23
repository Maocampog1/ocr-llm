

##  Proyecto IA: OCR + LLM

###  Demo en línea

-> [Abrir la aplicación en Streamlit](https://ocr-llm-fda3pjkqvxe8gxlpuabwcq.streamlit.app/)

---

### Descripción

Este proyecto integra **Visión Artificial (OCR)** y **Procesamiento de Lenguaje Natural (LLM)** para crear una aplicación que:

1. **Lee texto desde imágenes** usando EasyOCR.
2. **Procesa el texto** con un modelo LLM de **GROQ (Llama 3.1)** para realizar tareas como:

   * Resumen en puntos clave
   * Análisis de sentimiento
   * Identificación de entidades principales
   * Traducción al inglés

Todo dentro de una **interfaz web interactiva construida con Streamlit**.

---

### Instalación y ejecución local

#### 1️. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/ocr-llm.git
cd ocr-llm
```

#### 2️. Crear y activar un entorno virtual

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3️. Instalar dependencias

```bash
pip install -r requirements.txt
```

#### 4️. Crear el archivo `.env`

Crea un archivo llamado `.env` en la raíz del proyecto con tu clave de GROQ:

```bash
GROQ_API_KEY="tu_clave_aqui"
```

#### 5️. Ejecutar la aplicación

```bash
streamlit run app.py
```

Luego abre el enlace que aparece en la terminal, normalmente:

```
http://localhost:8501
```

---
