

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

## Puntos de Discusión y Reflexión Final

### Diferencias de velocidad entre GROQ y Hugging Face
Durante el desarrollo del proyecto se intentó conectar Hugging Face con el modelo `mistralai/Mistral-7B-Instruct-v0.2`,  
pero la API devolvía errores de conexión y tiempos de respuesta muy altos.  
Por eso, se decidió trabajar solo con GROQ, el cual resultó ser mucho más rápido y estable.

Explicación técnica:
- GROQ utiliza infraestructura optimizada en hardware para ejecutar modelos con baja latencia, lo que hace que las respuestas sean casi instantáneas.  
- Hugging Face, en cambio, funciona sobre servidores compartidos, lo que puede generar demoras variables por uso simultáneo o “cold starts”.  
- GROQ ofrece una velocidad superior y una experiencia más fluida, mientras que Hugging Face destaca por su variedad de modelos, aunque con tiempos de respuesta menos predecibles.

### Cómo afecta el parámetro temperature
El parámetro temperature controla la creatividad o precisión del modelo:

- Valores bajos (0.0 – 0.3): respuestas precisas, repetibles y más lógicas.  
  Ideal para análisis de sentimiento, resumen estructurado o traducción literal.  
- Valores altos (0.6 – 1.0): respuestas más creativas, variadas y con matices.  
  Útil para redacciones más libres o generación de ideas nuevas.

En este taller se usaron valores bajos para mantener la coherencia y evitar respuestas aleatorias.

### Importancia del texto extraído por el OCR
La calidad del texto obtenido con OCR fue clave para el rendimiento del modelo LLM.

Al inicio, se probó un método que no detectaba bien las palabras ni los acentos, lo que provocaba que las respuestas del modelo fueran confusas.  
Después, con la implementación de EasyOCR cacheado y configurado correctamente, el texto se extrajo de forma más limpia y coherente, mejorando el análisis del LLM.

Un texto claro y bien estructurado permite al modelo generar resultados más acertados.  
La calidad del OCR define la calidad del análisis final.

### Qué más se podría integrar en la aplicación
Algunas ideas futuras para ampliar esta app serían:

- Clasificación automática del texto (por tipo: noticia, carta, reseña, etc.).  
- Preguntas y respuestas sobre el texto detectado.  
- Mejoras gramaticales y ortográficas al texto OCR.  
- Traducciones a más idiomas (francés, portugués, alemán, etc.).  
- Identificación de lugares, personas o fechas dentro del texto.  
- Generación de texto creativo, como resúmenes poéticos o narrativos a partir del contenido.  

Estas extensiones podrían hacerse fácilmente reutilizando el mismo flujo OCR + LLM y ampliando las opciones de tarea.

## Conclusión General
Este proyecto permitió aprender de forma práctica cómo combinar dos ramas muy diferentes de la Inteligencia Artificial:  
la Visión Artificial (OCR) para reconocer texto desde imágenes, y el Procesamiento de Lenguaje Natural (LLM) para entender, analizar y transformar ese texto.

Durante el desarrollo se comprendió cómo funcionan herramientas modernas como Streamlit, EasyOCR y GROQ,  
así como la importancia de manejar correctamente parámetros como la temperature y la limpieza del texto de entrada.

Además de ser un proyecto técnico, fue una experiencia muy interesante y divertida,  
ya que permitió integrar conocimientos de programación, IA y diseño de interfaces en una aplicación completa y funcional.

Gracias profe por las clases.  
**Att:** Luis, Camilo y Aleja.

