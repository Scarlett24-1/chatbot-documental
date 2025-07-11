import os
import faiss
import numpy as np
import subprocess
from sentence_transformers import SentenceTransformer

# Cargar índice y fragmentos
index = faiss.read_index("embeddings/index.faiss")
fragmentos = np.load("embeddings/textos.npy", allow_pickle=True)
modelo = SentenceTransformer('all-MiniLM-L6-v2')

print("🤖 Chatbot iniciado. Escribe 'salir' para terminar.")

while True:
    pregunta = input("\nTú: ")
    if pregunta.strip().lower() == "salir":
        print("👋 ¡Hasta luego!")
        break

    # Crear embedding de la pregunta
    emb_pregunta = modelo.encode([pregunta])
    # Buscar el fragmento más cercano
    D, I = index.search(emb_pregunta, k=1)
    contexto = fragmentos[I[0][0]]

    # Preparar el prompt para DeepSeek
    prompt = f"""
    Eres un experto en el siguiente texto y respondes como tutor académico.
    Texto:
    {contexto}

    Pregunta:
    {pregunta}

    Respuesta:
    """

    # Llamar al modelo local con Ollama
    resultado = subprocess.run(
        ["ollama", "run", "deepseek-coder:instruct"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    respuesta = resultado.stdout.decode("utf-8").strip()
    # Start Generation Here
    # Definiciones clave de Machine Learning
    definiciones = {
        "Aprendizaje Supervisado": "Es un tipo de aprendizaje automático donde el modelo aprende a partir de datos etiquetados. El algoritmo recibe ejemplos de entrada junto con las respuestas correctas (etiquetas) y aprende a mapear las entradas a las salidas correctas.",
        "Aprendizaje No Supervisado": "Es un tipo de aprendizaje donde el modelo encuentra patrones ocultos en datos sin etiquetar. No hay respuestas correctas predefinidas, el algoritmo debe descubrir la estructura por sí mismo.",
        "Aprendizaje por Refuerzo": "Es un tipo de aprendizaje donde un agente aprende a tomar decisiones mediante la interacción con un entorno, recibiendo recompensas o castigos por sus acciones.",
        "Overfitting": "Ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento, perdiendo la capacidad de generalizar a nuevos datos.",
        "Underfitting": "Ocurre cuando un modelo es demasiado simple y no puede capturar los patrones en los datos de entrenamiento.",
        "Validación Cruzada": "Técnica para evaluar modelos dividiendo los datos en múltiples subconjuntos y entrenando/validando en diferentes combinaciones.",
        "Feature Engineering": "Proceso de crear, transformar o seleccionar características (variables) que mejoran el rendimiento del modelo.",
        "Normalización": "Proceso de escalar los datos para que estén en un rango específico, típicamente entre 0 y 1.",
        "Estandarización": "Proceso de transformar los datos para que tengan media 0 y desviación estándar 1."
    }
    
    print("\n📚 Definiciones importantes de Machine Learning:")
    for termino, definicion in definiciones.items():
        print(f"\n🔹 {termino}:")
        print(f"   {definicion}")
