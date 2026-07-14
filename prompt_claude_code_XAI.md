# Prompt para Claude Code — Práctica B4-T2 (XAI, concesión de crédito)

Copia y pega esto (ajusta rutas si hace falta) como primer mensaje a Claude Code en el directorio del proyecto.

---

## Contexto

Estoy cursando el máster MIAX y tengo que entregar una práctica de credit scoring con foco fuerte en auditoría y explicabilidad (XAI), no solo en performance. La nota se reparte:
- 50% resultado: **coste promedio del modelo en el dataset de producción**, bajo dos matrices de coste distintas.
- 50% análisis/auditoría/explicaciones: calidad, coherencia y reflexión crítica.

Quiero que construyas un proyecto completo, reproducible, en un notebook Jupyter (`.ipynb`) más un par de módulos `.py` auxiliares si conviene para mantener el notebook legible. El objetivo es maximizar ambas partes de la nota, con especial cuidado en que la parte de auditoría sea *sustanciosa y crítica*, no un análisis superficial de relleno.

## Datos

Los datos de partida están en esta carpeta de Google Drive: `https://drive.google.com/drive/folders/1G39FiV3R7v8c5sDtB5NRbwz8d40BuTu7`. Yo los descargaré manualmente y los dejaré en `./data/raw/`. Empieza haciendo un inventario de los ficheros que encuentres ahí (esquema, tipos, nulos, desbalanceo de la variable objetivo, duplicados, fugas de información potenciales) antes de tocar nada de modelado. No asumas nombres de columnas: primero explóralos y muéstramelos.

## Tarea a resolver

Dos escenarios de coste sobre el mismo problema de concesión de crédito:
1. **Escenario 1**: Coste Falso Positivo = Coste Falso Negativo = 1 → predicciones de producción en `cs_produccion1.csv`.
2. **Escenario 2**: Coste Falso Positivo = 1, Coste Falso Negativo = 10 → predicciones de producción en `cs_produccion2.csv`.

El formato exacto de estos CSV de producción debe replicar el que usa el notebook de partida de la carpeta de Drive (mismo id de cliente / columnas esperadas) — revisa ese notebook de ejemplo antes de definir el formato de salida.

## Enfoque de modelado que quiero que sigas

1. **Modelo principal**: gradient boosting (XGBoost o LightGBM) como candidato base, con validación cruzada estratificada. Compara al menos un segundo modelo (regresión logística regularizada, como baseline interpretable) para poder argumentar la elección en la sección de discusión.
2. **Optimización cost-sensitive real, no solo threshold tuning naive**:
   - Ajusta el umbral de decisión minimizando el coste esperado según la matriz de cada escenario (no uses 0.5 por defecto).
   - Evalúa también `sample_weight` o `scale_pos_weight` cost-sensitive en el entrenamiento, y compara contra el ajuste de umbral puro en post-processing. Quiero ver el trade-off documentado, no solo el resultado final.
   - Usa nested cross-validation o un split train/val/test bien separado para que la elección de umbral no haga overfitting sobre el mismo conjunto que reporta el coste final.
3. **Multi-armed bandit**: dado que el enunciado lo menciona como alternativa, añade una sección breve (no el núcleo del proyecto) donde discutas por qué un enfoque supervisado es más adecuado que un bandit aquí (no hay exploración online real posible sobre un dataset estático de producción), pero simula un enfoque contextual bandit simplificado como comparación conceptual y argumenta por qué lo descartas o no.
4. Reporta el coste promedio final en producción para cada escenario de forma clara y destacada (tabla resumen al principio y al final del notebook).

## Auditoría y explicabilidad (aquí está la mitad de la nota, dale peso real)

- **Modelo subrogado**: entrena un árbol de decisión poco profundo (o reglas tipo RuleFit/Skope-rules) sobre las predicciones del modelo caja negra, reporta fidelidad (agreement rate) del subrogado respecto al modelo real, y extrae reglas humanamente legibles.
- **Contrafactuales**: selecciona 2-3 ejemplos reales de clase 0 y 2-3 de clase 1 (casos con distinto nivel de confianza del modelo, no solo los más obvios), genera contrafactuales (usa `dice-ml` si es compatible con el entorno, si no, implementa una búsqueda por gradiente/optimización simple) y redacta, para cada uno, la explicación en lenguaje natural que le darías a ese cliente si te pregunta por qué se le deniega el crédito. Esto es literalmente una pregunta del enunciado — respóndela explícitamente y no de forma genérica.
- **SHAP global y local**: summary plot, dependence plots de las 3-4 variables más influyentes, y explicación local (waterfall/force plot) para los mismos ejemplos usados en contrafactuales, cruzando ambos análisis (¿coinciden contrafactual y SHAP en qué variables importan para ese cliente?).
- **Extra que suma nota**: análisis de fairness/sesgo si hay variables demográficas o proxies (aunque no se pida explícitamente, una mención crítica eleva la calidad del análisis), y un análisis de estabilidad del modelo (¿las explicaciones SHAP cambian mucho entre escenario 1 y 2 al cambiar el umbral/pesos?).

## Estructura del notebook

1. Introducción y objetivo (breve)
2. Carga y exploración de datos (EDA con foco en variable objetivo y calidad de datos)
3. Preprocesamiento (justifica cada decisión: encoding, missing values, escalado)
4. Modelado — Escenario 1 (entrenamiento, validación, ajuste de umbral, coste final)
5. Modelado — Escenario 2 (mismo esquema)
6. Comparación de escenarios y discusión de sensibilidad de la matriz de coste
7. Auditoría: modelo subrogado
8. Auditoría: contrafactuales
9. Auditoría: SHAP global/local
10. (Opcional) Discusión bandit contextual
11. Generación de `cs_produccion1.csv` y `cs_produccion2.csv`
12. Reflexión final crítica: limitaciones del modelo, riesgos de despliegue, qué harías con más tiempo/datos

## Requisitos técnicos

- Entorno Python, todo reproducible con seeds fijas.
- **Para cualquier generación de números aleatorios con NumPy usa siempre `np.random.default_rng()`, nunca `np.random.randint` ni otras funciones legacy del estado global.**
- Usa un `requirements.txt` o celda de instalación al principio.
- Comenta el código lo justo para que se entienda la lógica de negocio (coste, no solo la mecánica de ML).
- Antes de cerrar cada sección, añade una celda markdown corta con la interpretación de resultados — es lo que se evalúa en el 50% de "análisis y reflexión crítica", no solo el código que corre.

## Cómo quiero que trabajes

Empieza explorando los datos y el notebook de ejemplo de la carpeta de Drive, muéstrame lo que encuentras y el plan detallado de modelado antes de lanzarte a construir todo el notebook de golpe. Iteremos por secciones.
