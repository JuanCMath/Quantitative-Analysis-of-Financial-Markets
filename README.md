# 📊 Análisis Cuantitativo de Mercados Financieros

**Proyecto de Estadística — MATCOM, Universidad de La Habana**  
Curso 2025–2026

Un análisis estadístico integral de activos representativos del mercado bursátil estadounidense, aplicando técnicas avanzadas de análisis de datos para evaluar volatilidad, correlaciones, patrones de comportamiento y capacidad predictiva.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Preguntas de Investigación](#-preguntas-de-investigación)
- [Activos Analizados](#-activos-analizados)
- [Características Principales](#-características-principales)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Metodología](#-metodología)
- [Resultados Esperados](#-resultados-esperados)
- [Limitaciones](#-limitaciones)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

Este proyecto aplica técnicas estadísticas avanzadas para analizar el comportamiento de activos financieros del mercado estadounidense. A través de Python y diversas bibliotecas especializadas, se realiza un análisis cuantitativo exhaustivo que incluye:

- Análisis exploratorio de datos (EDA)
- Pruebas de hipótesis estadísticas
- Modelos de regresión lineal y logística
- Análisis de componentes principales (PCA)
- Clustering para identificación de patrones

El objetivo principal es comprender las relaciones entre diferentes activos, identificar patrones de comportamiento y evaluar la capacidad predictiva de modelos estadísticos simples.

---

## ❓ Preguntas de Investigación

1. **¿Qué activo presenta mayor volatilidad en el período analizado?**
2. **¿Existen correlaciones significativas entre los activos seleccionados?**
3. **¿Se pueden identificar clusters o grupos naturales según su comportamiento estadístico?**
4. **¿Es posible predecir el movimiento diario (subida/bajada) de un activo usando variables estadísticas simples?**

---

## 📈 Activos Analizados

- **AAPL** — Apple Inc.
- **MSFT** — Microsoft Corporation
- **NVDA** — NVIDIA Corporation
- **TSLA** — Tesla Inc.
- **GLD** — SPDR Gold Shares (oro)

**Rango temporal:** 2018-01-01 hasta 2025-01-01

---

## ✨ Características Principales

### 📊 Análisis Exploratorio
- Cálculo de rendimientos diarios y logarítmicos
- Estadísticos descriptivos (media, varianza, desviación estándar)
- Visualizaciones: histogramas, boxplots, series temporales
- Matriz de correlación con mapa de calor

### 🧪 Técnicas Estadísticas
- **Pruebas de Hipótesis:** t-test, ANOVA, pruebas de normalidad
- **Regresión Lineal:** relaciones entre rendimientos de activos
- **Regresión Logística:** predicción de movimientos del mercado
- **PCA:** reducción de dimensionalidad e identificación de patrones
- **K-Means Clustering:** agrupación de activos por comportamiento

### 📉 Métricas de Evaluación
- Volatilidad y riesgo
- Coeficientes de correlación
- R² y métricas de regresión
- Accuracy, matriz de confusión para clasificación
- Varianza explicada por componentes principales

---

## 🛠️ Tecnologías Utilizadas

### Lenguajes y Entorno
- **Python 3.13** (gestionado con `uv`)
- **uv** — Gestor de paquetes y entornos virtuales ultrarrápido
- **Jupyter Notebook** para análisis interactivo

### Bibliotecas Principales

| Biblioteca | Versión | Propósito |
|------------|---------|-----------|
| `numpy` | ≥2.3.5 | Computación numérica |
| `pandas` | ≥2.3.3 | Manipulación de datos |
| `matplotlib` | ≥3.10.7 | Visualización de datos |
| `seaborn` | ≥0.13.2 | Gráficos estadísticos |
| `scipy` | ≥1.16.3 | Pruebas estadísticas |
| `scikit-learn` | ≥1.7.2 | Machine Learning |
| `yfinance` | ≥0.2.66 | Descarga de datos financieros |

---

## 🚀 Instalación

### Requisitos Previos
- **[uv](https://docs.astral.sh/uv/)** — Instalador de paquetes Python ultrarrápido
- Python 3.13 (se instalará automáticamente con `uv` si no está disponible)

### Instalación de uv

Si aún no tienes `uv` instalado:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Configuración del Proyecto

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/quantitative-analysis-of-financial-markets.git
cd quantitative-analysis-of-financial-markets

# Sincronizar dependencias (crea el entorno virtual automáticamente)
uv sync

# El proyecto está listo para usar
```

### Comandos útiles con uv

```bash
# Ejecutar Python en el entorno del proyecto
uv run python main.py

# Crear el kernel para Jupyter
uv run python -m ipykernel install --user --name bolsa-uv --display-name "Python (bolsa-uv)"
# Abres tu Notebook.ipynby en la esquina superior derecha eliges el kernel: Kernel → Change kernel → Python (bolsa-uv)


# Ejecutar Jupyter Notebook
uv run jupyter notebook

# Agregar una nueva dependencia
uv add nombre-paquete

# Actualizar dependencias
uv sync --upgrade
```

> **Nota:** `uv` gestiona automáticamente el entorno virtual y las dependencias definidas en `pyproject.toml`. No necesitas activar manualmente ningún entorno virtual.

---

## 📝 Uso

### 1. Descargar Datos

Los datos se descargan automáticamente desde Yahoo Finance usando `yfinance`. Ejecuta las celdas correspondientes en el notebook:

```python
# La descarga se realiza en la sección 3.2 del notebook
# Los archivos se guardan en la carpeta data/
```

### 2. Ejecutar el Análisis

#### Opción A: Con VS Code (Recomendado)
Simplemente abre el archivo `notebook/Notebook.ipynb` en VS Code y ejecuta las celdas secuencialmente. VS Code detectará automáticamente el entorno de `uv`.

#### Opción B: Con Jupyter Notebook
```bash
# Ejecutar Jupyter Notebook con uv
uv run jupyter notebook notebook/Notebook.ipynb
```

#### Opción C: Con JupyterLab
```bash
# Primero instalar jupyterlab si no está instalado
uv add jupyterlab

# Ejecutar JupyterLab
uv run jupyter lab
```

### 3. Explorar Resultados

El notebook está organizado en secciones:
1. **Configuración inicial** — Importación de librerías
2. **Recopilación de datos** — Descarga y carga de datos
3. **Análisis exploratorio** — EDA completo
4. **Preparación de datos** — Transformaciones
5. **Técnicas estadísticas** — Aplicación de modelos
6. **Resultados y conclusiones** — Interpretación

---

## 📁 Estructura del Proyecto

```
quantitative-analysis-of-financial-markets/
│
├── data/                          # Datos de activos financieros (CSV)
│   ├── SPY.csv
│   ├── QQQ.csv
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── NVDA.csv
│   ├── TSLA.csv
│   └── GLD.csv
│
├── notebook/                      # Jupyter Notebooks
│   └── Notebook.ipynb            # Notebook principal del análisis
│
├── presentation/                  # Presentaciones y reportes
│
├── .python-version               # Versión de Python (3.13)
├── main.py                        # Script principal (opcional)
├── pyproject.toml                # Configuración del proyecto y dependencias
├── uv.lock                       # Lock file de dependencias (gestiona uv)
├── roadmap.md                    # Roadmap del proyecto
└── README.md                     # Este archivo
```

---

## 🔬 Metodología

### 1. Recopilación de Datos
- Descarga de datos históricos desde Yahoo Finance
- Rango temporal: 2018-2025 (7 años)
- Variables: Open, High, Low, Close, Adj Close, Volume

### 2. Análisis Exploratorio (EDA)
- Cálculo de rendimientos diarios: `(P_t - P_{t-1}) / P_{t-1}`
- Visualizaciones de distribuciones y tendencias
- Análisis de correlaciones entre activos

### 3. Preparación de Datos
- Limpieza de valores faltantes
- Estandarización para PCA y clustering
- Creación de variable objetivo para clasificación

### 4. Aplicación de Técnicas Estadísticas

#### 🧪 Pruebas de Hipótesis
- **t-test de Welch:** comparación de medias entre activos
- Nivel de significancia: α = 0.05

#### 📈 Regresión Lineal
- Modelado de relaciones entre rendimientos
- Evaluación mediante R²

#### 🎯 Regresión Logística
- Predicción de dirección del mercado (subida/bajada)
- Métricas: accuracy, precision, recall, F1-score

#### 🔍 PCA
- Reducción de dimensionalidad
- Identificación de componentes principales
- Visualización 2D de patrones

#### 🎨 K-Means Clustering
- Agrupación de activos por comportamiento
- Visualización en espacio PCA

---

## 🎯 Resultados Esperados

Al finalizar el análisis, se espera obtener:

✅ **Identificación del activo más volátil**  
✅ **Mapa de correlaciones entre activos**  
✅ **Grupos naturales de activos con comportamiento similar**  
✅ **Modelo predictivo básico con métricas de desempeño**  
✅ **Interpretación estadística de patrones del mercado**  
✅ **Visualizaciones claras y profesionales**  

---

## ⚠️ Limitaciones

- **Alcance temporal:** El análisis se limita al período 2018-2025
- **Simplicidad de modelos:** Se utilizan modelos estadísticos básicos, no técnicas avanzadas de ML
- **Factores externos:** No se consideran variables macroeconómicas, noticias o eventos geopolíticos
- **Eficiencia del mercado:** Los mercados financieros son altamente eficientes y difíciles de predecir
- **Datos históricos:** El rendimiento pasado no garantiza resultados futuros

---

## 🤝 Contribuciones

Este es un proyecto académico. Si deseas contribuir o tienes sugerencias:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

---

## 📚 Referencias

- Apuntes y materiales del curso de Estadística — MATCOM, UH
- [Documentación de uv](https://docs.astral.sh/uv/) — Gestor de paquetes
- [Documentación de pandas](https://pandas.pydata.org/docs/)
- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Documentación de yfinance](https://pypi.org/project/yfinance/)
- Yahoo Finance para datos de mercado

---

## 📄 Licencia

Este proyecto es de uso académico para el curso de Estadística de MATCOM, Universidad de La Habana.

---

## 👨‍💻 Autores

- **Juan Carlos Carmenate Díaz**  
Estudiante de MATCOM, Universidad de La Habana  
Curso 2025–2026

- **Sebastian González Alfonso**  
Estudiante de MATCOM, Universidad de La Habana  
Curso 2025–2026

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub ⭐**

</div>
