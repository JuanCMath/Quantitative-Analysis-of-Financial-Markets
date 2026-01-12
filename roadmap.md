# 📊 Proyecto de Estadística — Análisis de Activos Representativos de la Bolsa  
Curso 2025–2026 — MATCOM, UH  
> Roadmap del proyecto

Este documento resume la planificación completa del proyecto siguiendo las orientaciones oficiales del curso.

---

## 🟦 1. Tema del Proyecto
**Análisis estadístico de activos representativos del mercado bursátil estadounidense**, evaluando volatilidad, correlaciones, patrones comunes y capacidad predictiva básica.

Activos sugeridos:
- SPY (S&P 500)
- QQQ (Nasdaq 100)
- AAPL, MSFT, NVDA, TSLA
- BTC-USD (como comparación alternativa)

---

## 🟦 2. Preguntas de Investigación
1. ¿Qué activo presenta mayor volatilidad en el período analizado?  
2. ¿Existen correlaciones significativas entre los activos seleccionados?  
3. ¿Se pueden identificar clusters o grupos naturales según su comportamiento estadístico?  
4. ¿Es posible predecir el movimiento diario (subida/bajada) de un activo usando variables estadísticas simples?

---

## 🟦 3. Dataset y Obtención de Datos
Fuente: Yahoo Finance vía `yfinance` o datasets equivalentes.

Variables a utilizar:
- Open, High, Low, Close, Adj Close  
- Volume  
- Retorno diario  
- Retorno logarítmico  
- Volatilidad móvil  

Pasos:
- Descarga de datos históricos
- Revisión de estructura del dataset
- Manejo de valores faltantes
- Selección del rango temporal (ej. últimos 5–10 años)

---

## 🟦 4. Análisis Exploratorio de Datos (EDA)
Tareas principales:
- Estadísticos descriptivos: media, varianza, desviación estándar
- Histogramas de retornos
- Boxplots de volatilidad
- Series temporales comparadas
- Scatter plots entre activos
- Heatmap de correlaciones
- Identificación de outliers
- Discusión inicial conectada con las preguntas de investigación

---

## 🟦 5. Preparación de Datos
Transformaciones necesarias:
- Cálculo de retornos diarios y logarítmicos
- Cálculo de volatilidades móviles
- Estandarización de variables para PCA y clustering
- Creación de variable target para clasificación:  
  - `1` → el activo sube mañana  
  - `0` → el activo baja mañana  

---

## 🟦 6. Técnicas Estadísticas a Aplicar (mínimo 3)
### ✔ 6.1 Pruebas de Hipótesis
- t-test entre dos activos (ej. AAPL vs MSFT)
- ANOVA para comparar medias entre todos los activos
- Pruebas de normalidad sobre los retornos

### ✔ 6.2 Regresión
- **Lineal:** relación entre retornos de SPY y activos individuales  
- **Logística:** predicción de subida/bajada del mercado

### ✔ 6.3 PCA (Análisis de Componentes Principales)
- Reducción de dimensionalidad  
- Interpretación de componentes  
- Visualización 2D

### ✔ 6.4 Clustering
- K-Means para identificar grupos de activos
- Visualización en espacio PCA

---

## 🟦 7. Resultados y Conclusiones
- Resumen de métricas clave  
- Respuestas claras a las preguntas iniciales  
- Interpretación estadística fundamentada  
- Identificación del activo más volátil  
- Análisis de correlaciones fuertes  
- Hallazgos del PCA y clustering  
- Evaluación del modelo predictivo  
- Discusión de limitaciones:
  - ruido de mercado  
  - rango temporal  
  - modelos simples  
- Propuestas de mejora futura

---

## 🟦 8. Entregables
1. **Notebook final (.ipynb)** con:
   - Flujo completo del proyecto  
   - Código limpio y comentado  
   - Interpretaciones claras  
   - Gráficos y análisis  

2. **Presentación (máx. 12 diapositivas)**:
   - Contexto  
   - Preguntas  
   - EDA  
   - Técnicas aplicadas  
   - Resultados claves  
   - Conclusiones  

3. **Exposición oral (10–12 min)**:
   - Explicación de decisiones  
   - Interpretación de técnicas  
   - Defensa de resultados  

---

## 🟦 9. Estructura del Repo

├── data/ # Datos descargados
├── notebook/ # Jupyter Notebook del proyecto
├── presentation/ # Presentación final
├── ROADMAP.md # Este archivo
└── README.md # Información general del proyecto


---

## 🟦 10. Estado del Proyecto
- [ ] Recopilación de datos  
- [ ] Exploración inicial  
- [ ] Preparación de datos  
- [ ] Técnicas estadísticas aplicadas  
- [ ] Resultados y conclusiones  
- [ ] Preparar presentación  
- [ ] Ensayo de la exposición  

---
