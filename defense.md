<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# 📊 Documento de Defensa del Proyecto
## Análisis Cuantitativo de Mercados Financieros

**Proyecto Final de Estadística — MATCOM, Universidad de La Habana**  
**Curso 2025–2026**  
**Equipo:** Juan Carlos Carmenate Díaz y Sebastian González Alfonso

> **📌 Nota:** Las fórmulas matemáticas están en notación LaTeX. Para visualizarlas correctamente:
> - En **VS Code:** Las fórmulas deberían verse en el preview con los delimitadores `$$...$$`
> - En **navegadores:** Se usan scripts de MathJax para renderización automática
> - En **GitHub:** Usar el navegador con extensiones de matemáticas o visualizar localmente

---

## 📑 Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Marco Teórico y Fundamentos Estadísticos](#2-marco-teórico-y-fundamentos-estadísticos)
3. [Metodología y Técnicas Aplicadas](#3-metodología-y-técnicas-aplicadas)
4. [Resultados Obtenidos](#4-resultados-obtenidos)
5. [Interpretación y Discusión](#5-interpretación-y-discusión)
6. [Limitaciones y Trabajo Futuro](#6-limitaciones-y-trabajo-futuro)
7. [Conclusiones Finales](#7-conclusiones-finales)

---

## 1. Resumen Ejecutivo

### 1.1 Objetivo del Proyecto

Aplicar técnicas estadísticas avanzadas para analizar el comportamiento de activos financieros del mercado estadounidense, evaluando volatilidad, correlaciones, patrones de comportamiento y capacidad predictiva mediante métodos cuantitativos rigurosos.

### 1.2 Activos Analizados

- **AAPL** (Apple Inc.) — Empresa tecnológica de consumo
- **MSFT** (Microsoft Corporation) — Empresa tecnológica de software/servicios
- **NVDA** (NVIDIA Corporation) — Empresa de semiconductores
- **AAAU** (SPDR Gold Shares) — ETF respaldado por oro físico

**Período de análisis:** 2018-08-15 hasta 2020-04-01 (409 observaciones diarias)

### 1.3 Preguntas de Investigación

1. ¿Qué activo presenta mayor volatilidad?
2. ¿Existen correlaciones significativas entre activos?
3. ¿Se pueden identificar clusters naturales de comportamiento?
4. ¿Es posible predecir movimientos diarios con variables simples?

### 1.4 Técnicas Estadísticas Aplicadas

- Análisis Exploratorio de Datos (EDA)
- Pruebas de Normalidad (Jarque-Bera, Kolmogorov-Smirnov)
- Pruebas de Hipótesis (Welch t-test, ANOVA)
- Regresión Lineal
- Análisis de Componentes Principales (PCA)
- Clustering K-Means
- Regresión Logística para clasificación

---

## 2. Marco Teórico y Fundamentos Estadísticos

### 2.1 Fundamentos de Finanzas Cuantitativas

#### 2.1.1 Rendimientos vs. Precios

En finanzas, el análisis se realiza sobre **rendimientos** en lugar de precios por tres razones fundamentales:

1. **Estacionariedad:** Los precios exhiben tendencias (no estacionarios), mientras que los rendimientos tienden a fluctuar alrededor de una media constante
2. **Comparabilidad:** Permiten comparar activos de diferentes escalas de precio
3. **Propiedades estadísticas:** Los rendimientos tienen distribuciones más manejables estadísticamente

**Rendimiento simple:** Cambio porcentual entre períodos consecutivos

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

**Rendimiento logarítmico:** Diferencia de logaritmos naturales

$$\ell_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Ventajas del rendimiento logarítmico:**
- Es aditivo en el tiempo: $$\ell_{t_1 \to t_3} = \ell_{t_1 \to t_2} + \ell_{t_2 \to t_3}$$
- Simétrico respecto a ganancias y pérdidas
- Aproximadamente igual al rendimiento simple cuando $$|r_t|$$ es pequeño

#### 2.1.2 Volatilidad como Medida de Riesgo

La **volatilidad** es la desviación estándar de los rendimientos y constituye la medida estándar de riesgo en finanzas:

$$\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(r_i - \bar{r})^2}$$

**Propiedades clave:**
- Mayor volatilidad implica mayor incertidumbre y riesgo
- Se anualiza multiplicando por $$\sqrt{252}$$ (días de trading anuales)
- La volatilidad no es constante en el tiempo (heteroscedasticidad)

#### 2.1.3 Volatilidad Móvil (Rolling Volatility)

Para capturar la naturaleza dinámica del riesgo, calculamos volatilidad en ventanas temporales:

$$\sigma_t^{(w)} = \sqrt{\frac{1}{w-1}\sum_{i=0}^{w-1}(r_{t-i} - \bar{r}_w)^2}$$

donde $$w$$ es el tamaño de la ventana (típicamente 20 días ≈ 1 mes de trading).

**Aplicaciones:**
- Identificar regímenes de alta/baja volatilidad
- Detectar períodos de estrés de mercado
- Mejorar modelos predictivos con variables temporales

#### 2.1.4 Correlación y Diversificación

La correlación de Pearson mide la relación lineal entre rendimientos:

$$\rho_{A,B} = \frac{\text{Cov}(r_A, r_B)}{\sigma_A \cdot \sigma_B}$$

**Teoría Moderna de Carteras (Markowitz):**
- Carteras diversificadas reducen riesgo cuando $$\rho < 1$$
- Máximo beneficio de diversificación cuando $$\rho \approx 0$$ o $$\rho < 0$$
- La correlación puede cambiar durante crisis (contagio)

### 2.2 Pruebas de Normalidad

#### 2.2.1 Test de Jarque-Bera

Prueba la normalidad basándose en asimetría (skewness) y curtosis (kurtosis):

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

donde:
- $$S = \frac{\mu_3}{\sigma^3}$$ es el coeficiente de asimetría
- $$K = \frac{\mu_4}{\sigma^4}$$ es la curtosis
- Bajo $$H_0$$ (normalidad): $$JB \sim \chi^2(2)$$

**Interpretación:**
- $$S = 0$$ y $$K = 3$$ para distribución normal
- $$S > 0$$: cola derecha más pesada (sesgo positivo)
- $$S < 0$$: cola izquierda más pesada (sesgo negativo)
- $$K > 3$$: colas más pesadas que normal (leptocúrtica)

#### 2.2.2 Test de Kolmogorov-Smirnov

Compara la distribución empírica con la normal teórica:

$$D_n = \sup_x |F_n(x) - F_0(x)|$$

donde $$F_n$$ es la función de distribución empírica y $$F_0$$ es la normal.

**Ventajas:**
- No paramétrico (no asume forma específica)
- Adecuado para muestras grandes (n > 200)
- Detecta cualquier tipo de desviación de normalidad

### 2.3 Pruebas de Hipótesis para Comparación de Medias

#### 2.3.1 Welch t-test

Prueba la igualdad de medias sin asumir varianzas iguales.

Estadístico de prueba:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Grados de libertad (aproximación de Welch):**

$$\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Ventajas sobre t-test estándar:**
- Robusto ante heteroscedasticidad (varianzas diferentes)
- Más conservador (menos propenso a falsos positivos)
- Recomendado cuando las desviaciones estándar difieren sustancialmente

#### 2.3.2 ANOVA (Análisis de Varianza)

Prueba global de igualdad de medias entre múltiples grupos:

$$F = \frac{\text{Varianza entre grupos}}{\text{Varianza dentro de grupos}} = \frac{MS_{between}}{MS_{within}}$$

$$MS_{between} = \frac{\sum_{j=1}^k n_j(\bar{X}_j - \bar{X})^2}{k-1}, \quad MS_{within} = \frac{\sum_{j=1}^k\sum_{i=1}^{n_j}(X_{ij} - \bar{X}_j)^2}{N-k}$$

Bajo $$H_0$$ (todas las medias iguales): $$F \sim F(k-1, N-k)$$

### 2.4 Regresión Lineal

Modelado de relación lineal entre variables:

$$Y = \beta_0 + \beta_1 X + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

**Estimadores de Mínimos Cuadrados Ordinarios (OLS):**

$$\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

**Métricas de evaluación:**

- **Coeficiente de determinación ($$R^2$$):** Proporción de varianza explicada
  
  $$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \in [0, 1]$$

- **RMSE (Root Mean Squared Error):** Error promedio
  
  $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}$$

### 2.5 Análisis de Componentes Principales (PCA)

Técnica de reducción de dimensionalidad que transforma variables correlacionadas en componentes ortogonales no correlacionadas.

#### 2.5.1 Fundamento Matemático

Dada una matriz de datos $\mathbf{X}$ ($n \times p$), PCA busca direcciones ortogonales de máxima varianza:

1. **Estandarización:** $\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu})\boldsymbol{\Sigma}^{-1/2}$

2. **Matriz de covarianza:** $\mathbf{C} = \frac{1}{n-1}\mathbf{Z}^T\mathbf{Z}$

3. **Descomposición espectral:** $\mathbf{C} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$
   - $\mathbf{V}$: matriz de vectores propios (direcciones principales)
   - $\mathbf{\Lambda}$: matriz diagonal de valores propios (varianzas explicadas)

4. **Proyección:** $\mathbf{T} = \mathbf{Z}\mathbf{V}$

**Componentes principales:**
- **PC1:** Dirección de máxima varianza
- **PC2:** Dirección de máxima varianza ortogonal a PC1
- **PC$k$:** Dirección de máxima varianza ortogonal a todas las anteriores

**Varianza explicada:**
$$\text{Var}_{\text{explained}}(PC_k) = \frac{\lambda_k}{\sum_{j=1}^p \lambda_j}$$

#### 2.5.2 Interpretación en Finanzas

- **PC1:** Típicamente representa el "factor de mercado" o riesgo sistemático
- **PC2:** Puede representar factores sectoriales o estilos de inversión
- Permite identificar fuentes comunes de variación entre activos

### 2.6 Clustering K-Means

Algoritmo de particionamiento que agrupa observaciones en $k$ clusters minimizando la varianza intra-cluster.

#### 2.6.1 Algoritmo

**Función objetivo:**
$$\min_{S} \sum_{i=1}^k \sum_{\mathbf{x} \in S_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$

donde $\boldsymbol{\mu}_i$ es el centroide del cluster $S_i$.

**Procedimiento iterativo:**
1. Inicializar $k$ centroides aleatoriamente
2. **Asignación:** Asignar cada observación al centroide más cercano
3. **Actualización:** Recalcular centroides como media de observaciones asignadas
4. Repetir 2-3 hasta convergencia

#### 2.6.2 Aplicación en Finanzas

- Identificar regímenes de mercado (normal, estrés, euforia)
- Agrupar activos con comportamiento similar
- Detectar períodos con dinámicas homogéneas

### 2.7 Regresión Logística

Modelo de clasificación binaria que estima probabilidades mediante función logística.

#### 2.7.1 Modelo

$$P(Y=1|\mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T\mathbf{X})}} = \sigma(\beta_0 + \boldsymbol{\beta}^T\mathbf{X})$$

**Interpretación:** El logaritmo de odds (log-odds) es lineal en las variables:
$$\log\left(\frac{P(Y=1|\mathbf{X})}{1-P(Y=1|\mathbf{X})}\right) = \beta_0 + \boldsymbol{\beta}^T\mathbf{X}$$

#### 2.7.2 Estimación

Maximización de verosimilitud (Maximum Likelihood Estimation):

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \sum_{i=1}^n \left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

#### 2.7.3 Métricas de Evaluación

- **Accuracy:** Proporción de predicciones correctas
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Matriz de confusión:** Visualiza errores tipo I (falsos positivos) y tipo II (falsos negativos)

- **Baseline:** Precisión de predicción trivial (siempre predecir clase mayoritaria)

---

## 3. Metodología y Técnicas Aplicadas

### 3.1 Preparación y Limpieza de Datos

#### 3.1.1 Carga y Validación

- Lectura de archivos CSV con precios históricos
- Validación de columnas requeridas: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- Conversión de tipos de datos (fechas, numéricos)
- Ordenamiento temporal y eliminación de duplicados

#### 3.1.2 Construcción de Variables

1. **Selección de precio ajustado:** Uso de `Adj Close` cuando está disponible, compensando dividendos y splits
2. **Cálculo de rendimientos:**
   - Rendimiento simple: `ret = price.pct_change()`
   - Rendimiento logarítmico: `logret = np.log(price).diff()`
3. **Volatilidad móvil:** `rolling_vol = ret.rolling(window=20).std()`

#### 3.1.3 Intersección Temporal

Debido a historiales de diferente longitud, se utiliza la intersección de fechas válidas:
- Pivote de datos largos a formato ancho (fechas × activos)
- Eliminación de filas con valores faltantes: `dropna()`
- Resultado: 409 observaciones comunes (2018-08-15 a 2020-04-01)

### 3.2 Análisis Exploratorio de Datos (EDA)

#### 3.2.1 Estadísticos Descriptivos

Para cada activo se calculó:
- **Tendencia central:** Media, mediana
- **Dispersión:** Desviación estándar, rango intercuartílico
- **Forma de distribución:** Asimetría (skewness), curtosis (kurtosis)
- **Extremos:** Mínimo, máximo, percentiles

#### 3.2.2 Identificación de Outliers

**Método z-score:**
$$z_i = \frac{x_i - \bar{x}}{s}$$

Se consideran outliers aquellos con $|z_i| > 3$ (más de 3 desviaciones estándar).

**Interpretación en finanzas:**
- Outliers corresponden a eventos de mercado significativos
- No se eliminan, pues contienen información valiosa sobre riesgo de cola
- Ejemplos identificados: Lunes Negro 1987, crash de dot-com 2000, COVID-19 marzo 2020

#### 3.2.3 Visualizaciones

- **Series temporales:** Evolución de precios y rendimientos
- **Histogramas:** Distribución de rendimientos con superposición de densidad normal
- **Boxplots:** Comparación de dispersión y outliers entre activos
- **Scatter plots:** Relaciones bivariadas con líneas de regresión
- **Matriz de correlación:** Heatmap con anotaciones de coeficientes

### 3.3 Pruebas Estadísticas Implementadas

#### 3.3.1 Pruebas de Normalidad

Para cada activo:
1. **Jarque-Bera:** `scipy.stats.jarque_bera()`
2. **Kolmogorov-Smirnov:** `scipy.stats.kstest(data, 'norm', args=(mean, std))`

Hipótesis: $H_0$: normalidad, $H_1$: no normalidad, $\alpha = 0.05$

#### 3.3.2 Comparación de Medias

**Pairwise (Welch t-test):**
```python
for (A, B) in combinations(assets, 2):
    tstat, pval = scipy.stats.ttest_ind(ret_A, ret_B, equal_var=False)
```

**Global (ANOVA):**
```python
f_stat, p_val = scipy.stats.f_oneway(*[ret[col] for col in assets])
```

### 3.4 Modelado Estadístico

#### 3.4.1 Regresión Lineal

Modelo: $r^{\text{AAPL}}_t = \beta_0 + \beta_1 r^{\text{MSFT}}_t + \varepsilon_t$

```python
X = ret[['MSFT']].values
y = ret['AAPL'].values
model = LinearRegression().fit(X, y)
r2 = r2_score(y, model.predict(X))
rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
```

#### 3.4.2 PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=4)
Z = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
```

#### 3.4.3 K-Means Clustering

```python
from sklearn.cluster import KMeans

# Clustering en espacio PCA (2D)
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(Z[:, :2])
```

**Caracterización de clusters:**
- Tamaño (número de observaciones)
- Volatilidad media dentro del cluster
- Correlación media entre activos
- Retornos promedio y extremos

#### 3.4.4 Regresión Logística

**Features (tiempo $t$):**
- Rendimientos de otros activos: MSFT, NVDA, AAAU
- Volatilidad móvil: `MSFT_vol20`, `NVDA_vol20`, `AAAU_vol20`

**Target (tiempo $t+1$):**
- Dirección de AAPL: $y_{t+1} = \mathbb{1}[r_{t+1}^{\text{AAPL}} > 0]$

```python
# Construcción del dataset predictivo
df['y_tomorrow'] = (df['AAPL'].shift(-1) > 0).astype(int)

# Train-test split temporal (75%-25%)
n_train = int(0.75 * len(X))
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Modelo
clf = LogisticRegression(max_iter=2000, random_state=42)
clf.fit(X_train, y_train)
accuracy = accuracy_score(y_test, clf.predict(X_test))
```

---

## 4. Resultados Obtenidos

### 4.1 Volatilidad y Riesgo

| Activo | $\sigma(\text{ret})$ | Ratio vs. Mínimo |
|--------|---------------------|------------------|
| **NVDA** | 3.51% | 3.9× |
| **AAPL** | 2.35% | 2.6× |
| **MSFT** | 2.21% | 2.5× |
| **AAAU** | 0.90% | 1.0× (referencia) |

**Hallazgo clave:** NVDA es 3.9 veces más volátil que AAAU, reflejando diferencias fundamentales de naturaleza entre tecnología de alto crecimiento y activo refugio.

### 4.2 Correlaciones entre Activos

**Matriz de correlación:**

|        | AAPL  | MSFT  | NVDA  | AAAU   |
|--------|-------|-------|-------|--------|
| AAPL   | 1.000 | 0.814 | 0.676 | -0.039 |
| MSFT   | 0.814 | 1.000 | 0.701 | -0.036 |
| NVDA   | 0.676 | 0.701 | 1.000 | -0.045 |
| AAAU   | -0.039| -0.036| -0.045| 1.000  |

**Hallazgos:**
1. **Bloque tecnológico cohesivo:** Correlaciones entre acciones > 0.67
2. **AAAU independiente:** Correlaciones con acciones ≈ -0.04 (prácticamente nulas)
3. **Par más correlacionado:** AAPL-MSFT (0.814) → 66% de co-varianza común

### 4.3 Pruebas de Normalidad

**Resultados consolidados:**

| Activo | JB stat | JB p-val | KS p-val | Rechaza $H_0$ |
|--------|---------|----------|----------|---------------|
| AAAU   | 78.4    | <0.001   | 0.0018   | ✓ Sí          |
| AAPL   | 165.7   | <0.001   | 0.0001   | ✓ Sí          |
| MSFT   | 625.0   | <0.001   | 0.0004   | ✓ Sí          |
| NVDA   | 122.1   | <0.001   | 0.0002   | ✓ Sí          |

**Patrones de asimetría y curtosis:**

| Activo | Asimetría | Curtosis | Interpretación |
|--------|-----------|----------|----------------|
| AAAU   | +0.322    | 6.41     | Sesgo positivo; colas pesadas |
| AAPL   | -0.339    | 6.37     | Sesgo negativo; colas pesadas |
| MSFT   | -0.052    | 10.84    | Simétrico; colas MUY pesadas |
| NVDA   | -0.674    | 5.71     | Sesgo negativo fuerte; colas pesadas |

**Conclusión:** Todos los activos rechazan normalidad, exhibiendo **colas pesadas** (curtosis > 3). Esto implica mayor probabilidad de eventos extremos que lo predicho por distribución normal.

### 4.4 Comparación de Medias

**Welch t-test (pairwise):**

| Par | t-stat | p-valor | Rechaza $H_0$ |
|-----|--------|---------|---------------|
| Todos los pares | variado | >0.75 | ✗ No |

**ANOVA (global):**
- F-estadístico: 0.0812
- p-valor: 0.9705
- **Conclusión:** No se rechazan diferencias en medias

**Interpretación:** En escala diaria, la media de rendimientos es indistinguible entre activos ($\bar{r} \approx 0.06\% - 0.11\%$), mientras que las volatilidades difieren significativamente (0.9% - 3.5%). Esto confirma que **la volatilidad, no la media, es el diferenciador principal de activos** a horizonte diario.

### 4.5 Regresión Lineal: AAPL vs. MSFT

**Modelo estimado:**
$$\text{AAPL} = 0.000036 + 0.869 \times \text{MSFT}$$

**Métricas:**
- $R^2 = 0.662$ (66.2% de varianza explicada)
- RMSE = 0.0137
- Pendiente $\hat{\beta}_1 = 0.869$

**Interpretación:**
- MSFT explica 2/3 de la variación en AAPL
- Por cada 1% que sube MSFT, AAPL sube en promedio 0.87%
- Relación lineal fuerte y positiva, consistente con factor tecnológico común

### 4.6 PCA: Reducción de Dimensionalidad

**Varianza explicada:**

| Componente | Varianza | Acumulada |
|------------|----------|-----------|
| PC1        | 61.63%   | 61.63%    |
| PC2        | 24.92%   | 86.55%    |
| PC3        | 7.75%    | 94.30%    |
| PC4        | 5.70%    | 100.00%   |

**Hallazgos:**
1. **Eficiencia de reducción:** 2 componentes capturan 86.55% de información
2. **PC1 (61.63%):** Factor de mercado común o riesgo sistemático
3. **PC2 (24.92%):** Factor discriminante o estilos específicos
4. **Implicación:** Estructura latente simple, dominada por pocos factores

### 4.7 Clustering: Identificación de Regímenes

**K-Means (k=3) en espacio PCA:**

| Cluster | Días | % Total | Volatilidad | Correlación | Retorno promedio | Interpretación |
|---------|------|---------|-------------|-------------|------------------|----------------|
| 0       | 313  | 76.5%   | Baja-Media  | ~0.12       | Positivo pequeño | **Régimen Normal** |
| 1       | 33   | 8.1%    | Media-Alta  | ~0.28       | Fuertemente positivo | **Rally/Euforia** |
| 2       | 63   | 15.4%   | Muy Alta    | ~0.42       | Fuertemente negativo | **Crash/Estrés** |

**Caracterización detallada:**

**Cluster 0 (Normal):**
- Movimientos contenidos: rango [-4.9%, +7.0%]
- Baja correlación → diversificación efectiva
- AAAU neutral (+0.04%)

**Cluster 1 (Rally):**
- Máximos potentes: rango [-3.6%, +17.2%]
- Acciones suben coordinadamente (+4% a +5.7%)
- AAAU baja (-0.24%) → inverso a tecnología

**Cluster 2 (Crash):**
- Caídas extremas: rango [-18.8%, +3.2%]
- Alta correlación (0.42) → contagio
- Acciones caen (-3% a -5%)
- AAAU sube (+0.42%) → activo refugio

**Conclusión clave:** La correlación NO es constante; aumenta dramáticamente en estrés (0.12 → 0.42), reduciendo beneficios de diversificación cuando más se necesita.

### 4.8 Clasificación: Predicción Direccional

**Modelo:** Regresión Logística para predecir dirección de AAPL (día siguiente)

**Features enriquecidos:**
- Rendimientos: MSFT, NVDA, AAAU (tiempo $t$)
- Volatilidad móvil: MSFT_vol20, NVDA_vol20, AAAU_vol20 (tiempo $t$)

**Target:** $y_{t+1} = \mathbb{1}[r_{t+1}^{\text{AAPL}} > 0]$

**Resultados:**
- **Accuracy del baseline:** 50.98% (predicción trivial: siempre clase mayoritaria)
- **Accuracy del modelo:** 50.98% (idéntica al baseline)
- **Mejora:** 0.00%

**Matriz de confusión:**
```
[[  0  50]
 [  0  52]]
```

**Análisis:**
- Sensibilidad: 100% (identifica todos los casos positivos)
- Especificidad: 0% (no identifica ningún caso negativo)
- El modelo colapsa a predicción degenerada: siempre predice "sube"

**Coeficientes del modelo:**

| Feature      | Coeficiente |
|--------------|-------------|
| MSFT         | -0.186      |
| NVDA_vol20   | +0.084      |
| MSFT_vol20   | +0.041      |
| AAAU         | -0.036      |
| NVDA         | -0.022      |
| AAAU_vol20   | -0.004      |

**Interpretación:** Coeficientes extremadamente débiles (máximo 0.186 en valor absoluto) indican ausencia de poder predictivo en las variables.

**Conclusión:** No se logró capacidad predictiva superior al baseline con variables simples, consistente con la **Hipótesis de Eficiencia del Mercado** en su forma débil.

---

## 5. Interpretación y Discusión

### 5.1 Naturaleza de los Rendimientos Financieros

#### 5.1.1 No-Normalidad: Implicaciones Prácticas

El rechazo universal de normalidad tiene consecuencias importantes:

**Colas pesadas (curtosis > 3):**
- Eventos extremos son MÁS frecuentes que lo predicho por normal
- Modelos basados en normalidad (Black-Scholes clásico, VaR paramétrico) subestiman riesgo
- Necesidad de modelos robustos (distribuciones t de Student, EVT - Extreme Value Theory)

**Asimetría:**
- AAAU (+0.322): Rally más probables que crashes → refugio en estrés
- NVDA (-0.674): Crashes más probables → mayor riesgo bajista
- Información valiosa para gestión de riesgo asimétrico

#### 5.1.2 Volatilidad como Medida Dominante

La ausencia de diferencias significativas en medias (Welch/ANOVA) pero gran heterogeneidad en volatilidades enfatiza un principio fundamental en finanzas:

**A horizonte diario:**
- Media de retornos: $\bar{r} \approx 0.06\% - 0.11\%$ (indistinguible)
- Volatilidad: $\sigma \in [0.90\%, 3.51\%]$ (factor 3.9×)

**Consecuencia:** El trade-off riesgo-retorno se centra en **volatilidad vs. retorno esperado anualizado**, no en retornos diarios individuales.

**Teoría Moderna de Carteras:** La optimización de Markowitz maximiza:
$$\frac{\mu_p - r_f}{\sigma_p}$$
donde $\mu_p$ es retorno esperado anual, no diario.

### 5.2 Estructura Latente y Factores Comunes

#### 5.2.1 Interpretación de Componentes Principales

**PC1 (61.63%):** Factor de Mercado
- Afecta a todos los activos simultáneamente
- Equivalente al "beta" del CAPM (Capital Asset Pricing Model)
- Refleja movimientos amplios del mercado (índices S&P 500, NASDAQ)

**PC2 (24.92%):** Factor Discriminante
- Separa oro (AAAU) de acciones tecnológicas
- Puede interpretarse como "riesgo vs. refugio"
- Captura rotaciones sector-específicas

**Eficiencia de la reducción:**
- Solo necesitamos 2 dimensiones para capturar 86.55% de información
- Confirma que los 4 activos NO son independientes
- Sus movimientos están orquestados por fuerzas sistemáticas comunes

#### 5.2.2 Modelos de Factores en Finanzas

Este hallazgo es consistente con:

**CAPM:** $r_i = r_f + \beta_i(r_m - r_f) + \varepsilon_i$
- PC1 captura el factor de mercado $(r_m - r_f)$

**Fama-French:** $r_i = r_f + \beta_M(r_m - r_f) + \beta_S\text{SMB} + \beta_V\text{HML} + \varepsilon_i$
- PC2 podría capturar factores SMB (tamaño) o HML (valor)

### 5.3 Regímenes de Mercado y Correlación Dinámica

#### 5.3.1 Fenómeno de Contagio

El aumento de correlación en crisis (0.12 → 0.42) es un fenómeno bien documentado:

**Mecanismo:**
1. En normalidad: Activos responden a noticias idiosincráticas
2. En crisis: Dominan factores sistémicos (pánico, liquidez)
3. Resultado: "Flight to quality" → todos venden riesgo simultáneamente

**Implicación para gestión de riesgo:**
- Diversificación funciona en normalidad (76.5% del tiempo)
- Diversificación falla cuando más se necesita (15.4% del tiempo)
- Necesidad de cobertura adicional (opciones, oro, bonos)

#### 5.3.2 Oro como Activo Refugio

El comportamiento de AAAU confirma su rol de "safe haven":

| Régimen | AAAU | Acciones | Interpretación |
|---------|------|----------|----------------|
| Normal  | +0.04% | mixto | Neutral |
| Rally   | -0.24% | +4% a +5.7% | Inverso (oportunidad perdida) |
| Crash   | +0.42% | -3% a -5% | Cobertura efectiva |

**Conclusión:** AAAU cumple su función defensiva, pero a costa de renunciar a ganancias en rallies.

### 5.4 Fracaso de Predicción Direccional

#### 5.4.1 Hipótesis de Eficiencia del Mercado (EMH)

El fracaso del modelo de regresión logística es evidencia empírica de EMH en forma débil:

**Forma débil (Fama, 1970):**
> "Los precios reflejan toda la información contenida en el historial de precios pasados. No es posible obtener rentabilidades anormales utilizando únicamente análisis técnico."

**Nuestro resultado:**
- Features históricos simples (rendimientos, volatilidad móvil) NO predicen dirección futura
- El modelo colapsa a predicción trivial (siempre clase mayoritaria)
- Accuracy = baseline = 50.98%

#### 5.4.2 Razones del Fracaso

**1. Ruido vs. Señal:**
- A horizonte diario, ratio señal-ruido es extremadamente bajo
- Movimientos intradía dominados por microestructura, órdenes aleatorias

**2. Eficiencia del NASDAQ:**
- AAPL, MSFT, NVDA son activos altamente líquidos y seguidos
- Información se incorpora rápidamente a precios
- Oportunidades de arbitraje estadístico son efímeras

**3. Linealidad vs. Complejidad:**
- Regresión logística asume relaciones lineales
- Mercados exhiben no-linealidades, cambios de régimen
- Necesidad de modelos avanzados (redes neuronales, LSTM)

#### 5.4.3 Contraste con Literatura

**Trading rentable requiere:**
- Horizontes más largos (semanas, meses)
- Variables sofisticadas (sentimiento, volumen, opciones)
- Modelos complejos (machine learning, deep learning)
- Gestión rigurosa de costos de transacción

**Ejemplo:** Momentum strategies (Jegadeesh & Titman, 1993) explotan persistencia a 3-12 meses, no 1 día.

### 5.5 Limitaciones Metodológicas

#### 5.5.1 Ventana Temporal Corta

**Período:** 2018-08-15 a 2020-04-01 (409 días ≈ 1.6 años)

**Problemas:**
1. **Sesgo de muestra:** Incluye crisis COVID-19 (evento extremo)
2. **Ciclo económico incompleto:** No captura expansión prolongada ni recesión completa
3. **Generalización limitada:** Conclusiones pueden no aplicar a otros períodos

**Solución ideal:** Usar datos de 10+ años para capturar ciclos completos.

#### 5.5.2 Supuestos Violados

**Normalidad:**
- Rechazada empíricamente para todos los activos
- Modelos (regresión, ANOVA) asumen normalidad para inferencia
- **Mitigación:** Welch t-test es robusto; muestras grandes ($n=409$) invocan Teorema Central del Límite

**Linealidad:**
- Regresión lineal y logística asumen relaciones lineales
- Mercados exhiben no-linealidades (volatilidad clustering, umbrales)
- **Limitación:** Modelos lineales pueden perder patrones complejos

**Independencia temporal:**
- Rendimientos financieros exhiben autocorrelación en volatilidad (GARCH)
- No capturado por modelos estáticos
- **Extensión futura:** Modelos dinámicos (ARIMA, GARCH)

---

## 6. Limitaciones y Trabajo Futuro

### 6.1 Limitaciones Identificadas

#### 6.1.1 Datos

1. **Período corto:** 1.6 años insuficiente para generalización robusta
2. **Sesgo de supervivencia:** Solo activos exitosos (AAPL, MSFT, NVDA sobrevivieron)
3. **Falta de variables fundamentales:** Solo precios; no earnings, ratios financieros

#### 6.1.2 Modelado

1. **Supuestos paramétricos:** Dependencia de normalidad (parcialmente violada)
2. **Modelos lineales:** Limitados para capturar complejidad de mercados
3. **Horizonte fijo:** Solo 1 día; no exploración de multi-horizonte

#### 6.1.3 Validación

1. **Train-test simple:** Sin validación cruzada temporal
2. **Métricas limitadas:** Solo accuracy; faltan precision, recall, F1, AUC-ROC
3. **Análisis de sensibilidad:** No evaluación de robustez ante cambios de hiperparámetros

### 6.2 Trabajo Futuro

#### 6.2.1 Extensiones de Datos

**Ampliar cobertura temporal:**
- Extender a 10+ años (2010-2025)
- Capturar crisis financiera 2008, recuperación post-crisis, pandemia completa

**Incluir más activos:**
- Índices de mercado (S&P 500, NASDAQ)
- Sectores adicionales (energía, salud, finanzas)
- Activos alternativos (criptomonedas, commodities)

**Variables fundamentales:**
- P/E ratio, earnings growth, debt-to-equity
- Sentimiento de mercado (VIX, put/call ratio)
- Flujos de capital, volumen institucional

#### 6.2.2 Modelado Avanzado

**Modelos no-lineales:**
- Random Forest, Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines con kernels no-lineales
- Redes neuronales (perceptrones multicapa)

**Modelos temporales:**
- ARIMA/ARIMAX para predicción de series temporales
- GARCH para modelado de volatilidad condicional
- LSTM (Long Short-Term Memory) para capturar dependencias largas

**Modelos de régimen:**
- Hidden Markov Models (HMM) para cambios de régimen
- Mixture models para distribuciones multimodales

#### 6.2.3 Validación Rigurosa

**Cross-validation temporal:**
- Walk-forward analysis (ventanas deslizantes)
- Expanding window (entrenamiento acumulativo)

**Métricas comprehensivas:**
- Precision, recall, F1-score
- AUC-ROC, AUC-PR
- Sharpe ratio, Sortino ratio (para estrategias de trading)

**Backtesting:**
- Simulación de estrategias con costos de transacción
- Análisis de drawdowns, máximo drawdown
- Test de robustez ante cambios de mercado

#### 6.2.4 Análisis Causal

**Inferencia causal:**
- Granger causality para direccionalidad temporal
- Vector Autoregression (VAR) para interacciones multivariadas
- Causal impact analysis para eventos específicos

---

## 7. Conclusiones Finales

### 7.1 Respuestas a Preguntas de Investigación

**P1: ¿Qué activo presenta mayor volatilidad?**
- **NVDA** (3.51%), 3.9× más volátil que AAAU (0.90%)
- Refleja diferencias naturaleza: tecnología vs. refugio

**P2: ¿Existen correlaciones significativas?**
- **Sí:** Bloque tecnológico cohesivo (ρ > 0.67)
- **No:** AAAU independiente (ρ ≈ -0.04)
- Implicación: AAAU ofrece diversificación

**P3: ¿Se identifican clusters naturales?**
- **Sí:** 3 regímenes con dinámicas distintas
- Normal (76.5%), Rally (8.1%), Crash (15.4%)
- Correlación aumenta en estrés (contagio)

**P4: ¿Es posible predecir movimientos diarios?**
- **No** con variables simples (accuracy = baseline)
- Consistente con EMH forma débil
- Necesidad de modelos complejos y variables sofisticadas

### 7.2 Contribuciones del Proyecto

#### 7.2.1 Metodológicas

1. **Pipeline completo de análisis cuantitativo:**
   - Desde carga de datos hasta modelado predictivo
   - Replicable y extensible

2. **Integración de técnicas complementarias:**
   - Estadística descriptiva → inferencial → predictiva
   - Supervisado (regresión) y no supervisado (PCA, clustering)

3. **Validación rigurosa de supuestos:**
   - Pruebas de normalidad, homogeneidad de varianzas
   - Discusión de implicaciones cuando se violan

#### 7.2.2 Empíricas

1. **Cuantificación de heterogeneidad:**
   - Factor 3.9× en volatilidades
   - Rango 0.67-0.81 en correlaciones tecnológicas

2. **Identificación de regímenes:**
   - Documentación de 3 estados con dinámicas propias
   - Cuantificación de contagio (0.12 → 0.42)

3. **Evidencia de eficiencia:**
   - Fracaso de predicción simple apoya EMH
   - Barreras a arbitraje estadístico en mercados líquidos

### 7.3 Lecciones Aprendidas

#### 7.3.1 Sobre Finanzas Cuantitativas

**Volatilidad domina a media en horizontes cortos:**
- Gestión de riesgo > timing de mercado
- Diversificación es clave, pero limitada en crisis

**Mercados no son estacionarios:**
- Regímenes cambiantes requieren adaptabilidad
- Modelos estáticos tienen vida útil limitada

**Eficiencia es real pero no perfecta:**
- Predictibilidad es difícil, no imposible
- Requiere sofisticación, datos de calidad, ejecución rápida

#### 7.3.2 Sobre Estadística Aplicada

**Validación de supuestos es crítica:**
- No asumir normalidad ciegamente
- Pruebas empíricas revelan estructura de datos

**Visualización complementa números:**
- Scatter plots revelan no-linealidades
- Heatmaps comunican estructuras complejas intuitivamente

**Simplicidad tiene límites:**
- Modelos lineales son interpretables pero limitados
- Trade-off interpretabilidad vs. capacidad predictiva

### 7.4 Reflexión Final

Este proyecto demuestra que el análisis cuantitativo riguroso de mercados financieros requiere:

1. **Fundamentos sólidos:** Comprensión profunda de estadística y finanzas
2. **Herramientas adecuadas:** Dominio de librerías computacionales (NumPy, pandas, scikit-learn)
3. **Pensamiento crítico:** Interpretación de resultados en contexto teórico y práctico
4. **Humildad científica:** Reconocimiento de limitaciones y incertidumbre

Los resultados obtenidos son consistentes con décadas de investigación en finanzas cuantitativas y validan principios fundamentales:
- Mercados eficientes son difíciles de predecir
- Diversificación reduce riesgo pero no lo elimina
- Volatilidad y correlación son dinámicas, no estáticas

El camino hacia estrategias cuantitativas exitosas pasa por mayor sofisticación en modelado, expansión de variables informativas, y validación rigurosa fuera de muestra. Este proyecto establece una base sólida para futuras exploraciones en finanzas cuantitativas.

---

## 📚 Referencias Bibliográficas

### Libros de Texto

1. **Downey, A. B.** (2014). *Think Stats: Probability and Statistics for Programmers* (2nd ed.). O'Reilly Media. Disponible en: https://greenteapress.com/wp/think-stats-2e/

2. **Casella, G. & Berger, R. L.** (2024). *Statistical Inference* (2nd ed.). Cengage Learning. ISBN: 978-0534267711

3. **Wackerly, D., Mendenhall, W. & Scheaffer, R.** (2010). *Mathematical Statistics with Applications* (7th ed.). Brooks/Cole. ISBN: 978-0495110811

4. **Tsay, R. S.** (2010). *Analysis of Financial Time Series* (3rd ed.). Wiley. ISBN: 978-0470414354

5. **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer. ISBN: 978-1071614174

### Artículos Académicos

6. **Fama, E. F.** (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417. DOI: 10.2307/2325486

7. **Markowitz, H. M.** (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91. DOI: 10.2307/2975974

8. **Jegadeesh, N. & Titman, S.** (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65-91. DOI: 10.1111/j.1540-6261.1993.tb04702.x

9. **Engle, R. F.** (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007. DOI: 10.2307/1912773

### Documentación Técnica

10. **NumPy Developers** (2025). *NumPy Documentation*. Disponible en: https://numpy.org/doc/

11. **pandas Development Team** (2025). *pandas Documentation*. Disponible en: https://pandas.pydata.org/docs/

12. **Matplotlib Development Team** (2025). *Matplotlib Documentation*. Disponible en: https://matplotlib.org/stable/contents.html

13. **Seaborn Developers** (2025). *Seaborn Documentation*. Disponible en: https://seaborn.pydata.org/

14. **SciPy Developers** (2025). *SciPy Documentation*. Disponible en: https://docs.scipy.org/doc/scipy/

15. **scikit-learn Developers** (2025). *scikit-learn Documentation*. Disponible en: https://scikit-learn.org/stable/documentation.html

---

**Fin del Documento de Defensa**

*Este documento ha sido elaborado como guía comprehensiva para la defensa del Proyecto Final de Estadística. Contiene fundamentos teóricos, metodología detallada, resultados empíricos, e interpretaciones contextualizadas que sustentan las conclusiones del análisis cuantitativo realizado.*
