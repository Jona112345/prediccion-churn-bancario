# Predicción de Abandono de Clientes Bancarios

Sistema avanzado de Machine Learning para predecir la probabilidad de abandono (churn) de clientes bancarios, permitiendo implementar estrategias proactivas de retención y maximizar el valor de la cartera de clientes.

## Resultados Principales

- **AUC Score:** 0.88.06
- **Precisión:** 87.67%
- **Recall:** 81.3%  
- **F1-Score:** 84.2%
- **ROI Estimado:** 11,791.5%
- **Beneficio Anual Proyectado:** €1,173,250
- **Tasa de Detección:** 48.4% de clientes en riesgo identificados correctamente

## Tecnologías Utilizadas

### Machine Learning
- **LightGBM** - Algoritmo principal optimizado con GridSearchCV
- **scikit-learn** - Preprocesamiento, evaluación y validación
- **pandas & numpy** - Manipulación y análisis de datos
- **matplotlib & seaborn** - Visualización exploratoria

### Aplicación Web Interactiva  
- **Streamlit** - Framework de aplicación web
- **Plotly** - Visualizaciones interactivas avanzadas
- **Diseño futurista** - Interface moderna con glassmorphism

### Entorno de Desarrollo
- **Python 3.9+**
- **Jupyter Notebook** - Análisis y prototipado
- **Git & GitHub** - Control de versiones

## Estructura del Proyecto

```
prediccion-churn-bancario/
│
├── README.md                                    # Documentación principal
├── requirements.txt                             # Dependencias del proyecto
├── .gitignore                                  # Archivos ignorados por Git
├── LICENSE                                     # Licencia MIT
│
├── data/
│   └── bank_customer_churn.csv                # Dataset original (10,000 registros)
│
├── notebooks/
│   └── analisis_churn_bancario.ipynb          # Análisis completo y entrenamiento
│
├── models/
│   ├── modelo_abandono_bancario_20250922.pkl  # Modelo LightGBM entrenado
│   └── escalador_caracteristicas_20250922.pkl # Escalador StandardScaler
│
└── deployment/
    └── app.py                                  # Aplicación web Streamlit
```

## Instalación y Uso

### Prerrequisitos
```bash
Python 3.9+
Git
```

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/Jona112345/prediccion-churn-bancario.git
cd prediccion-churn-bancario

# Crear y activar entorno virtual
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar Aplicación Web

```bash
streamlit run deployment/app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Ejecutar Análisis Completo

```bash
jupyter notebook notebooks/analisis_churn_bancario.ipynb
```

## Características de la Aplicación

### Dashboard Interactivo
- **Predicción en tiempo real** con algoritmo LightGBM optimizado
- **Sistema de alertas** codificado por colores según nivel de riesgo
- **Métricas de negocio** calculadas dinámicamente
- **Interface futurista** con efectos visuales modernos

### Simulador Avanzado
- **Análisis de sensibilidad** - Impacto de cada variable en la predicción
- **Escenarios What-If** - Comparación optimista vs pesimista  
- **Proyección temporal** - Evolución del riesgo a 12 meses
- **Recomendaciones inteligentes** adaptadas por nivel de riesgo

### Visualizaciones Interactivas
- **Gauge dinámico** de probabilidad de churn
- **Gráficos de impacto** por factor de riesgo
- **Tendencias temporales** con y sin intervención
- **Métricas empresariales** en tiempo real

## Metodología Técnica

### 1. Análisis Exploratorio (EDA)
- Análisis de 10,000 registros de clientes bancarios
- Identificación de patrones de comportamiento
- Detección de correlaciones y outliers
- Segmentación por variables demográficas y financieras

### 2. Preprocesamiento Avanzado
- **Feature Engineering:** Creación de variables derivadas (ratios financieros, índices compuestos)
- **Encoding:** LabelEncoder para variables categóricas
- **Escalado:** StandardScaler para algoritmos sensibles a escala
- **Validación:** División estratificada 70/30 entrenamiento/prueba

### 3. Modelado y Optimización
- **Comparación de algoritmos:** LightGBM, Random Forest, SVM, KNN
- **Optimización de hiperparámetros:** GridSearchCV con validación cruzada 5-fold
- **Métricas de evaluación:** AUC, Precisión, Recall, F1-Score
- **Selección final:** LightGBM por mejor AUC (0.8791)

### 4. Validación y Interpretabilidad
- **Análisis de feature importance** - Top 10 variables más predictivas
- **Curva ROC** y análisis de umbral óptimo
- **Matriz de confusión** detallada
- **Análisis de casos límite** y falsos positivos/negativos

## Impacto Empresarial

### Contexto del Negocio
- **Base de clientes:** 10,000 clientes analizados
- **Tasa de churn actual:** 20.4% anual
- **Clientes que abandonan:** 2,037 anuales
- **Valor promedio por cliente:** €1,200 anuales

### Resultados del Modelo
- **Clientes de alto riesgo identificados:** 199 (6.6%)
- **Clientes de riesgo medio:** 290 (9.7%)
- **Clientes de bajo riesgo:** 2,511 (83.7%)
- **Tasa de detección:** 48.4% de abandonos predichos correctamente

### Impacto Financiero
- **Ingresos salvados:** €1,183,200 anuales
- **Costo de campañas de retención:** €9,950 anuales  
- **Beneficio neto:** €1,173,250 anuales
- **ROI:** 11,791.5%
- **Payback period:** < 1 mes

### Estrategias de Retención
- **Alto riesgo (>70%):** Contacto inmediato, ofertas exclusivas, gestión personalizada
- **Riesgo medio (40-70%):** Monitoreo estrecho, mejora de servicios, engagement proactivo  
- **Bajo riesgo (<40%):** Mantenimiento, oportunidades de upselling, fidelización

## Estructura Técnica de Archivos

### Notebooks
- **Análisis exploratorio** completo con visualizaciones
- **Comparación de modelos** y métricas detalladas  
- **Optimización de hiperparámetros** documentada
- **Evaluación de negocio** con cálculos financieros

### Modelos Guardados
- **modelo_abandono_bancario_20250922.pkl** - LightGBM optimizado (178KB)
- **escalador_caracteristicas_20250922.pkl** - StandardScaler ajustado (1.2KB)

### Aplicación Web
- **Interface moderna** con CSS personalizado y efectos visuales
- **Lógica de predicción** integrada con modelos entrenados
- **Simuladores interactivos** para análisis de escenarios
- **Sistema de recomendaciones** automático por riesgo

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Autor

**Jonathan Ibáñez**
- **Especialización:** Data Scientist enfocado en ML aplicado a problemas empresariales
- **Email:** jonathan_herraiz@yahoo.es  
- **LinkedIn:** [linkedin.com/in/jonathan-ibañez-33896a1b2](https://www.linkedin.com/in/jonathan-ibañez-33896a1b2)
- **GitHub:** [github.com/Jona112345](https://github.com/Jona112345)

---

**¿Te resulta útil este proyecto?** ⭐ ¡No olvides darle una estrella en GitHub para apoyar el desarrollo de más soluciones de Machine Learning aplicado a negocios!
