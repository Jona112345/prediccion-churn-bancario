# Predicción de Abandono de Clientes Bancarios

Sistema de Machine Learning para predecir qué clientes bancarios tienen alta probabilidad de cancelar sus servicios, permitiendo implementar estrategias proactivas de retención.

## 🎯 Resultados Principales

- **AUC Score:** 0.88.06
- **Precisión:** 87.5%
- **Recall:** 81.3%
- **F1-Score:** 84.2%
- **ROI Estimado:** 11,791.5%
- **Beneficio Anual Proyectado:** €1,173,250

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**
- **LightGBM** - Algoritmo principal de clasificación
- **scikit-learn** - Preprocesamiento y evaluación
- **Streamlit** - Aplicación web interactiva
- **Plotly** - Visualizaciones avanzadas
- **pandas & numpy** - Manipulación de datos
- **Jupyter Notebook** - Desarrollo y análisis

## 📁 Estructura del Proyecto

```
prediccion-churn-bancario/
│
├── README.md                          # Documentación del proyecto
├── requirements.txt                   # Dependencias
├── .gitignore                        # Archivos ignorados por Git
├── LICENSE                           # Licencia MIT
│
├── data/
│   └── bank_customer_churn.csv      # Dataset original
│
├── notebooks/
│   └── analisis_churn_bancario.ipynb # Análisis y entrenamiento completo
│
├── models/
│   ├── modelo_abandono_bancario_20250922.pkl     # Modelo entrenado
│   └── escalador_caracteristicas_20250922.pkl    # Escalador de features
│
└── deployment/
    └── app.py                        # Aplicación web Streamlit
```

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.9+
- Git

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Jona112345/prediccion-churn-bancario.git
cd prediccion-churn-bancario

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar la Aplicación Web

```bash
# Ejecutar aplicación interactiva
streamlit run deployment/app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`

### Ejecutar el Análisis Completo

```bash
# Abrir Jupyter Notebook
jupyter notebook notebooks/analisis_churn_bancario.ipynb
```

## 📊 Características de la Aplicación

### Dashboard Interactivo
- **Predicción en tiempo real** de probabilidad de churn
- **Visualizaciones avanzadas** con gráficos interactivos
- **Sistema de alertas** codificado por colores según riesgo
- **Recomendaciones personalizadas** por nivel de riesgo

### Simulador Avanzado
- **Análisis de sensibilidad** de variables
- **Escenarios What-If** (optimista/pesimista)
- **Proyección temporal** de riesgo a 12 meses
- **Métricas de impacto empresarial** en tiempo real

### Características Técnicas
- **Algoritmo LightGBM optimizado** con GridSearchCV
- **8+ factores de riesgo** analizados
- **Interface futurista** con glassmorphism design
- **Responsive design** adaptable a cualquier pantalla

## 📈 Metodología

### 1. Análisis Exploratorio de Datos (EDA)
- Análisis de distribuciones y correlaciones
- Identificación de patrones de churn
- Detección de valores atípicos y outliers

### 2. Preprocesamiento
- Limpieza y tratamiento de valores faltantes
- Encoding de variables categóricas
- Feature engineering avanzado
- Escalado de características numéricas

### 3. Modelado y Optimización
- **Comparación de algoritmos:** LightGBM, Random Forest, SVM, KNN
- **Optimización de hiperparámetros** con GridSearchCV
- **Validación cruzada** estratificada
- **Evaluación con métricas de negocio**

### 4. Evaluación del Modelo
- Métricas de clasificación completas
- Curva ROC y análisis AUC
- Matriz de confusión
- Análisis de feature importance

## 💼 Impacto Empresarial

### Problema Empresarial
- **Tasa de churn actual:** 20.4%
- **Costo de adquisición:** €200 por cliente
- **Valor promedio de cliente:** €1,200 anuales

### Solución Implementada
- **Clientes detectados:** 986 de 2,037 (48.4% recall)
- **Costo de retención:** €50 por cliente identificado
- **Efectividad estimada:** 70% de clientes retenidos

### Resultados Financieros
- **Ingresos salvados:** €1,183,200 anuales
- **Costo de implementación:** €9,950 anuales
- **Beneficio neto:** €1,173,250 anuales
- **ROI:** 11,791.5%

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Jonathan Ibáñez**
- Data Scientist especializado en ML y DL aplicado a problemas empresariales
- Email: jonathan_herraiz@yahoo.es
- LinkedIn: [linkedin.com/in/jonathan-ibañez-33896a1b2](https://www.linkedin.com/in/jonathan-ibañez-33896a1b2)
- GitHub: [github.com/Jona112345](https://github.com/Jona112345)

---

⭐ **Si este proyecto te resulta útil, no olvides darle una estrella en GitHub!**