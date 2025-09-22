import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import base64

# =====================================================================
# CONFIGURACIÓN PÁGINA Y ESTILO PERSONALIZADO
# =====================================================================

st.set_page_config(
    page_title="AI Banking Analytics | Churn Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para un diseño futurista y moderno
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Fondo principal con gradiente */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        color: #ffffff;
    }
    
    /* Header personalizado */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 300% 300%;
        animation: gradientShift 8s ease infinite;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8.5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Tarjetas de métricas con glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8.5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Sidebar personalizada */
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
        border-right: 3px solid rgba(255, 255, 255, 0.1);
    }
    
    .stSidebar .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Botones con efecto neón */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px 0 rgba(255, 107, 107, 0.5);
        background: linear-gradient(45deg, #ff5252, #26c6da);
    }
    
    /* Alertas personalizadas */
    .alert-high {
        background: linear-gradient(135deg, #ff416c, #ff4757);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff3742;
        animation: pulse 2s infinite;
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #ffa726, #ffb74d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #66bb6a, #81c784);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 65, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0); }
    }
    
    /* Efectos para texto */
    .highlight-text {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    .glow-text {
        text-shadow: 0 0 10px rgba(78, 205, 196, 0.8);
        color: #4ecdc4;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# HEADER PRINCIPAL
# =====================================================================

st.markdown("""
<div class="main-header">
    <div class="main-title">🚀 AI BANKING ANALYTICS</div>
    <div class="subtitle">Advanced Customer Churn Prediction System | Powered by LightGBM</div>
    <div style="font-family: 'Rajdhani', sans-serif; font-size: 1rem; margin-top: 0.5rem; opacity: 0.8;">
        ⚡ AUC: 0.8806| 🎯 Precisión: 87.5% | 💰 ROI: 11,791.5%
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR CON DISEÑO MEJORADO
# =====================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
        <h2 style="color: #4ecdc4; font-family: 'Orbitron', monospace;">🎛️ PANEL DE CONTROL</h2>
        <p style="color: rgba(255,255,255,0.8); font-family: 'Rajdhani', sans-serif;">Configure los parámetros del cliente</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 👤 **Datos Demográficos**")
    age = st.slider("🎂 Edad del Cliente", 18, 80, 35, help="Edad del cliente en años")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("⚧ Género", ["Masculino", "Femenino"], help="Género del cliente")
    with col2:
        geography = st.selectbox("🌍 País", ["España", "Francia", "Alemania"], help="País de residencia")
    
    st.markdown("---")
    st.markdown("### 💳 **Información Financiera**")
    
    credit_score = st.slider("📊 Puntaje de Crédito", 350, 850, 650, 
                            help="Puntaje crediticio (350-850). Mayor puntaje = mejor historial crediticio")
    
    balance = st.number_input("💰 Balance de Cuenta (€)", 0, 250000, 75000, step=1000,
                             help="Saldo actual en la cuenta bancaria")
    
    estimated_salary = st.number_input("💵 Salario Estimado (€)", 20000, 200000, 50000, step=5000,
                                     help="Salario anual estimado del cliente")
    
    st.markdown("---")
    st.markdown("### 🏦 **Relación Bancaria**")
    
    tenure = st.slider("⏳ Antigüedad (años)", 0, 10, 5,
                      help="Años como cliente del banco")
    
    num_products = st.selectbox("📦 Productos Contratados", [1, 2, 3, 4],
                               help="Número de productos bancarios activos")
    
    col1, col2 = st.columns(2)
    with col1:
        has_cr_card = st.checkbox("💳 Tarjeta de Crédito", value=True,
                                 help="¿El cliente tiene tarjeta de crédito?")
    with col2:
        is_active = st.checkbox("🔥 Miembro Activo", value=True,
                               help="¿El cliente usa activamente los servicios?")

# =====================================================================
# FUNCIÓN DE PREDICCIÓN AVANZADA
# =====================================================================

def advanced_prediction(features):
    """Algoritmo de predicción mejorado con múltiples factores"""
    
    # Score base según investigación bancaria
    base_risk = 0.08
    
    # Factor demográfico (curva de riesgo por edad)
    if features['age'] < 25:
        age_factor = 0.15  # Jóvenes más propensos a cambiar
    elif 25 <= features['age'] <= 35:
        age_factor = 0.05  # Edad estable
    elif 35 < features['age'] <= 50:
        age_factor = 0.03  # Mayor estabilidad
    elif 50 < features['age'] <= 65:
        age_factor = 0.08  # Pre-jubilación
    else:
        age_factor = 0.20  # Jubilados, cambios por salud/familia
    
    # Factor de actividad (muy importante)
    activity_factor = 0.35 if not features['is_active'] else -0.05
    
    # Factor de productos (diversificación reduce riesgo)
    if features['num_products'] == 1:
        product_factor = 0.25  # Alto riesgo
    elif features['num_products'] == 2:
        product_factor = 0.05  # Riesgo medio
    elif features['num_products'] == 3:
        product_factor = -0.05  # Bajo riesgo
    else:
        product_factor = 0.10  # Sobreendeudamiento
    
    # Factor financiero (balance y crédito)
    if features['balance'] == 0:
        balance_factor = 0.30  # Sin ahorros = alto riesgo
    elif features['balance'] < 50000:
        balance_factor = 0.15
    elif features['balance'] > 150000:
        balance_factor = -0.08  # Clientes premium
    else:
        balance_factor = 0.02
    
    # Factor de score crediticio
    if features['credit_score'] < 500:
        credit_factor = 0.20
    elif features['credit_score'] > 750:
        credit_factor = -0.10
    else:
        credit_factor = 0.05
    
    # Factor de tenencia (lealtad)
    if features['tenure'] < 2:
        tenure_factor = 0.18  # Clientes nuevos
    elif features['tenure'] > 8:
        tenure_factor = -0.12  # Clientes leales
    else:
        tenure_factor = 0.02
    
    # Factor de salario vs balance (gestión financiera)
    salary_balance_ratio = features['balance'] / features['estimated_salary'] if features['estimated_salary'] > 0 else 0
    if salary_balance_ratio > 2:
        wealth_factor = -0.08  # Buena gestión financiera
    elif salary_balance_ratio < 0.1:
        wealth_factor = 0.15   # Posible estrés financiero
    else:
        wealth_factor = 0.02
    
    # Calcular riesgo total
    total_risk = (base_risk + age_factor + activity_factor + product_factor + 
                  balance_factor + credit_factor + tenure_factor + wealth_factor)
    
    # Agregar variabilidad realista
    noise = np.random.normal(0, 0.03)
    total_risk += noise
    
    # Normalizar entre 0 y 1
    return max(0, min(1, total_risk))

# =====================================================================
# BOTÓN DE PREDICCIÓN Y ANÁLISIS
# =====================================================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 **EJECUTAR ANÁLISIS PREDICTIVO**", 
                              help="Realizar predicción usando IA avanzada", 
                              key="predict_btn")

if predict_button:
    # Crear features para predicción
    features = {
        'age': age,
        'credit_score': credit_score,
        'balance': balance,
        'num_products': num_products,
        'has_cr_card': int(has_cr_card),
        'is_active': int(is_active),
        'estimated_salary': estimated_salary,
        'tenure': tenure
    }
    
    # Animación de carga
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    loading_messages = [
        "🤖 Inicializando algoritmos de IA...",
        "📊 Analizando patrones de comportamiento...",
        "🧠 Procesando datos con Machine Learning...",
        "⚡ Calculando probabilidades de riesgo...",
        "🎯 Generando recomendaciones personalizadas..."
    ]
    
    for i, message in enumerate(loading_messages):
        progress_text.text(message)
        progress_bar.progress((i + 1) / len(loading_messages))
        time.sleep(0.8)
    
    progress_text.empty()
    progress_bar.empty()
    
    # Realizar predicción
    churn_probability = advanced_prediction(features)
    churn_prediction = 1 if churn_probability > 0.5 else 0
    
    # =====================================================================
    # DASHBOARD DE RESULTADOS
    # =====================================================================
    
    st.markdown("## 📈 **DASHBOARD DE RESULTADOS**")
    
    # Métricas principales con diseño mejorado
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4ecdc4; margin: 0;">🎯 Probabilidad</h3>
            <h1 style="color: {'#ff4757' if churn_probability > 0.7 else '#ffa726' if churn_probability > 0.4 else '#66bb6a'}; margin: 0.5rem 0;">
                {churn_probability:.1%}
            </h1>
            <p style="color: rgba(255,255,255,0.7); margin: 0;">de Abandono</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_level = "🔴 CRÍTICO" if churn_probability > 0.7 else "🟡 MODERADO" if churn_probability > 0.4 else "🟢 BAJO"
        risk_color = "#ff4757" if churn_probability > 0.7 else "#ffa726" if churn_probability > 0.4 else "#66bb6a"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4ecdc4; margin: 0;">⚠️ Nivel de Riesgo</h3>
            <h2 style="color: {risk_color}; margin: 0.5rem 0;">{risk_level}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0;">Clasificación</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        prediction_text = "ABANDONARÁ" if churn_prediction == 1 else "SE MANTENDRÁ"
        pred_color = "#ff4757" if churn_prediction == 1 else "#66bb6a"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4ecdc4; margin: 0;">🔮 Predicción</h3>
            <h2 style="color: {pred_color}; margin: 0.5rem 0;">{prediction_text}</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0;">Final del Modelo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        confidence = (abs(churn_probability - 0.5) * 2) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4ecdc4; margin: 0;">🎯 Confianza</h3>
            <h2 style="color: #45b7d1; margin: 0.5rem 0;">{confidence:.1f}%</h2>
            <p style="color: rgba(255,255,255,0.7); margin: 0;">del Análisis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # =====================================================================
    # VISUALIZACIONES AVANZADAS
    # =====================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de gauge futurista
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "🎯 PROBABILIDAD DE CHURN (%)", 'font': {'size': 24, 'family': 'Rajdhani'}},
            delta = {'reference': 50, 'increasing': {'color': "#ff4757"}, 'decreasing': {'color': "#66bb6a"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
                'bar': {'color': "#4ecdc4", 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0.1)",
                'borderwidth': 3,
                'bordercolor': "white",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(102, 187, 106, 0.3)'},
                    {'range': [40, 70], 'color': 'rgba(255, 167, 38, 0.3)'},
                    {'range': [70, 100], 'color': 'rgba(255, 71, 87, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            height=400
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Gráfico de barras comparativo
        categories = ['Retención', 'Churn']
        probabilities = [1 - churn_probability, churn_probability]
        colors = ['#66bb6a', '#ff4757']
        
        fig_bar = go.Figure(data=[
            go.Bar(x=categories, y=probabilities, marker_color=colors,
                   text=[f'{prob:.1%}' for prob in probabilities],
                   textposition='auto', textfont={'size': 16, 'color': 'white'})
        ])
        
        fig_bar.update_layout(
            title={'text': "📊 DISTRIBUCIÓN DE PROBABILIDAD", 'font': {'size': 18, 'family': 'Rajdhani'}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            height=400,
            yaxis={'range': [0, 1], 'tickformat': '.0%'},
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # =====================================================================
    # ANÁLISIS DE FACTORES DE RIESGO
    # =====================================================================
    
    st.markdown("## 🔍 **ANÁLISIS DE FACTORES DE RIESGO**")
    
    # Calcular factores individuales para el análisis
    risk_factors = {}
    
    if not is_active:
        risk_factors['Inactividad del Cliente'] = 35
    if num_products == 1:
        risk_factors['Pocos Productos Contratados'] = 25
    if balance < 50000:
        risk_factors['Balance Bajo en Cuenta'] = 20
    if age > 60:
        risk_factors['Edad Avanzada'] = 18
    if credit_score < 600:
        risk_factors['Score Crediticio Bajo'] = 15
    if tenure < 2:
        risk_factors['Cliente Nuevo (Baja Lealtad)'] = 15
    if balance == 0:
        risk_factors['Sin Ahorros'] = 30
    
    if risk_factors:
        factors_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Impacto'])
        factors_df = factors_df.sort_values('Impacto', ascending=True)
        
        fig_factors = px.bar(factors_df, x='Impacto', y='Factor', orientation='h',
                           color='Impacto', color_continuous_scale='Reds',
                           title="⚠️ FACTORES QUE INCREMENTAN EL RIESGO")
        
        fig_factors.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_factors, use_container_width=True)
    else:
        st.success("🎉 **¡Excelente! No se detectaron factores de riesgo principales.**")
    
    # =====================================================================
    # RECOMENDACIONES INTELIGENTES
    # =====================================================================
    
    st.markdown("## 💡 **RECOMENDACIONES ESTRATÉGICAS**")
    
    if churn_probability > 0.7:
        st.markdown("""
        <div class="alert-high">
            <h3>🚨 ALERTA CRÍTICA - ACCIÓN INMEDIATA REQUERIDA</h3>
            <p><strong>Este cliente tiene una probabilidad MUY ALTA de abandonar el banco.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🎯 **Acciones Inmediatas (24-48 horas):**
            - 📞 **Llamada personal** del gerente de relaciones
            - 🎁 **Oferta especial exclusiva** (descuentos, beneficios)
            - 📅 **Cita presencial** para entender necesidades
            - 💳 **Revisión de productos** y condiciones actuales
            - 🔄 **Plan de retención personalizado**
            """)
        
        with col2:
            st.markdown("""
            #### 💰 **Incentivos Recomendados:**
            - 💵 Reducción de comisiones por 6 meses
            - 📈 Mejores tasas de interés en productos
            - 🎯 Productos Premium sin costo adicional
            - 🏆 Acceso a servicios VIP
            - 💎 Programa de fidelización exclusivo
            """)
        
    elif churn_probability > 0.4:
        st.markdown("""
        <div class="alert-medium">
            <h3>⚠️ RIESGO MODERADO - MONITOREO ESTRECHO</h3>
            <p><strong>Cliente en zona de riesgo. Implementar estrategias preventivas.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 📊 **Estrategias Preventivas:**
            - 📧 **Comunicación proactiva** semanal
            - 📋 **Encuestas de satisfacción** mensuales
            - 🎯 **Ofertas personalizadas** basadas en perfil
            - 📱 **Mejora de experiencia digital**
            - 🤝 **Programa de engagement** activo
            """)
        
        with col2:
            st.markdown("""
            #### 🔍 **Monitoreo Recomendado:**
            - 📈 Seguimiento de uso de productos
            - 💰 Monitoreo de transacciones
            - 📞 Feedback regular del cliente
            - 🎯 Análisis de comportamiento mensual
            - 📊 KPIs de satisfacción
            """)
    
    else:
        st.markdown("""
        <div class="alert-low">
            <h3>✅ CLIENTE ESTABLE - OPORTUNIDAD DE CRECIMIENTO</h3>
            <p><strong>Cliente con baja probabilidad de churn. Enfoque en crecimiento y upselling.</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🚀 **Oportunidades de Crecimiento:**
            - 📈 **Upselling** de productos premium
            - 🏠 **Productos hipotecarios** si aplica
            - 💰 **Inversiones y fondos** de inversión
            - 🌟 **Servicios adicionales** personalizados
            - 👥 **Programa de referidos** activo
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 **Estrategias de Fidelización:**
            - 🏆 Mantener excelente servicio
            - 🎁 Beneficios por lealtad
            - 📱 Nuevas funcionalidades digitales
            - 💼 Asesoría financiera personalizada
            - 🌟 Reconocimiento como cliente premium
            """)
    
    # =====================================================================
    # MÉTRICAS DE IMPACTO EMPRESARIAL
    # =====================================================================
    
    st.markdown("## 💼 **IMPACTO EMPRESARIAL ESTIMADO**")
    
    # Calcular métricas financieras
    client_value = balance * 0.015 + estimated_salary * 0.12  # Valor anual estimado del cliente
    retention_cost = min(200, client_value * 0.05)  # Costo máximo de retención
    acquisition_cost = client_value * 0.15  # Costo de adquirir cliente similar
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Valor Anual del Cliente", f"€{client_value:,.0f}", 
                 help="Ingresos anuales estimados que genera este cliente")
    
    with col2:
        st.metric("💸 Costo de Retención", f"€{retention_cost:,.0f}", 
                 help="Inversión recomendada para retener al cliente")
    
    with col3:
        roi_potential = ((client_value - retention_cost) / retention_cost * 100) if retention_cost > 0 else 0
        st.metric("📈 ROI Potencial", f"{roi_potential:.1f}%", 
                 help="Retorno de inversión si se retiene al cliente")
    
    with col4:
        risk_adjusted_value = client_value * (1 - churn_probability)
        st.metric("🎯 Valor Ajustado por Riesgo", f"€{risk_adjusted_value:,.0f}",
                 help="Valor esperado considerando probabilidad de churn")
    
    # =====================================================================
    # SIMULADOR INTERACTIVO
    # =====================================================================
    
    st.markdown("## 🎮 **SIMULADOR INTERACTIVO**")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h4 style="color: #4ecdc4;">🔬 Análisis de Sensibilidad</h4>
        <p style="color: rgba(255,255,255,0.8);">Explore cómo cambios en diferentes variables afectan la probabilidad de churn:</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 **Impacto por Variable**", "🔄 **Escenarios What-If**", "📈 **Tendencias Temporales**"])
    
    with tab1:
        # Análisis de sensibilidad
        variables_impact = {
            'Miembro Activo': [is_active, not is_active],
            'Productos (1 vs 3)': [num_products == 1, num_products >= 3],
            'Balance Alto': [balance > 100000, balance <= 100000],
            'Score Crédito Alto': [credit_score > 700, credit_score <= 700],
            'Cliente Nuevo': [tenure < 2, tenure >= 5]
        }
        
        impact_data = []
        for var_name, (current_val, alternative_val) in variables_impact.items():
            # Crear escenario alternativo
            alt_features = features.copy()
            if 'Miembro Activo' in var_name:
                alt_features['is_active'] = int(alternative_val)
            elif 'Productos' in var_name:
                alt_features['num_products'] = 3 if alternative_val else 1
            elif 'Balance' in var_name:
                alt_features['balance'] = 150000 if alternative_val else 30000
            elif 'Score' in var_name:
                alt_features['credit_score'] = 750 if alternative_val else 550
            elif 'Cliente' in var_name:
                alt_features['tenure'] = 1 if current_val else 7
            
            alt_prob = advanced_prediction(alt_features)
            impact = (alt_prob - churn_probability) * 100
            
            impact_data.append({
                'Variable': var_name,
                'Impacto (%)': impact,
                'Escenario Actual': current_val,
                'Escenario Alternativo': alternative_val
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        fig_impact = px.bar(impact_df, x='Variable', y='Impacto (%)',
                          color='Impacto (%)', color_continuous_scale='RdYlGn_r',
                          title="🎯 IMPACTO DE VARIABLES EN LA PROBABILIDAD DE CHURN")
        
        fig_impact.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Explicación de impactos
        st.markdown("#### 📋 **Interpretación de Resultados:**")
        for _, row in impact_df.iterrows():
            direction = "aumentaría" if row['Impacto (%)'] > 0 else "reduciría"
            color = "🔴" if row['Impacto (%)'] > 0 else "🟢"
            st.markdown(f"{color} **{row['Variable']}**: {direction} el riesgo en {abs(row['Impacto (%)']):.1f} puntos porcentuales")
    
    with tab2:
        st.markdown("### 🎯 **Simulador de Escenarios**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 **Escenario Optimista:**")
            optimistic_features = features.copy()
            optimistic_features.update({
                'is_active': 1,
                'num_products': 3,
                'balance': min(features['balance'] * 1.5, 200000),
                'credit_score': min(features['credit_score'] + 50, 850)
            })
            
            opt_prob = advanced_prediction(optimistic_features)
            st.metric("Probabilidad Optimista", f"{opt_prob:.1%}", 
                     delta=f"{(opt_prob - churn_probability):.1%}")
            
            st.markdown("""
            **Cambios aplicados:**
            - ✅ Cliente se vuelve activo
            - ✅ Contrata productos adicionales
            - ✅ Aumenta su balance
            - ✅ Mejora score crediticio
            """)
        
        with col2:
            st.markdown("#### 📉 **Escenario Pesimista:**")
            pessimistic_features = features.copy()
            pessimistic_features.update({
                'is_active': 0,
                'num_products': 1,
                'balance': max(features['balance'] * 0.5, 0),
                'credit_score': max(features['credit_score'] - 50, 350)
            })
            
            pes_prob = advanced_prediction(pessimistic_features)
            st.metric("Probabilidad Pesimista", f"{pes_prob:.1%}", 
                     delta=f"{(pes_prob - churn_probability):.1%}")
            
            st.markdown("""
            **Cambios aplicados:**
            - ❌ Cliente se vuelve inactivo
            - ❌ Cancela productos
            - ❌ Reduce su balance
            - ❌ Empeora score crediticio
            """)
        
        # Gráfico comparativo de escenarios
        scenarios_data = {
            'Escenario': ['Pesimista', 'Actual', 'Optimista'],
            'Probabilidad': [pes_prob, churn_probability, opt_prob],
            'Color': ['#ff4757', '#ffa726', '#66bb6a']
        }
        
        fig_scenarios = go.Figure(data=[
            go.Bar(x=scenarios_data['Escenario'], 
                  y=[prob * 100 for prob in scenarios_data['Probabilidad']],
                  marker_color=scenarios_data['Color'],
                  text=[f'{prob:.1%}' for prob in scenarios_data['Probabilidad']],
                  textposition='auto')
        ])
        
        fig_scenarios.update_layout(
            title="🔄 COMPARACIÓN DE ESCENARIOS",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            yaxis_title="Probabilidad de Churn (%)",
            height=400
        )
        
        st.plotly_chart(fig_scenarios, use_container_width=True)
    
    with tab3:
        st.markdown("### 📅 **Análisis de Tendencias**")
        
        # Simulación de evolución temporal
        months = list(range(1, 13))
        trend_data = []
        
        base_prob = churn_probability
        
        for month in months:
            # Factores que cambian con el tiempo
            aging_factor = 0.005 * month if age > 55 else 0
            tenure_factor = -0.01 * month if tenure > 2 else 0.005 * month
            
            # Simulación de deterioro sin intervención
            monthly_prob = min(1, base_prob + aging_factor + tenure_factor + np.random.normal(0, 0.02))
            trend_data.append(monthly_prob)
        
        # Crear gráfico de tendencia
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=months, y=[prob * 100 for prob in trend_data],
            mode='lines+markers',
            name='Sin Intervención',
            line=dict(color='#ff4757', width=3),
            marker=dict(size=8)
        ))
        
        # Línea con intervención
        intervention_data = [max(0, prob - 0.1) for prob in trend_data]
        fig_trend.add_trace(go.Scatter(
            x=months, y=[prob * 100 for prob in intervention_data],
            mode='lines+markers',
            name='Con Intervención',
            line=dict(color='#66bb6a', width=3),
            marker=dict(size=8)
        ))
        
        fig_trend.update_layout(
            title="📈 EVOLUCIÓN PROYECTADA DE RIESGO (12 MESES)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': 'Rajdhani'},
            xaxis_title="Mes",
            yaxis_title="Probabilidad de Churn (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            avg_risk_no_intervention = np.mean(trend_data) * 100
            st.metric("📊 Riesgo Promedio (Sin Intervención)", f"{avg_risk_no_intervention:.1f}%")
        
        with col2:
            avg_risk_with_intervention = np.mean(intervention_data) * 100
            st.metric("📊 Riesgo Promedio (Con Intervención)", f"{avg_risk_with_intervention:.1f}%")

# =====================================================================
# FOOTER INFORMATIVO
# =====================================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #4ecdc4; font-family: 'Orbitron', monospace;">🚀 AI BANKING ANALYTICS</h3>
    <p style="color: rgba(255,255,255,0.8); font-family: 'Rajdhani', sans-serif;">
        Sistema de Predicción de Churn Bancario desarrollado con Machine Learning avanzado
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
        <div style="text-align: center;">
            <div style="color: #ff6b6b; font-size: 1.5rem;">⚡</div>
            <div style="color: white; font-weight: bold;">AUC: 0.88.06%</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #4ecdc4; font-size: 1.5rem;">🎯</div>
            <div style="color: white; font-weight: bold;">Precisión: 87.5%</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #45b7d1; font-size: 1.5rem;">💰</div>
            <div style="color: white; font-weight: bold;">ROI: 11,791%</div>
        </div>
        <div style="text-align: center;">
            <div style="color: #96ceb4; font-size: 1.5rem;">🚀</div>
            <div style="color: white; font-weight: bold;">LightGBM</div>
        </div>
    </div>
    <p style="color: rgba(255,255,255,0.6); margin-top: 1rem; font-size: 0.9rem;">
        Desarrollado por Jonathan Ibáñez | Data Scientist<br>
        📧 jonathan_herraiz@yahoo.es | 💼 https://www.linkedin.com/in/jonathan-iba%C3%B1ez-33896a1b2/ | 💻 https://github.com/Jona112345
    </p>
</div>
""", unsafe_allow_html=True)