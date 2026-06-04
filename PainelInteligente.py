%%writefile PainelInteligente.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# 1. CONFIGURAÇÃO E ESTILIZAÇÃO DO PAINEL
# ==============================================================================
st.set_page_config(
    page_title="Portal de Sistemas IA - Daphne", 
    page_icon="🖥️", 
    layout="centered"
)

# Customização visual simples via Markdown
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .subtitle { font-size: 18px; color: #4B5563; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BARRA LATERAL DE NAVEGAÇÃO (INTERFACE DO USUÁRIO)
# ==============================================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=70)
st.sidebar.title("Navegação do Sistema")
st.sidebar.markdown("Escolha qual modelo preditivo de Inteligência Artificial deseja testar:")

# Menu suspenso para alternar entre os módulos
modulo = st.sidebar.selectbox(
    "Selecione o Módulo:", 
    ["📊 Gestão de Equipes (Regressão)", "🏥 Triagem Hospitalar (Classificação)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Desenvolvedora")
st.sidebar.info("**Daphne da Silva Pereira**\n\nProjeto Prático para Portfólio de Ciência de Dados & IA.")

# ==============================================================================
# MÓDULO A: GESTÃO DE EQUIPES (PREVISÃO DE DEMANDA)
# ==============================================================================
if modulo == "📊 Gestão de Equipes (Regressão)":
    st.markdown('<p class="main-title">🔮 IA: Previsão Estatística de Atendimentos</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Modelo de Regressão Linear para dimensionamento de pessoal e otimização de fluxos.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Engenharia de Dados: Gerando histórico de treino simulado
    np.random.seed(42)
    funcionarios = np.random.randint(2, 12, size=100)
    atendimentos = (funcionarios * 15) + np.random.randint(-10, 10, size=100)
    dados_gestao = pd.DataFrame({'Qtd_Funcionarios': funcionarios, 'Atendimentos': atendimentos})

    # Treinamento do Algoritmo
    X_g = dados_gestao[['Qtd_Funcionarios']]
    y_g = dados_gestao['Atendimentos']
    modelo_gestao = LinearRegression()
    modelo_gestao.fit(X_g, y_g)

    # Painel de Controle de Entrada do Usuário
    st.subheader("Simulação de Cenário Operacional")
    qtd_selecionada = st.slider("Selecione a quantidade de funcionários planejada para amanhã:", min_value=2, max_value=12, value=8)

    # Executando a Predição
    previsao = modelo_gestao.predict([[qtd_selecionada]])
    resultado_final = int(previsao[0])

    # Exibição dos Resultados com foco em UI/UX
    st.markdown("### 📋 Resultado do Dimensionamento")
    st.metric(label="Capacidade Estimada de Atendimentos Excedidos", value=f"{resultado_final} pacientes/clientes")
    
    st.success(f"💡 **Recomendação da IA:** Com uma equipe de **{qtd_selecionada} pessoas**, o sistema está calibrado para suportar com segurança até **{resultado_final} atendimentos**.")


# ==============================================================================
# MÓDULO B: TRIAGEM HOSPITALAR (CLASSIFICAÇÃO DE RISCO)
# ==============================================================================
elif modulo == "🏥 Triagem Hospitalar (Classificação)":
    st.markdown('<p class="main-title">🏥 IA: Triagem e Classificação de Risco Clínico</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Algoritmo de Aprendizado de Máquina supervisionado para suporte à decisão em prontos-socorros.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Engenharia de Dados: Simulando histórico clínico (Protocolo Manchester simplificado)
    np.random.seed(42)
    n_pacientes = 200
    dor_peito = np.random.randint(0, 2, n_pacientes)
    febre_alta = np.random.randint(0, 2, n_pacientes)
    dor_leve = np.random.randint(0, 2, n_pacientes)

    gravidade = []
    for i in range(n_pacientes):
        if dor_peito[i] == 1: 
            gravidade.append("🔴 Emergência (Atendimento Imediato)")
        elif febre_alta[i] == 1: 
            gravidade.append("🟡 Urgente (Atendimento em até 60 min)")
        else: 
            gravidade.append("🟢 Pouco Urgente (Pode aguardar na fila)")

    dados_saude = pd.DataFrame({'Dor_Peito': dor_peito, 'Febre_Alta': febre_alta, 'Dor_Leve': dor_leve, 'Classe': gravidade})

    # Treinamento da Árvore de Decisão do Zero
    X_s = dados_saude[['Dor_Peito', 'Febre_Alta', 'Dor_Leve']]
    y_s = dados_saude['Classe']
    modelo_saude = DecisionTreeClassifier(random_state=42)
    modelo_saude.fit(X_s, y_s)

    # Coleta de Sinais e Sintomas na Interface
    st.subheader("🩺 Formulário de Triagem (Entrada de Dados)")
    st.write("Marque as condições clínicas observadas na admissão do paciente:")
    
    s1 = st.checkbox("Paciente com Dor no Peito irradiando, Sinais de Infarto ou Falta de Ar severa?")
    s2 = st.checkbox("Paciente com Quadro de Febre Alta persistente (Acima de 38.5°C)?")
    s3 = st.checkbox("Paciente apresenta apenas Queixas Leves (Ex: Coriza, dor de garganta ou escoriações)?")

    # Vetor de características para predição
    entrada_clinica = [[1 if s1 else 0, 1 if s2 else 0, 1 if s3 else 0]]

    st.markdown("---")
    
    # Gatilho de Processamento da Triagem
    if st.button("🔴 Executar Classificação de Risco"):
        diagnostico_ia = modelo_saude.predict(entrada_clinica)[0]
        
        # Lógica de alertas visuais baseada na gravidade predita pela IA
        st.markdown("### 📢 Classificação do Paciente")
        if "🔴" in diagnostico_ia:
            st.error(f"**Status:** {diagnostico_ia}")
            st.warning("🚨 **ALERTA CLÍNICO:** Encaminhar imediatamente o paciente à Sala Vermelha / Reanimação.")
        elif "🟡" in diagnostico_ia:
            st.warning(f"**Status:** {diagnostico_ia}")
            st.info("⚠️ **Aviso:** Direcionar à equipe médica para atendimento prioritário.")
        else:
            st.success(f"**Status:** {diagnostico_ia}")
            st.info("✅ **Aviso:** Fluxo normal. Paciente estável para aguardar em consultório ambulatorial.")



