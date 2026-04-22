import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Cutting Stock Problem - CSP", layout="wide", page_icon="📐")

# --- INÍCIO DO PROJETO ---
st.write("---") # Linha divisória

# Pegando a data atual automaticamente
data_atual = datetime.datetime.now().strftime("%d/%m/%Y")

# Usando um bloco de código ou markdown para o rodapé
texto_rodape = (
    f"\n\n**PROJETO IA - OTIMIZAÇÃO DE CORTE 1D (PYMOO + NESTING MAP)**\n"
    f"**Artigo Base: A Genetic Algorithm for the 1D Cutting Stock Problem**\n"
    f"**Universidade Federal de Catalão (UFCAT)**\n"
    f"**Docente:** Dr. Sérgio\n"
    f"**Discente:** Edílson Alves da Silva \n"
    f"**Disciplina:** Inteligência Artificial\n"
    f"**Data:** {data_atual}"
)

st.markdown(f"""
<style>
    .rodape-google {{
        font-family: 'Courier New', Courier, monospace;
        color: #808080;
        font-size: 0.85em;
        line-height: 1.6;
        white-space: pre-wrap; /* Mantém as quebras de linha e espaços */
        word-wrap: break-word;
        border: none;
        background: none;
        padding: 20px 0px;
    }}
</style>
<div class="rodape-google">
{texto_rodape}
</div>
""", unsafe_allow_html=True)
st.write("---") # Linha divisória


# Estilização customizada
st.markdown("""
    <style>
    /* Estilização das métricas para se adaptar ao tema */
    [data-testid="stMetric"] {
        background-color: rgba(125, 125, 125, 0.1); /* Fundo sutil transparente */
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(125, 125, 125, 0.2);
    }
    
    /* Garante que o texto da métrica siga a cor do tema do Streamlit */
    [data-testid="stMetric"] label, [data-testid="stMetric"] div {
        color: var(--text-color);
    }

    /* Ajuste para o formulário na sidebar */
    [data-testid="stForm"] {
        border: 1px solid rgba(125, 125, 125, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Otimizador de Corte 1D - Cutting Stock Problem (CSP)")
st.markdown("Desenvolvido para otimização de estoque utilizando **Algoritmos Genéticos (IA)**.")

# --- GERENCIAMENTO DE ESTADO (DEMANDAS) ---
if 'demandas' not in st.session_state:
    st.session_state.demandas = {1350: 15, 2100: 12, 850: 20}

# --- SIDEBAR: CONFIGURAÇÕES ---
st.sidebar.header("⚙️ Parâmetros do Sistema")
tamanho_barra = st.sidebar.number_input("Tamanho da Barra (mm)", value=6000, min_value=1)

with st.sidebar.form("form_add"):
    st.write("### Adicionar Peça")
    tam = st.number_input("Comprimento (mm)", min_value=1, value=500)
    qtd = st.number_input("Quantidade", min_value=1, value=1)
    if st.form_submit_button("Adicionar à Lista"):
        st.session_state.demandas[tam] = st.session_state.demandas.get(tam, 0) + qtd

if st.sidebar.button("🗑️ Limpar Lista de Peças"):
    st.session_state.demandas = {}
    st.rerun()

# --- INTERFACE PRINCIPAL: LISTAGEM ---
st.write("### 📋 Demandas Cadastradas")
if st.session_state.demandas:
    cols = st.columns(len(st.session_state.demandas))
    for i, (t, q) in enumerate(st.session_state.demandas.items()):
        cols[i % len(cols)].metric(label=f"Peça {t}mm", value=f"{q} un")
else:
    st.info("A lista está vazia. Adicione peças na barra lateral.")

# --- LÓGICA DO PROBLEMA (GA) ---
class ProblemaCorte(ElementwiseProblem):
    def __init__(self, pecas, limite):
        self.pecas = pecas
        self.limite = limite
        super().__init__(n_var=len(pecas), n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        indices = np.argsort(x)
        barras = []
        for idx in indices:
            peca = self.pecas[idx]
            foi_alocada = False
            for i in range(len(barras)):
                if barras[i] + peca <= self.limite:
                    barras[i] += peca
                    foi_alocada = True
                    break
            if not foi_alocada:
                barras.append(peca)
        out["F"] = [len(barras)]

# --- BOTÃO DE EXECUÇÃO ---
if st.button("🚀 Iniciar Otimização IA") and st.session_state.demandas:
    
    # Preparar dados para o algoritmo
    pecas_sistema = []
    for t, q in st.session_state.demandas.items():
        pecas_sistema.extend([t] * q)
    
    with st.spinner('A IA está processando as melhores combinações...'):
        # 1. Baseline FFD (Heurística Humana)
        pecas_ord = sorted(pecas_sistema, reverse=True)
        barras_ffd = []
        for p in pecas_ord:
            fit = False
            for b in barras_ffd:
                if sum(b) + p <= tamanho_barra:
                    b.append(p)
                    fit = True
                    break
            if not fit: barras_ffd.append([p])
        val_ffd = len(barras_ffd)

        # 2. IA GA Solver via Pymoo
        problem = ProblemaCorte(pecas_sistema, tamanho_barra)
        algorithm = NSGA2(pop_size=100, sampling=FloatRandomSampling(), 
                          crossover=SBX(prob=0.9, eta=15), 
                          mutation=PolynomialMutation(prob=0.05, eta=20))
        
        res = minimize(problem, algorithm, ('n_gen', 60), seed=42)
        
        # Extração segura do resultado
        val_ga = int(np.ravel(res.F)[0])
        melhor_x = res.X[0] if res.X.ndim > 1 else res.X
        indices_finais = np.argsort(melhor_x.flatten())

    # --- MÉTRICAS DE RESULTADO ---
    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Barras (Heurística FFD)", f"{val_ffd}")
    m2.metric("Barras (IA - Genético)", f"{val_ga}")
    melhoria = ((val_ffd - val_ga)/val_ffd)*100
    m3.metric("Ganho de Eficiência", f"{melhoria:.1f}%")

    # --- MAPA DE NESTING INTERATIVO ---
    st.write("### 🏗️ Mapa de Corte Detalhado")
    
    barras_ga_final = []
    for idx in indices_finais:
        if idx < len(pecas_sistema):
            p = pecas_sistema[idx]
            fit = False
            for b in barras_ga_final:
                if sum(b) + p <= tamanho_barra:
                    b.append(p)
                    fit = True
                    break
            if not fit: barras_ga_final.append([p])

    fig_nesting = go.Figure()
    unique_sizes = sorted(list(set(pecas_sistema)))
    colors = px.colors.qualitative.Dark24
    color_map = {s: colors[i % len(colors)] for i, s in enumerate(unique_sizes)}

    for i, barra in enumerate(barras_ga_final):
        curr_x = 0
        for p in barra:
            fig_nesting.add_trace(go.Bar(
                y=[f"Barra {i+1}"], x=[p], orientation='h',
                marker=dict(color=color_map[p], line=dict(color='black', width=1)),
                hovertemplate=f"Tamanho: {p}mm<extra></extra>",
                showlegend=False
            ))
            curr_x += p
        
        # Sobra (Retalho)
        sobra = tamanho_barra - curr_x
        if sobra > 0:
            fig_nesting.add_trace(go.Bar(
                y=[f"Barra {i+1}"], x=[sobra], orientation='h',
                marker=dict(color='rgba(200,200,200,0.3)', line=dict(color='red', width=1)),
                hovertemplate=f"Sobra: {sobra}mm<extra></extra>",
                showlegend=False
            ))

    fig_nesting.update_layout(
        barmode='stack', height=max(400, len(barras_ga_final)*35),
        xaxis_title="Comprimento (mm)", 
        xaxis=dict(range=[0, tamanho_barra + 100]),
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_nesting, use_container_width=True)

    # --- RESUMO DE OCUPAÇÃO ---
    st.write("### 📊 Eficiência de Preenchimento por Barra")
    ocupacoes = [(sum(b)/tamanho_barra)*100 for b in barras_ga_final]
    
    fig_ocupa = go.Figure(go.Bar(
        x=[f"B{i+1}" for i in range(len(ocupacoes))],
        y=ocupacoes,
        marker_color=['#2E7D32' if o > 95 else '#F9A825' if o > 80 else '#C62828' for o in ocupacoes],
        hovertemplate="Ocupação: %{y:.1f}%<extra></extra>"
    ))
    fig_ocupa.add_hline(y=100, line_dash="dash", line_color="black")
    fig_ocupa.update_layout(yaxis_title="Ocupação (%)", xaxis_title="Barras Utilizadas", yaxis=dict(range=[0,115]))
    st.plotly_chart(fig_ocupa, use_container_width=True)

elif not st.session_state.demandas:
    st.warning("Adicione peças na barra lateral para habilitar a otimização.")

