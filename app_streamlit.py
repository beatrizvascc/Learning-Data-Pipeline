"""
Interface Web - Pipeline de Análise de Primos

Dashboard interativo usando Streamlit para análise de números primos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Importa módulos do pipeline
from research.primos.padroes_primos.distribuicao_primos.config import (
    get_dataset_path, get_results_img_dir, get_results_rel_dir,
    criar_diretorios, DATA_DIR
)
from research.primos.padroes_primos.distribuicao_primos.gerar_dataset import (
    gerar_dataset_teste, verificar_ou_gerar_dataset
)
from research.primos.padroes_primos.distribuicao_primos.pipeline_primos import (
    carregar_dataset, analise_fourier, analise_wavelet, 
    analise_pca, analise_autoencoder, analise_gnn
)

# Configuração da página
st.set_page_config(
    page_title="Pipeline de Análise de Primos",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path=None):
    """Carrega o dataset com cache."""
    try:
        if file_path is None:
            file_path = verificar_ou_gerar_dataset()
        df = carregar_dataset(file_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {e}")
        return None


def plot_gaps_interactive(gaps, title="Distribuição de Gaps entre Primos"):
    """Cria gráfico interativo de gaps usando Plotly."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(gaps))),
        y=gaps,
        mode='lines+markers',
        name='Gaps',
        line=dict(color='#1f77b4', width=1),
        marker=dict(size=3, opacity=0.6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Índice do Primo",
        yaxis_title="Gap",
        hovermode='closest',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_histogram_gaps(gaps, title="Histograma de Gaps"):
    """Cria histograma interativo de gaps."""
    fig = px.histogram(
        x=gaps,
        nbins=50,
        title=title,
        labels={'x': 'Gap', 'y': 'Frequência'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_statistics(df):
    """Cria gráfico de estatísticas básicas."""
    if 'gap' not in df.columns:
        return None
    
    gaps = df['gap'].values
    stats = {
        'Média': np.mean(gaps),
        'Mediana': np.median(gaps),
        'Desvio Padrão': np.std(gaps),
        'Mínimo': np.min(gaps),
        'Máximo': np.max(gaps)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            marker_color='#1f77b4',
            text=[f'{v:.2f}' for v in stats.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Estatísticas dos Gaps",
        xaxis_title="Estatística",
        yaxis_title="Valor",
        height=400,
        template='plotly_white'
    )
    
    return fig, stats


def main():
    """Função principal da aplicação."""
    
    # Header
    st.markdown('<div class="main-header">🔢 Pipeline de Análise de Primos</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Dataset")
        
        # Opções de dataset
        dataset_option = st.radio(
            "Fonte do Dataset",
            ["Gerar Automaticamente", "Usar Dataset Existente"],
            help="Escolha gerar um novo dataset ou usar um existente"
        )
        
        if dataset_option == "Gerar Automaticamente":
            num_primos = st.slider(
                "Número de Primos",
                min_value=100,
                max_value=50000,
                value=5000,
                step=100,
                help="Número de primos para gerar no dataset"
            )
            
            limite = st.slider(
                "Limite para Crivo",
                min_value=1000,
                max_value=200000,
                value=100000,
                step=1000,
                help="Limite máximo para o Crivo de Eratóstenes"
            )
            
            if st.button("🔄 Gerar Dataset", type="primary"):
                with st.spinner("Gerando dataset..."):
                    try:
                        df = gerar_dataset_teste(num_primos=num_primos, limite=limite)
                        st.success(f"✅ Dataset gerado: {len(df)} registros")
                        st.session_state['dataset'] = df
                        st.session_state['dataset_path'] = None
                    except Exception as e:
                        st.error(f"Erro ao gerar dataset: {e}")
        else:
            dataset_path = st.text_input(
                "Caminho do Dataset",
                value=str(get_dataset_path()),
                help="Caminho para o arquivo CSV do dataset"
            )
            
            if st.button("📂 Carregar Dataset"):
                with st.spinner("Carregando dataset..."):
                    try:
                        df = load_data(dataset_path)
                        if df is not None:
                            st.success(f"✅ Dataset carregado: {len(df)} registros")
                            st.session_state['dataset'] = df
                            st.session_state['dataset_path'] = dataset_path
                    except Exception as e:
                        st.error(f"Erro ao carregar dataset: {e}")
        
        st.markdown("---")
        
        # Seleção de análises
        st.header("🔬 Análises")
        
        analises = {
            'Fourier': st.checkbox("📊 Fourier (FFT)", value=True),
            'Wavelet': st.checkbox("🌊 Wavelet", value=True),
            'PCA': st.checkbox("📉 PCA", value=True),
            'Autoencoder': st.checkbox("🧠 Autoencoder", value=False, 
                                      help="Pode demorar alguns minutos"),
            'GNN': st.checkbox("🕸️ Graph Neural Network", value=False)
        }
        
        st.session_state['analises'] = analises
        
        st.markdown("---")
        
        # Configurações
        st.header("⚙️ Configurações")
        
        range_inicio = st.number_input(
            "Índice Inicial",
            min_value=0,
            value=0,
            help="Índice inicial para filtrar dados"
        )
        
        range_fim = st.number_input(
            "Índice Final",
            min_value=0,
            value=None,
            help="Índice final para filtrar dados (None = todos)"
        )
        
        st.session_state['range_inicio'] = range_inicio
        st.session_state['range_fim'] = range_fim if range_fim else None
    
    # Main area
    if 'dataset' not in st.session_state or st.session_state['dataset'] is None:
        st.info("👈 Configure o dataset na barra lateral para começar")
        
        # Mostra informações sobre o projeto
        with st.expander("ℹ️ Sobre este projeto"):
            st.markdown("""
            ### Pipeline de Análise de Primos
            
            Este projeto oferece uma análise completa de números primos usando:
            
            - **Fourier (FFT)**: Análise espectral dos gaps
            - **Wavelet**: Decomposição multi-nível
            - **PCA**: Redução de dimensionalidade
            - **Autoencoder**: Detecção de padrões com Deep Learning
            - **GNN**: Representação como grafo
            
            ### Como usar:
            
            1. Configure o dataset na barra lateral
            2. Selecione as análises desejadas
            3. Clique em "Executar Análises"
            4. Explore os resultados interativos
            """)
        return
    
    df = st.session_state['dataset']
    
    # Aplica filtro de range se especificado
    if st.session_state.get('range_fim'):
        df_filtered = df.iloc[st.session_state['range_inicio']:st.session_state['range_fim']]
    else:
        df_filtered = df.iloc[st.session_state['range_inicio']:]
    
    if len(df_filtered) == 0:
        st.warning("⚠️ Nenhum dado disponível com os filtros selecionados")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualizações", 
        "🔬 Análises", 
        "📈 Estatísticas",
        "📄 Sobre"
    ])
    
    with tab1:
        st.header("Visualizações Interativas")
        
        if 'gap' not in df_filtered.columns:
            st.error("Dataset não contém coluna 'gap'")
            return
        
        gaps = df_filtered['gap'].values
        
        # Gráfico de gaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                plot_gaps_interactive(gaps[:1000], "Gaps entre Primos (primeiros 1000)"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                plot_histogram_gaps(gaps, "Distribuição de Gaps"),
                use_container_width=True
            )
        
        # Estatísticas
        st.subheader("Estatísticas Descritivas")
        fig_stats, stats = plot_statistics(df_filtered)
        if fig_stats:
            st.plotly_chart(fig_stats, use_container_width=True)
            
            # Métricas em cards
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Média", f"{stats['Média']:.2f}")
            with col2:
                st.metric("Mediana", f"{stats['Mediana']:.2f}")
            with col3:
                st.metric("Desvio Padrão", f"{stats['Desvio Padrão']:.2f}")
            with col4:
                st.metric("Mínimo", f"{int(stats['Mínimo'])}")
            with col5:
                st.metric("Máximo", f"{int(stats['Máximo'])}")
    
    with tab2:
        st.header("Executar Análises")
        
        if st.button("🚀 Executar Análises Selecionadas", type="primary"):
            analises_selecionadas = st.session_state['analises']
            
            if not any(analises_selecionadas.values()):
                st.warning("⚠️ Selecione pelo menos uma análise")
                return
            
            gaps = df_filtered['gap'].values
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            resultados = []
            
            total_analises = sum(analises_selecionadas.values())
            analise_atual = 0
            
            # Fourier
            if analises_selecionadas['Fourier']:
                analise_atual += 1
                status_text.text(f"Executando Fourier ({analise_atual}/{total_analises})...")
                progress_bar.progress(analise_atual / total_analises)
                try:
                    resultado = analise_fourier(gaps, salvar_grafico=True)
                    resultados.append(("Fourier", resultado, "✅"))
                except Exception as e:
                    resultados.append(("Fourier", f"Erro: {e}", "❌"))
            
            # Wavelet
            if analises_selecionadas['Wavelet']:
                analise_atual += 1
                status_text.text(f"Executando Wavelet ({analise_atual}/{total_analises})...")
                progress_bar.progress(analise_atual / total_analises)
                try:
                    resultado = analise_wavelet(gaps, salvar_grafico=True)
                    resultados.append(("Wavelet", resultado, "✅"))
                except Exception as e:
                    resultados.append(("Wavelet", f"Erro: {e}", "❌"))
            
            # PCA
            if analises_selecionadas['PCA']:
                analise_atual += 1
                status_text.text(f"Executando PCA ({analise_atual}/{total_analises})...")
                progress_bar.progress(analise_atual / total_analises)
                try:
                    resultado = analise_pca(df_filtered, salvar_grafico=True)
                    resultados.append(("PCA", resultado, "✅"))
                except Exception as e:
                    resultados.append(("PCA", f"Erro: {e}", "❌"))
            
            # Autoencoder
            if analises_selecionadas['Autoencoder']:
                analise_atual += 1
                status_text.text(f"Executando Autoencoder ({analise_atual}/{total_analises})...")
                progress_bar.progress(analise_atual / total_analises)
                st.info("⏳ Autoencoder pode demorar alguns minutos...")
                try:
                    resultado = analise_autoencoder(gaps, salvar_grafico=True)
                    resultados.append(("Autoencoder", resultado, "✅"))
                except Exception as e:
                    resultados.append(("Autoencoder", f"Erro: {e}", "❌"))
            
            # GNN
            if analises_selecionadas['GNN']:
                analise_atual += 1
                status_text.text(f"Executando GNN ({analise_atual}/{total_analises})...")
                progress_bar.progress(1.0)
                try:
                    resultado = analise_gnn(df_filtered, salvar_grafico=True)
                    resultados.append(("GNN", resultado, "✅"))
                except Exception as e:
                    resultados.append(("GNN", f"Erro: {e}", "❌"))
            
            # Mostra resultados
            status_text.text("✅ Análises concluídas!")
            progress_bar.empty()
            
            st.subheader("📊 Resultados")
            for nome, resultado, status in resultados:
                st.markdown(f"{status} **{nome}**: {resultado}")
            
            st.success(f"✅ {len([r for r in resultados if r[2] == '✅'])}/{len(resultados)} análises concluídas com sucesso!")
            
            # Links para visualizar resultados
            st.subheader("📁 Arquivos Gerados")
            resultados_dir = get_results_img_dir()
            st.info(f"Imagens salvas em: `{resultados_dir}`")
    
    with tab3:
        st.header("Estatísticas Detalhadas")
        
        if 'gap' in df_filtered.columns:
            gaps = df_filtered['gap'].values
            
            # Tabela de estatísticas
            st.subheader("Tabela de Estatísticas")
            stats_df = pd.DataFrame({
                'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Variância', 
                               'Mínimo', 'Máximo', 'Q1 (25%)', 'Q3 (75%)'],
                'Valor': [
                    np.mean(gaps),
                    np.median(gaps),
                    np.std(gaps),
                    np.var(gaps),
                    np.min(gaps),
                    np.max(gaps),
                    np.percentile(gaps, 25),
                    np.percentile(gaps, 75)
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
            
            # Box plot
            st.subheader("Box Plot dos Gaps")
            fig_box = px.box(y=gaps, title="Distribuição de Gaps (Box Plot)")
            fig_box.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_box, use_container_width=True)
    
    with tab4:
        st.header("Sobre o Projeto")
        
        st.markdown("""
        ### 🔢 Pipeline de Análise de Primos
        
        Sistema completo de análise de números primos usando técnicas avançadas de 
        machine learning e análise de sinais.
        
        #### 🎯 Funcionalidades
        
        - **Geração Automática de Dataset**: Cria datasets de primos sem arquivos externos
        - **Análise de Fourier**: Transformada de Fourier para análise espectral
        - **Análise Wavelet**: Decomposição multi-nível usando PyWavelets
        - **PCA**: Redução de dimensionalidade e visualização
        - **Autoencoder**: Deep Learning para detecção de padrões
        - **Graph Neural Networks**: Representação de primos como grafos
        
        #### 🛠️ Tecnologias
        
        - Python 3.8+
        - Streamlit (Interface Web)
        - Plotly (Visualizações Interativas)
        - TensorFlow/Keras (Deep Learning)
        - scikit-learn (Machine Learning)
        - PyWavelets (Análise Wavelet)
        - NetworkX (Grafos)
        
        #### 📚 Documentação
        
        Consulte os arquivos README.md e GUIA_INSTALACAO.md para mais informações.
        
        #### 🚀 Versão
        
        **v2.0.0** - Fase 2: Interface Web
        """)
        
        st.markdown("---")
        st.markdown("**Desenvolvido com ❤️ para análise científica de números primos**")


if __name__ == "__main__":
    # Cria diretórios necessários
    criar_diretorios()
    
    # Executa a aplicação
    main()


