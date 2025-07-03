import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
import gspread
from google.oauth2.service_account import Credentials
from io import BytesIO
import json
from typing import Dict, Set, Tuple, List

# Configuração da página
st.set_page_config(
    page_title="Classificador de Gênero por Nome",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para obter credenciais do Google Sheets
def get_google_credentials():
    """Obtém credenciais do Google Sheets a partir dos secrets do Streamlit"""
    try:
        # Tenta obter das variáveis de ambiente do Streamlit
        credentials_dict = {
            "type": "service_account",
            "project_id": st.secrets["google"]["project_id"],
            "private_key_id": st.secrets["google"]["private_key_id"],
            "private_key": st.secrets["google"]["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["google"]["client_email"],
            "client_id": st.secrets["google"]["client_id"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": st.secrets["google"]["client_x509_cert_url"],
            "universe_domain": "googleapis.com"
        }
        return credentials_dict
    except Exception as e:
        st.error(f"Erro ao obter credenciais: {str(e)}")
        return None

# IDs das planilhas
IBGE_SHEET_ID = st.secrets["google"]["spreadsheet_id"]

class GenderClassifier:
    def __init__(self):
        self.gender_dict = {}
        self.loaded = False
        
    def normalize_name(self, name: str) -> str:
        """Normaliza o nome removendo acentos e formatando"""
        if pd.isna(name):
            return ""
        
        # Remove acentos
        name = unicodedata.normalize('NFD', str(name))
        name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
        
        # Converte para maiúsculo e remove espaços extras
        name = name.upper().strip()
        
        # Remove caracteres especiais, mantendo apenas letras e espaços
        name = re.sub(r'[^A-Z\s]', '', name)
        
        return name
    
    def extract_alternative_names(self, alternative_names: str) -> Set[str]:
        """Extrai os nomes alternativos da string formatada com |"""
        if pd.isna(alternative_names):
            return set()
        
        # Split por | e remove strings vazias
        names = [name.strip() for name in str(alternative_names).split('|') if name.strip()]
        
        # Normaliza cada nome
        normalized_names = {self.normalize_name(name) for name in names if name.strip()}
        
        return normalized_names
    
    def load_ibge_data_from_sheets(self) -> bool:
        """Carrega dados do IBGE do Google Sheets"""
        try:
            # Obter credenciais
            credentials_dict = get_google_credentials()
            if not credentials_dict:
                return False
            
            # Configurar credenciais
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]
            
            credentials = Credentials.from_service_account_info(
                credentials_dict, scopes=scope
            )
            
            client = gspread.authorize(credentials)
            
            # Abrir planilha
            sheet = client.open_by_key(IBGE_SHEET_ID)
            worksheet = sheet.get_worksheet(0)  # Primeira aba
            
            # Obter dados
            data = worksheet.get_all_records()
            df_ibge = pd.DataFrame(data)
            
            # Processar dados
            expanded_names = []
            
            for _, row in df_ibge.iterrows():
                if 'classification' in row and 'alternative_names' in row:
                    classification = row['classification']
                    alternative_names = row['alternative_names']
                    
                    # Extrai todos os nomes alternativos
                    names = self.extract_alternative_names(alternative_names)
                    
                    # Adiciona cada nome com sua classificação
                    for name in names:
                        if name:  # Verifica se o nome não está vazio
                            expanded_names.append({
                                'name': name,
                                'classification': classification
                            })
            
            # Cria DataFrame expandido
            df_expanded = pd.DataFrame(expanded_names)
            
            # Remove duplicatas, mantendo o primeiro registro encontrado
            df_expanded = df_expanded.drop_duplicates(subset=['name'], keep='first')
            
            # Constrói o dicionário de gêneros
            self.gender_dict = dict(zip(df_expanded['name'], df_expanded['classification']))
            self.loaded = True
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao carregar dados do IBGE: {str(e)}")
            return False
    
    def extract_first_name(self, full_name: str) -> str:
        """Extrai o primeiro nome de um nome completo"""
        if pd.isna(full_name):
            return ""
        
        normalized = self.normalize_name(full_name)
        if not normalized:
            return ""
        
        # Pega apenas o primeiro nome
        first_name = normalized.split()[0] if normalized.split() else ""
        return first_name
    
    def classify_gender(self, name: str) -> str:
        """Classifica o gênero de um nome"""
        first_name = self.extract_first_name(name)
        
        if not first_name:
            return "UNKNOWN"
        
        # Busca direta no dicionário
        if first_name in self.gender_dict:
            return self.gender_dict[first_name]
        
        return "UNKNOWN"

class GenderAnalyzer:
    def __init__(self):
        # Base extensiva de nomes masculinos
        self.masculine_names = {
            'joão', 'josé', 'antonio', 'francisco', 'carlos', 'paulo', 'pedro', 'lucas', 'luiz', 'marcos',
            'luis', 'gabriel', 'rafael', 'daniel', 'marcelo', 'bruno', 'eduardo', 'felipe', 'raimundo',
            'rodrigo', 'manoel', 'nelson', 'roberto', 'sebastião', 'miguel', 'andré', 'fernando', 'fabio',
            'leonardo', 'gustavo', 'guilherme', 'diego', 'sergio', 'ricardo', 'alessandro', 'wellington',
            'clayton', 'edson', 'ronaldo', 'mauricio', 'julio', 'cesar', 'ivan', 'jefferson', 'leandro',
            'jorge', 'rubens', 'anderson', 'wanderson', 'emerson', 'alexandre', 'marcio', 'thiago',
            'vinicius', 'mateus', 'henrique', 'wagner', 'cristiano', 'patrick', 'renato', 'adriano',
            'jean', 'everton', 'reginaldo', 'wesley', 'elder', 'washington', 'valdenir', 'valdir',
            'wilson', 'william', 'wallace', 'walter', 'wander', 'widson', 'willian', 'yuri', 'yan',
            'yago', 'jhonathan', 'andre', 'filipe', 'michael', 'david', 'john', 'james', 'robert',
            'william', 'richard', 'charles', 'joseph', 'thomas', 'christopher', 'paul', 'mark',
            'donald', 'steven', 'kenneth', 'andrew', 'joshua', 'kevin', 'brian', 'george'
        }
        
        # Base extensiva de nomes femininos
        self.feminine_names = {
            'maria', 'ana', 'francisca', 'antônia', 'adriana', 'juliana', 'márcia', 'fernanda',
            'patricia', 'aline', 'sandra', 'camila', 'amanda', 'bruna', 'jessica', 'leticia',
            'juliane', 'cristiane', 'michele', 'fabiana', 'debora', 'carolina', 'jéssica',
            'tatiane', 'vanessa', 'simone', 'claudia', 'monica', 'andrea', 'luciana', 'mariana',
            'gabriela', 'rosana', 'viviane', 'daniela', 'cristina', 'regina', 'carla', 'rita',
            'edna', 'roseli', 'aparecida', 'fatima', 'joana', 'terezinha', 'vera', 'lucia',
            'helena', 'isabel', 'celia', 'marta', 'solange', 'silvia', 'valeria', 'eliane',
            'sonia', 'denise', 'elizabeth', 'janete', 'marli', 'odete', 'neusa', 'rosa',
            'conceição', 'angela', 'sueli', 'marlene', 'ines', 'alice', 'neuza', 'iracema',
            'ivone', 'magali', 'benedita', 'zilda', 'yara', 'yvonne', 'yone', 'yasmin',
            'vitoria', 'virginia', 'vilma', 'vanda', 'ursula', 'ketlen', 'beatriz', 'thais',
            'daniele', 'rayane', 'layane', 'anny', 'vivian', 'mary', 'patricia', 'jennifer'
        }
        
        # Terminações que indicam gênero
        self.masculine_endings = {
            'son', 'o', 'os', 'ão', 'al', 'el', 'il', 'ol', 'ul', 'aldo', 'ando', 'ardo',
            'ário', 'átio', 'ávio', 'élio', 'ério', 'ésio', 'ício', 'ólio', 'úlio', 'ndo',
            'rdo', 'berto', 'dro', 'gro', 'tro', 'fredo', 'neto', 'etto', 'itto', 'otto',
            'utto', 'om', 'ulio', 'ohn', 'dy', 'ick'
        }
        
        self.feminine_endings = {
            'a', 'as', 'ã', 'ães', 'ana', 'ina', 'ona', 'una', 'anda', 'enda', 'inda',
            'onda', 'unda', 'ália', 'élia', 'ília', 'ólia', 'úlia', 'ária', 'éria',
            'íria', 'ória', 'úria', 'ância', 'ência', 'íncia', 'ôncia', 'úncia',
            'ette', 'ine', 'elle', 'lia', 'dia', 'nia'
        }
        
        # Nomes ambíguos
        self.ambiguous_names = {
            'alex', 'andrea', 'chris', 'dana', 'frances', 'jean', 'jordan', 'kelly',
            'leslie', 'lindsay', 'morgan', 'pat', 'robin', 'sam', 'taylor', 'terry',
            'tracy', 'val', 'sidney', 'wesley', 'nicola', 'dominique', 'rene', 'jose'
        }

    def clean_name(self, name: str) -> str:
        """Limpa e normaliza o nome para análise"""
        if not isinstance(name, str):
            return ""
        
        # Remove acentos
        name = unicodedata.normalize('NFD', name)
        name = ''.join(char for char in name if unicodedata.category(char) != 'Mn')
        
        # Converte para minúsculo e remove caracteres especiais
        name = re.sub(r'[^a-zA-Z\s]', '', name.lower().strip())
        
        return name

    def extract_first_name(self, full_name: str) -> str:
        """Extrai o primeiro nome de um nome completo"""
        clean_name = self.clean_name(full_name)
        if not clean_name:
            return ""
        
        # Remove títulos comuns
        titles = ['sr', 'sra', 'dr', 'dra', 'prof', 'profa', 'mr', 'mrs', 'ms', 'miss']
        words = clean_name.split()
        
        # Remove títulos do início
        while words and words[0] in titles:
            words.pop(0)
        
        if not words:
            return ""
        
        return words[0]

    def analyze_by_database(self, first_name: str) -> Tuple[str, str, float]:
        """Analisa o gênero baseado na base de dados de nomes"""
        first_name = first_name.lower().strip()
        
        if first_name in self.masculine_names:
            return "Masculino", "Base de dados", 0.95
        elif first_name in self.feminine_names:
            return "Feminino", "Base de dados", 0.95
        elif first_name in self.ambiguous_names:
            return "Incerto", "Nome ambíguo", 0.3
        
        return None, None, 0.0

    def analyze_by_endings(self, first_name: str) -> Tuple[str, str, float]:
        """Analisa o gênero baseado nas terminações do nome"""
        first_name = first_name.lower().strip()
        
        # Verifica terminações femininas
        for ending in sorted(self.feminine_endings, key=len, reverse=True):
            if first_name.endswith(ending):
                confidence = 0.7 if len(ending) > 2 else 0.6
                return "Feminino", f"Terminação '{ending}'", confidence
        
        # Verifica terminações masculinas
        for ending in sorted(self.masculine_endings, key=len, reverse=True):
            if first_name.endswith(ending):
                confidence = 0.7 if len(ending) > 2 else 0.6
                return "Masculino", f"Terminação '{ending}'", confidence
        
        return None, None, 0.0

    def analyze_gender(self, name: str) -> Dict[str, any]:
        """Análise principal do gênero"""
        if not isinstance(name, str) or not name.strip():
            return {
                'genero': "Incerto",
                'confianca': "Baixa",
                'razao': "Nome inválido"
            }
        
        first_name = self.extract_first_name(name)
        if not first_name:
            return {
                'genero': "Incerto",
                'confianca': "Baixa", 
                'razao': "Não foi possível extrair primeiro nome"
            }
        
        # Análise sequencial por prioridade
        analyses = [
            self.analyze_by_database(first_name),
            self.analyze_by_endings(first_name)
        ]
        
        # Pega a primeira análise válida
        gender, reason, confidence = None, None, 0.0
        for g, r, c in analyses:
            if g:
                gender, reason, confidence = g, r, c
                break
        
        # Se não encontrou nada, marca como incerto
        if not gender:
            gender, reason, confidence = "Incerto", "Análise inconclusiva", 0.2
        
        # Determina nível de confiança
        if confidence >= 0.8:
            conf_level = "Alta"
        elif confidence >= 0.6:
            conf_level = "Média"
        else:
            conf_level = "Baixa"
        
        return {
            'genero': gender,
            'confianca': conf_level,
            'razao': reason
        }

def convert_df_to_excel(df):
    """Converte DataFrame para Excel em bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    output.seek(0)
    return output.getvalue()

def main():
    st.title("🔍 Classificador de Gênero por Nome")
    st.markdown("---")
    
    # Sidebar com informações
    st.sidebar.title("ℹ️ Informações")
    st.sidebar.markdown("""
    ### Como usar:
    1. Faça upload do seu arquivo CSV ou Excel
    2. Selecione a coluna que contém os nomes
    3. Escolha o método de classificação
    4. Clique em 'Processar'
    5. Baixe o resultado
    
    ### Métodos disponíveis:
    - **IBGE**: Usa base oficial de nomes brasileiros
    - **Análise Avançada**: Usa algoritmos de padrões linguísticos
    - **Híbrido**: Combina ambos os métodos
    """)
    
    # Inicializar sessão
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Upload do arquivo
    st.subheader("📁 Upload do Arquivo")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV ou Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo deve conter uma coluna com nomes para classificação"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar arquivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
            
            # Mostrar preview dos dados
            st.subheader("👀 Preview dos Dados")
            st.dataframe(df.head(), use_container_width=True)
            
            # Seleção da coluna de nomes
            st.subheader("🎯 Configuração")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Detectar colunas que podem conter nomes
                name_columns = [col for col in df.columns if 
                               any(keyword in col.lower() for keyword in ['name', 'nome', 'client', 'customer', 'pessoa'])]
                
                if name_columns:
                    default_col = name_columns[0]
                    name_column = st.selectbox(
                        "Selecione a coluna com os nomes:",
                        options=df.columns,
                        index=list(df.columns).index(default_col)
                    )
                else:
                    name_column = st.selectbox(
                        "Selecione a coluna com os nomes:",
                        options=df.columns
                    )
            
            with col2:
                method = st.selectbox(
                    "Escolha o método de classificação:",
                    options=["IBGE", "Análise Avançada", "Híbrido"],
                    help="IBGE: Mais preciso para nomes brasileiros\nAnálise Avançada: Melhor para nomes internacionais\nHíbrido: Combina ambos"
                )
            
            # Mostrar amostra dos nomes selecionados
            if name_column:
                st.write("**Amostra dos nomes selecionados:**")
                sample_names = df[name_column].dropna().head(10).tolist()
                st.write(", ".join(sample_names))
            
            # Botão para processar
            if st.button("🚀 Processar Classificação", type="primary"):
                
                # Barra de progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Inicializar classificadores
                    ibge_classifier = GenderClassifier()
                    advanced_analyzer = GenderAnalyzer()
                    
                    # Carregar dados do IBGE se necessário
                    if method in ["IBGE", "Híbrido"]:
                        status_text.text("Carregando base de dados do IBGE...")
                        if not ibge_classifier.load_ibge_data_from_sheets():
                            st.error("❌ Erro ao carregar dados do IBGE. Verifique as credenciais.")
                            return
                        progress_bar.progress(0.3)
                    
                    # Processar dados
                    status_text.text("Processando nomes...")
                    results = []
                    
                    for idx, name in enumerate(df[name_column]):
                        # Atualizar progresso
                        progress = 0.3 + (idx / len(df)) * 0.7
                        progress_bar.progress(progress)
                        
                        if method == "IBGE":
                            gender = ibge_classifier.classify_gender(name)
                            gender_info = {
                                'genero': "Masculino" if gender == "M" else "Feminino" if gender == "F" else "Incerto",
                                'confianca': "Alta" if gender in ["M", "F"] else "Baixa",
                                'razao': "Base IBGE" if gender in ["M", "F"] else "Não encontrado na base"
                            }
                        
                        elif method == "Análise Avançada":
                            gender_info = advanced_analyzer.analyze_gender(name)
                        
                        else:  # Híbrido
                            # Primeiro tenta IBGE
                            ibge_gender = ibge_classifier.classify_gender(name)
                            if ibge_gender in ["M", "F"]:
                                gender_info = {
                                    'genero': "Masculino" if ibge_gender == "M" else "Feminino",
                                    'confianca': "Alta",
                                    'razao': "Base IBGE"
                                }
                            else:
                                # Se não encontrou no IBGE, usa análise avançada
                                gender_info = advanced_analyzer.analyze_gender(name)
                                gender_info['razao'] = f"Análise avançada - {gender_info['razao']}"
                        
                        results.append(gender_info)
                    
                    # Adicionar resultados ao DataFrame
                    df['Genero'] = [r['genero'] for r in results]
                    df['Confianca'] = [r['confianca'] for r in results]
                    df['Metodo'] = [r['razao'] for r in results]
                    
                    # Salvar na sessão
                    st.session_state.processed_data = df
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processamento concluído!")
                    
                except Exception as e:
                    st.error(f"❌ Erro durante o processamento: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
            
            # Mostrar resultados se processados
            if st.session_state.processed_data is not None:
                st.markdown("---")
                st.subheader("📊 Resultados")
                
                result_df = st.session_state.processed_data
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    masculinos = len(result_df[result_df['Genero'] == 'Masculino'])
                    st.metric("👨 Masculinos", masculinos)
                
                with col2:
                    femininos = len(result_df[result_df['Genero'] == 'Feminino'])
                    st.metric("👩 Femininos", femininos)
                
                with col3:
                    incertos = len(result_df[result_df['Genero'] == 'Incerto'])
                    st.metric("❓ Incertos", incertos)
                
                with col4:
                    alta_confianca = len(result_df[result_df['Confianca'] == 'Alta'])
                    st.metric("🎯 Alta Confiança", alta_confianca)
                
                # Gráfico de distribuição
                st.subheader("📈 Distribuição por Gênero")
                gender_counts = result_df['Genero'].value_counts()
                st.bar_chart(gender_counts)
                
                # Tabela com resultados
                st.subheader("📋 Dados Processados")
                st.dataframe(result_df, use_container_width=True)
                
                # Botões de download
                st.subheader("💾 Download")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download CSV
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Baixar CSV",
                        data=csv_data,
                        file_name=f"classificacao_genero_{uploaded_file.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download Excel
                    excel_data = convert_df_to_excel(result_df)
                    st.download_button(
                        label="📊 Baixar Excel",
                        data=excel_data,
                        file_name=f"classificacao_genero_{uploaded_file.name.split('.')[0]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Casos para revisão manual
                uncertain_cases = result_df[result_df['Genero'] == 'Incerto']
                if not uncertain_cases.empty:
                    st.subheader("⚠️ Casos para Revisão Manual")
                    st.write(f"**{len(uncertain_cases)} nomes** não puderam ser classificados com confiança:")
                    st.dataframe(uncertain_cases[[name_column, 'Genero', 'Confianca', 'Metodo']], use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    
    else:
        st.info("👆 Faça upload de um arquivo CSV ou Excel para começar")
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔍 Classificador de Gênero por Nome | Desenvolvido com Streamlit</p>
        <p>💡 Dica: Para melhores resultados, use nomes limpos sem títulos ou abreviações</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()