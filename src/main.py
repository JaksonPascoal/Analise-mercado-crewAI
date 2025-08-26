import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai_tools import SerperDevTool

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configura as APIs
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# 1. Definir o modelo (LLM) que os agentes usarão
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True)

# 2. Definir as ferramentas que os agentes terão acesso
search_tool = SerperDevTool()

# 3. Definir os Agentes
research_agent = Agent(
    role='Pesquisador de Mercado Sênior',
    goal='Encontrar as últimas notícias, tendências e dados do mercado sobre um tema específico.',
    backstory='Você é um especialista em análise de mercado, com um olho afiado para capturar informações valiosas e tendências emergentes. Sua missão é fornecer a base de dados sólida para a equipe.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[search_tool]
)

analyst_agent = Agent(
    role='Analista de Estratégia',
    goal='Analisar as informações coletadas e gerar um relatório estratégico com insights e recomendações.',
    backstory='Você é um estrategista experiente, capaz de transformar dados brutos em insights acionáveis e planos de ação claros. Sua expertise é fundamental para guiar a tomada de decisão.',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# 4. Definir as Tarefas
research_task = Task(
    description='''Conduza uma pesquisa exaustiva sobre "o futuro da Inteligência Artificial em 2025". Identifique as principais tendências, avanços tecnológicos e os players mais importantes do mercado.''',
    expected_output='''Um relatório detalhado e bem estruturado, com os tópicos-chave, incluindo: tendências, avanços e análise dos principais players. As informações devem ser recentes e verificadas.''',
    agent=research_agent
)

analysis_task = Task(
    description='''Use as informações do relatório de pesquisa para gerar um relatório de estratégia. O relatório deve conter:
    - Um resumo executivo claro.
    - Insights sobre as oportunidades de mercado e os principais riscos.
    - Recomendações práticas para empresas que desejam investir no setor de IA.''',
    expected_output='''Um relatório estratégico completo, em formato de texto, com os pontos solicitados, pronto para ser apresentado a stakeholders.''',
    agent=analyst_agent
)

# 5. Criar a equipe (Crew)
project_crew = Crew(
    agents=[research_agent, analyst_agent],
    tasks=[research_task, analysis_task],
    verbose=2,
    process=Process.sequential
)

# 6. Iniciar o processo e ver a equipe em ação
result = project_crew.kickoff()

# 7. Imprimir o resultado final
print("\n##################################")
print("## Relatório de Análise Final ##")
print("##################################\n")
print(result)