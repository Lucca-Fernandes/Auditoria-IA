Sistema de Auditoria Automática de Aulas
 
Bem-vindo ao Sistema de Auditoria Automática de Aulas, uma solução desenvolvida para o Projeto Desenvolve, uma iniciativa que capacita jovens e adultos para o mercado de tecnologia com foco em inclusão social e empregabilidade. Este sistema automatiza a avaliação de aulas gravadas, garantindo qualidade educacional, eficiência operacional e feedback prático para professores, alinhado à missão de transformar vidas por meio da educação.
📖 Sobre o Projeto
O Projeto Desenvolve promove formação tecnológica em parceria com prefeituras do interior do Brasil, oferecendo uma jornada de aprendizado que vai do básico à empregabilidade. Em sua fase de reestruturação, o projeto busca eliminar tarefas manuais e modernizar processos. Este sistema atende a essa necessidade, automatizando a auditoria de aulas gravadas com inteligência artificial (IA) e integração com o Monday.com.
Objetivos

Escalabilidade: Analisar grandes quantidades de aulas sem intervenção manual.
Qualidade Educacional: Garantir aulas com alto padrão de conteúdo e didática.
Eficiência: Reduzir o tempo de auditorias de semanas para minutos.
Feedback Prático: Fornecer sugestões claras para professores melhorarem.
Inclusão: Tornar o processo acessível com uma interface web simples.

Funcionalidades

Upload de Vídeos: Interface web (Flask) para envio de aulas.
Transcrição Automática: Extrai e transcreve áudio com Whisper.
Análise de Linguagem Corporal: Avalia dinamismo com OpenCV.
Avaliação por IA: Atribui notas e sugestões com Gemini 1.5 Flash.
Relatórios PDF: Gera documentos profissionais com LaTeX.
Integração com Monday.com: Centraliza resultados em um board dedicado.
Retorno Imediato: Fornece relatórios PDF para download.

🚀 Como Funciona
O sistema avalia aulas com base em sete critérios:

Qualidade Técnica do Conteúdo: Precisão e domínio do tema.
Linguagem Corporal: Dinamismo do professor.
Tom de Voz: Clareza e engajamento.
Clareza do Roteiro: Organização e transições.
Ritmo: Fluidez e uso de pausas.
Didática: Capacidade de engajar e ensinar.
Qualidade Geral: Impressão geral da aula.

Fluxo

O usuário faz upload de um vídeo via interface web.
O áudio é extraído e transcrito.
A linguagem corporal é analisada por movimento.
A IA avalia os critérios, atribuindo notas e sugestões.
Um relatório PDF é gerado e enviado ao Monday.com.
O PDF é retornado ao usuário.

🛠️ Tecnologias Utilizadas

Python 3.8+: Linguagem principal.
Whisper (tiny): Transcrição de áudio em português.
OpenCV: Análise de movimento.
Gemini 1.5 Flash: Avaliação por IA via API.
LaTeX (PyLaTeX, Latexmk, XeLaTeX): Geração de relatórios PDF.
Flask: Interface web para upload.
Monday.com API: Integração para gestão.
NLTK: Tokenização de texto.

📋 Pré-requisitos

Python 3.8+
Ferramentas Externas:
FFmpeg
Latexmk e XeLaTeX (pacotes: texlive-xetex, texlive-fonts-extra)


APIs: Chaves válidas para Gemini e Monday.com
Sistema Operacional: Linux, macOS ou Windows (com FFmpeg configurado)

⚙️ Instalação

Clone o repositório:
git clone https://github.com/seu-usuario/auditoria-aula.git
cd auditoria-aula


Crie um ambiente virtual:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Instale as dependências:
pip install whisper opencv-python numpy pylatex requests nltk flask werkzeug


Instale FFmpeg:

Linux: sudo apt-get install ffmpeg
macOS: brew install ffmpeg
Windows: Baixe e adicione ao PATH.


Instale Latexmk e XeLaTeX:

Linux: sudo apt-get install latexmk texlive-xetex texlive-fonts-extra
macOS: brew install mactex
Windows: Instale MikTeX.


Configure recursos NLTK:
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


Configure variáveis de ambiente:Crie um arquivo .env ou edite auditoria_aula.py com:
API_KEY=seu_gemini_api_key
MONDAY_API_TOKEN=seu_monday_token
MONDAY_BOARD_ID=seu_board_id
MONDAY_FILES_COLUMN_ID=seu_files_column_id



🚀 Uso

Inicie o servidor:
python auditoria_aula.py


Acesse a interface:

Abra http://localhost:5000 no navegador.
Faça upload de um vídeo de aula.


Receba resultados:

O sistema gera um relatório PDF para download.
Resultados (notas, decisão, sugestões) são enviados ao Monday.com.


Consulte no Monday.com:

Verifique o board configurado para ver o item criado com o PDF anexado.




🛑 Limitações

Transcrição: Pode falhar em áudios ruidosos (modelo Whisper tiny).
Linguagem Corporal: Análise simplificada (baseada em movimento).
Dependência de APIs: Latência ou falhas externas podem afetar o sistema.
Hardware: Treinamento de modelos próprios foi limitado por recursos computacionais.

🌱 Melhorias Futuras

Treinar um modelo de IA específico para os conteúdos do Projeto Desenvolve.
Usar modelos Whisper maiores (ex.: medium ou large) para maior precisão.
Implementar análise avançada de gestos com YOLO.
Criar dashboards no Monday.com para visualização em tempo real.
Adicionar validação de vídeos (formato, tamanho) antes do upload.

💡 Desafios e Soluções

Limitações de Hardware: Não foi possível treinar um modelo próprio; solução: uso de APIs (Gemini) para avaliações.
Integração com Monday.com: Erros de parsing (ex.: "Bad Request") foram corrigidos com ajustes na query GraphQL.
Geração de PDF: Relatórios inconsistentes foram resolvidos com mapeamento preciso de dados na função evaluate_lesson.
Transcrição: Áudios ruidosos afetaram o modelo Whisper tiny; recomendação: testar modelos maiores.

📜 Licença
Este projeto é licenciado sob a Licença MIT.
🙌 Contribuições
Contribuições são bem-vindas! Para sugerir melhorias ou relatar problemas:

Fork o repositório.
Crie uma branch: git checkout -b minha-melhoria.
Faça commit: git commit -m "Minha melhoria".
Envie um pull request.

📬 Contato
Para dúvidas ou feedback, entre em contato com [lfdc212@gmail.com].

Desenvolvido com 💙 para o Projeto Desenvolve, promovendo educação inclusiva e impacto social.
