import whisper
import cv2
import numpy as np
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import NoEscape
import requests
import time
import re
import unicodedata
import os
import nltk
from nltk.tokenize import sent_tokenize
import subprocess
import datetime
import json
from flask import Flask, request, send_file, jsonify, make_response
from werkzeug.utils import secure_filename

# Download necessário para NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    if not nltk.data.find('tokenizers/punkt_tab'):
        raise Exception("Recurso punkt_tab não encontrado após download.")
except Exception as e:
    print(f"Erro ao baixar recursos do NLTK: {str(e)}. Continuando sem tokenização avançada.")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configurações
APPROVAL_THRESHOLD = 60
MODEL_WHISPER = "tiny"
API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
API_KEY = "AIzaSyCzZgOEtBMbws6AKcYmmsfLDMbP8MRL9nA"  
MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_API_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJ0aWQiOjUyMjgzNTYwNSwiYWFpIjoxMSwidWlkIjo3Njg3MjI4MSwiaWFkIjoiMjAyNS0wNi0wNVQyMzozMjowMy4wMDBaIiwicGVyIjoibWU6d3JpdGUiLCJhY3RpZCI6Mjk4NTQ1MjcsInJnbiI6InVzZTEifQ.WoKfXLkChx1AU16TafgaTOXALV2vO3T1sDh-I7JkSjg"
MONDAY_BOARD_ID = "9308063155" 
MONDAY_FILES_COLUMN_ID = "file_mkrmt6km" 

def transcribe_audio(video_path):
    try:
        print("Carregando modelo Whisper...")
        model = whisper.load_model(MODEL_WHISPER)
        audio_file = "temp_audio.wav"
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        print("Extraindo áudio com FFmpeg...")
        os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}")
        
        print("Transcrevendo áudio...")
        result = model.transcribe(audio_file, language="pt")
        transcription = result["text"]
        print(f"Transcrição completa: {transcription}")
        
        if os.path.exists(audio_file):
            os.remove(audio_file)
        return transcription
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return ""

def analyze_body_language(video_path):
    try:
        print("Abrindo vídeo para análise de linguagem corporal...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        prev_frame = None
        movement_score = 0
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                movement = np.mean(diff)
                movement_score += movement
                frame_count += 1
            prev_frame = gray
        
        cap.release()
        movement_score = movement_score / frame_count if frame_count > 0 else 0
        movement_score = min(100, movement_score * 10)
        print(f"Pontuação de linguagem corporal: {movement_score:.1f}")
        return movement_score
    except Exception as e:
        print(f"Erro na análise de linguagem corporal: {str(e)}")
        return 50

def parse_gemini_output(gemini_output):
   
    justifications = {}
    suggestions = {}
    final_score = 0.0
    decision = "Reprovada"

    lines = gemini_output.strip().split('\n')

    crit_regex = re.compile(r"Critério:\s*(.*?),\s*Nota:\s*([\d.]+),\s*Justificativa:\s*(.*)")
    sugg_regex = re.compile(r"Sugestão para\s*(.*?):\s*(.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match_crit = crit_regex.match(line)
        if match_crit:
            criterion_full_name = match_crit.group(1).strip()
            score_str = match_crit.group(2).strip()
            justification_text = match_crit.group(3).strip()
            
            normalized_name = normalize_criterion_name(criterion_full_name)
            justifications[normalized_name] = {
                "full_name": criterion_full_name,
                "score": float(score_str),
                "text": justification_text
            }
            continue

        match_sugg = sugg_regex.match(line)
        if match_sugg:
            criterion_sugg_full_name = match_sugg.group(1).strip()
            suggestion_text = match_sugg.group(2).strip()
            
            normalized_name = normalize_criterion_name(criterion_sugg_full_name)
            suggestions[normalized_name] = suggestion_text
            continue

        if line.startswith("Nota Final:"):
            try:
                score_part = line.split(":")[1].strip().split(',')[0]
                final_score = float(score_part)
            except ValueError:
                print(f"Erro ao parsear Nota Final: '{line}'")
                final_score = 0.0
            continue

        if line.startswith("Decisão:"):
            decision = line.split(":")[1].strip()
            continue

    return {
        "final_score": final_score,
        "decision": decision,
        "justifications": justifications,
        "suggestions": suggestions
    }

def normalize_criterion_name(name):
    name = re.sub(r'\(.*?\)', '', name)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()

def evaluate_lesson(transcription, movement_score):
    if not transcription:
        print("Transcrição vazia. Retornando valores padrão.")
        return {
            "final_score": 50,
            "decision": "Reprovada",
            "justifications": {
                "qualidade_tecnica": {"full_name": "Qualidade Técnica do Conteúdo Transmitido", "score": 50, "text": "Transcrição vazia ou inválida."},
                "linguagem_corporal": {"full_name": "Linguagem Corporal do Professor", "score": 50, "text": "Transcrição vazia ou inválida."},
                "tom_de_voz": {"full_name": "Tom de Voz", "score": 50, "text": "Transcrição vazia ou inválida."},
                "clareza_roteiro": {"full_name": "Clareza e Estrutura do Roteiro", "score": 50, "text": "Transcrição vazia ou inválida."},
                "ritmo": {"full_name": "Ritmo da Apresentação", "score": 50, "text": "Transcrição vazia ou inválida."},
                "didatica": {"full_name": "Didática", "score": 50, "text": "Transcrição vazia ou inválida."},
                "qualidade_geral": {"full_name": "Qualidade Geral da Aula", "score": 50, "text": "Transcrição vazia ou inválida."}
            },
            "suggestions": {
                "qualidade_tecnica": "Verifique o vídeo e a configuração do Whisper.",
                "linguagem_corporal": "Verifique o vídeo e a configuração do Whisper.",
                "tom_de_voz": "Verifique o vídeo e a configuração do Whisper.",
                "clareza_roteiro": "Verifique o vídeo e a configuração do Whisper.",
                "ritmo": "Verifique o vídeo e a configuração do Whisper.",
                "didatica": "Verifique o vídeo e a configuração do Whisper.",
                "qualidade_geral": "Verifique o vídeo e a configuração do Whisper."
            },
            "scores": {
                "qualidade_tecnica": 50,
                "linguagem_corporal": 50,
                "tom_de_voz": 50,
                "clareza_roteiro": 50,
                "ritmo": 50,
                "didatica": 50,
                "qualidade_geral": 50
            }
        }

    try:
        print("Tokenizando transcrição...")
        sentences = nltk.sent_tokenize(transcription, language="portuguese")
        if not sentences:
            sentences = [transcription]
    except Exception as e:
        print(f"Erro na tokenização: {str(e)}")
        sentences = [transcription]

    prompt = f"""
    Você é um auditor de aulas altamente inteligente, experiente e rigoroso. Sua função é analisar detalhadamente a transcrição de aulas, identificar todos os pontos fracos e fortes, e fornecer feedback construtivo e preciso. Sua análise deve buscar a excelência e a qualidade, sendo imparcial e contextualizada.

**Instruções Essenciais para Avaliação:**
- A **Qualidade Técnica do Conteúdo Transmitido** é o critério de **MAIOR PESO**. Sua avaliação deve ser **EXCLUSIVAMENTE** baseada em:
    - **Domínio do Assunto:** O professor demonstra profundo conhecimento?
    - **Precisão Conceitual:** As explicações são corretas e sem ambiguidades?
    - **Ausência de Erros:** Há erros de código, informações incorretas ou retrabalho excessivo?
    - **Fluidez na Execução:** A demonstração prática é suave e sem hesitações técnicas ou repetições desnecessárias de ações (como copiar/colar várias vezes a mesma coisa)?
    - **Clareza na Explicação Técnica:** O conteúdo técnico é apresentado de forma compreensível e direta?
- **Problemas técnicos graves (erros de código, imprecisões conceituais, insegurança visível no domínio técnico) devem resultar em PONTUAÇÕES MUITO BAIXAS para a Qualidade Técnica e impactar severamente a Nota Final.** A detecção de falhas críticas nesse aspecto pode levar à reprovação imediata.
- **Sobre a Informalidade (aplicável a Tom de Voz e Didática):**
    - A linguagem informal (ex: "bora pro vídeo", "a gente") **não é inerentemente negativa**.
    - Avalie se a informalidade contribui para o engajamento do aluno e a conexão com o professor.
    - Penalize a informalidade apenas quando ela se torna **excessiva, repetitiva (vícios de linguagem como "né", "ok", "aí" em demasia), confusa, ou prejudica a clareza e a credibilidade técnica** do conteúdo.
- Seja exigente nas notas, mas justo. Evite notas "padrão". Cada aula deve ser avaliada por seus méritos e deméritos específicos.
- As justificativas e sugestões devem ser extremamente específicas, acionáveis e diretas, sem floreios.

# Exemplos de Relatórios de Auditoria de Aula (para referência de estilo e profundidade esperados)

## Exemplo de Aula de Qualidade RUIM (DEVE SER REPROVADA):
### Análise Humana Resumida: O professor não demonstra habilidades com o componente, demonstrando insegurança e pouca dinâmica nas aulas. Além de errar bastante com os códigos e ficar refazendo a mesma coisa diversas vezes.
Critério: Qualidade Técnica do Conteúdo Transmitido, Nota: 15, Justificativa: O professor demonstrou clara falta de domínio do assunto, com erros frequentes de código e necessidade constante de refazer operações básicas. A insegurança técnica foi perceptível ao longo de toda a aula.
Critério: Linguagem Corporal do Professor, Nota: 30, Justificativa: A descrição de movimento indica pouca variação, sugerindo rigidez ou falta de expressividade que impacta o engajamento. A postura estática não complementa a comunicação verbal.
Critério: Tom de Voz, Nota: 40, Justificativa: O tom de voz foi monótono, sem variações que pudessem manter a atenção do aluno. O uso **excessivo e repetitivo de interjeições (ex: "né", "ok")** e a falta de dinamismo prejudicaram a fluidez e a percepção profissional.
Critério: Clareza e Estrutura do Roteiro, Nota: 20, Justificativa: O roteiro da aula foi completamente desorganizado e sem uma linha lógica. O professor saltou entre tópicos sem transições claras, gerando confusão.
Critério: Ritmo da Apresentação, Nota: 25, Justificativa: O ritmo foi inconsistentemente lento, com pausas excessivas e divagações. A aula não progrediu de forma eficiente, desperdiçando o tempo do espectador.
Critério: Didática, Nota: 10, Justificativa: A didática foi inexistente, com explicações confusas e ausência total de exemplos práticos. O professor não demonstrou capacidade de transmitir o conhecimento de forma eficaz.
Critério: Qualidade Geral da Aula, Nota: 15, Justificativa: A aula foi de baixíssima qualidade, com falhas críticas em todos os aspectos, especialmente na técnica. É inviável para fins educacionais.
Sugestão para Qualidade Técnica do Conteúdo Transmitido: Revisar exaustivamente o conteúdo técnico e praticar a execução dos exemplos para garantir fluidez e correção.
Sugestão para Linguagem Corporal do Professor: Incorporar gestos e expressões faciais mais dinâmicas para acompanhar a fala e engajar o público.
Sugestão para Tom de Voz: Praticar a modulação da voz para criar variações e manter o interesse do aluno. Minimizar o uso excessivo de interjeições que não agregam valor.
Sugestão para Clareza e Estrutura do Roteiro: Desenvolver um roteiro detalhado e seguir uma estrutura lógica clara, com introdução, desenvolvimento e conclusão bem definidos.
Sugestão para Ritmo da Apresentação: Cronometrar as seções da aula para garantir um ritmo adequado e evitar pausas prolongadas ou pressa.
Sugestão para Didática: Utilizar exemplos práticos e analogias claras, além de interagir mais com o conteúdo para facilitar a compreensão do aluno.
Sugestão para Qualidade Geral da Aula: Realizar um planejamento completo e uma simulação da aula antes da gravação final.
Nota Final: 22, Decisão: Reprovada.

## Exemplo de Aula de Qualidade MÉDIA (Pode ser Aprovada/Reprovada dependendo da nota final, mas com sugestões para aprimoramento):
### Análise Humana Resumida: O professor tem conhecimento sobre o assunto, mas a aula não é tão atrativa quanto outros componentes. Apresenta alguns momentos de hesitação e a didática poderia ser mais envolvente.
Critério: Qualidade Técnica do Conteúdo Transmitido, Nota: 70, Justificativa: O conteúdo técnico apresentado é preciso e relevante para o tema da aula. No entanto, houve algumas pequenas hesitações na execução do código que poderiam ser mais fluidas. O domínio do assunto é evidente.
Critério: Linguagem Corporal do Professor, Nota: 65, Justificativa: O nível de movimento demonstra uma presença razoável, mas pode ser mais expressivo. A postura pode ser aprimorada para transmitir maior confiança e dinamismo.
Critério: Tom de Voz, Nota: 60, Justificativa: O tom de voz é claro e com variação razoável. O uso de **linguagem informal (ex: "bora pro vídeo")** contribui para o engajamento, mas a presença ocasional de vícios de linguagem pode ser minimizada para otimizar a clareza.
Critério: Clareza e Estrutura do Roteiro, Nota: 68, Justificativa: O roteiro da aula é compreensível, mas em alguns momentos a transição entre tópicos não é fluida. A organização geral permite o acompanhamento, mas há espaço para maior concisão.
Critério: Ritmo da Apresentação, Nota: 62, Justificativa: O ritmo é aceitável, mas há pequenas pausas que quebram o fluxo da apresentação. A aula flui, mas pode ser otimizada para ser mais dinâmica.
Critério: Didática, Nota: 58, Justificativa: A didática é funcional, mas a aula carece de exemplos práticos mais envolventes ou analogias que facilitem a compreensão. A informalidade presente é neutra para o engajamento, mas a profundidade da explicação pode ser maior.
Critério: Qualidade Geral da Aula, Nota: 64, Justificativa: A aula é informativa e o professor domina o assunto, mas a apresentação geral não cativa completamente. Com alguns ajustes, a aula tem potencial para ser muito mais impactante.
Sugestão para Qualidade Técnica do Conteúdo Transmitido: Praticar a execução de trechos de código para garantir fluidez e eliminar hesitações.
Sugestão para Linguagem Corporal do Professor: Adicionar gestos mais expressivos e intencionais para complementar a fala e aumentar o engajamento visual.
Sugestão para Tom de Voz: Buscar um equilíbrio entre o engajamento proporcionado pela informalidade e a minimização de repetições e vícios de linguagem para maior profissionalismo.
Sugestão para Clareza e Estrutura do Roteiro: Otimizar as transições entre os tópicos, utilizando frases de ligação mais claras e resumos parciais.
Sugestão para Ritmo da Apresentação: Identificar e reduzir pausas desnecessárias, e praticar a fluidez das falas para manter um ritmo mais consistente.
Sugestão para Didática: Incluir exemplos mais elaborados ou casos de uso reais que reforcem o aprendizado e tornem a aula mais interativa.
Nota Final: 64.7, Decisão: Aprovada.

## Exemplo de Aula de Qualidade BOA (DEVE SER APROVADA):
### Análise Humana Resumida: Professores tiveram total domínio do assunto e fizeram a aula orientada por projeto, o que facilita o ensino para o estudante, além de conseguir criar uma ordem de continuidade com início, meio e fim.
Critério: Qualidade Técnica do Conteúdo Transmitido, Nota: 95, Justificativa: O professor demonstrou domínio completo do assunto, com explicações precisas e execução impecável do código. Não foram identificados erros, imprecisões técnicas ou hesitações.
Critério: Linguagem Corporal do Professor, Nota: 90, Justificativa: O nível de movimento indica uma linguagem corporal dinâmica e engajadora, que complementa a fala. A postura confiante e os gestos foram adequados ao contexto.
Critério: Tom de Voz, Nota: 92, Justificativa: O tom de voz é claro, modulado e com excelente projeção, mantendo o interesse do aluno. A entonação é variada e contribui para a clareza da explicação, utilizando a informalidade de forma estratégica e controlada para conectar-se com o público.
Critério: Clareza e Estrutura do Roteiro, Nota: 98, Justificativa: O roteiro da aula é exemplarmente claro e bem estruturado, com uma progressão lógica de início, meio e fim. A transição entre tópicos é suave e eficiente.
Critério: Ritmo da Apresentação, Nota: 95, Justificativa: O ritmo da apresentação é excelente, permitindo que o aluno absorva o conteúdo sem sentir pressa ou tédio. As pausas são estratégicas e bem utilizadas.
Critério: Didática, Nota: 96, Justificativa: A didática é altamente eficaz, com uso inteligente de exemplos práticos e uma metodologia orientada a projetos que facilita o aprendizado. O professor é engajador e inspira confiança, inclusive através do uso de uma linguagem acessível e pontualmente informal que beneficia o aprendizado.
Critério: Qualidade Geral da Aula, Nota: 95, Justificativa: A aula é de altíssima qualidade, superando as expectativas em todos os critérios. Representa um excelente recurso educacional.
Nota Final: 95.9, Decisão: Aprovada.

---

Agora, baseado na transcrição e nos dados fornecidos, avalie a aula atual:

- Transcrição: {transcription} (completa)
- Nível de movimento (linguagem corporal): {movement_score}

Avalie os seguintes critérios (atribua uma nota de 0 a 100 para cada e explique brevemente, com variações baseadas no conteúdo):
1. Qualidade técnica do conteúdo (clareza, precisão, relevância, ausência de erros, fluidez na execução, domínio do componente)
2. Linguagem corporal (baseado no movimento fornecido e inferências sobre postura/expressão)
3. Tom de voz (inferido da estrutura e escolha de palavras, entonação, dinamismo, equilíbrio entre formalidade e engajamento, presença de vícios de linguagem)
4. Clareza e estrutura do roteiro (organização, lógica, transições, concisão, eliminação de redundâncias)
5. Ritmo da apresentação (pausas, fluidez, dinamismo, eficiência no tempo)
6. Didática (engajamento, uso de exemplos, analogias, explicações aprofundadas, metodologia de ensino, impacto da informalidade no aprendizado)
7. Qualidade geral (impressão geral da aula, potencial de aprendizado, refinamento)

Forneça:
- Uma nota final (média dos critérios)
- Decisão: Aprovar (nota >= {APPROVAL_THRESHOLD}) ou Reprovada
- Justificativas detalhadas para cada critério (mínimo de 2 frases por critério, foco em pontos fracos e áreas de melhoria)
- Sugestões de melhoria (se reprovada ou se a nota do critério for inferior a 70, específicas e acionáveis, exatamente 1 sugestão por critério. Se a aula for aprovada e todas as notas forem >= 70, não é preciso fornecer sugestões)

Responda em português e em formato estruturado, **sem adicionar cabeçalhos ou títulos extras**. Mantenha **exatamente** o formato solicitado para cada linha:
Critério: [nome do critério], Nota: [valor numérico], Justificativa: [texto da justificativa].
Sugestão para [nome do critério]: [texto da sugestão].
Nota Final: [valor numérico], Decisão: [Aprovada/Reprovada]
    """

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        print("Enviando requisição para Gemini 1.5 Flash...")
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=data)
        response.raise_for_status()
        output = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"Resposta bruta da API (completa): {output}")

        evaluation = parse_gemini_output(output)

        # Mapeamento de critérios normalizados para chaves internas
        criterion_mapping = {
            "qualidade tecnica do conteudo": "qualidade_tecnica",
            "linguagem corporal": "linguagem_corporal",
            "tom de voz": "tom_de_voz",
            "clareza e estrutura do roteiro": "clareza_roteiro",
            "ritmo da apresentacao": "ritmo",
            "didatica": "didatica",
            "qualidade geral": "qualidade_geral"
        }

        # Reorganizar justifications e suggestions para as chaves internas
        justifications = {}
        suggestions = {}
        for norm_name, data in evaluation["justifications"].items():
            internal_key = criterion_mapping.get(norm_name)
            if internal_key:
                justifications[internal_key] = data

        for norm_name, text in evaluation["suggestions"].items():
            internal_key = criterion_mapping.get(norm_name)
            if internal_key:
                suggestions[internal_key] = text

        # Calcular scores a partir das justificativas
        scores = {k: v["score"] for k, v in justifications.items()}

        # Preencher valores padrão para critérios não retornados pela API
        all_criteria = {"qualidade_tecnica", "linguagem_corporal", "tom_de_voz", "clareza_roteiro", "ritmo", "didatica", "qualidade_geral"}
        for crit in all_criteria:
            if crit not in scores:
                scores[crit] = 50
                justifications[crit] = {"full_name": next((k for k, v in criterion_mapping.items() if v == crit), crit).replace("_", " ").title(), "score": 50, "text": "Avaliação não fornecida pela API."}
            if crit not in suggestions:
                suggestions[crit] = "Nenhuma sugestão específica fornecida."

        final_score = evaluation["final_score"]
        decision = "Aprovada" if final_score >= APPROVAL_THRESHOLD else "Reprovada"

        print(f"Nota final (API): {final_score:.1f}, Decisão (recalculada): {decision}")
        print(f"Scores: {scores}")
        print(f"Justifications: {justifications}")
        print(f"Suggestions: {suggestions}")

        return {
            "final_score": final_score,
            "decision": decision,
            "justifications": justifications,
            "suggestions": suggestions,
            "scores": scores
        }

    except requests.RequestException as e:
        print(f"Erro na chamada da API: {str(e)}")
        return {
            "final_score": 50,
            "decision": "Reprovada",
            "justifications": {
                "qualidade_tecnica": {"full_name": "Qualidade Técnica do Conteúdo Transmitido", "score": 50, "text": f"Erro na API: {str(e)}"},
                "linguagem_corporal": {"full_name": "Linguagem Corporal do Professor", "score": 50, "text": "Erro na API."},
                "tom_de_voz": {"full_name": "Tom de Voz", "score": 50, "text": "Erro na API."},
                "clareza_roteiro": {"full_name": "Clareza e Estrutura do Roteiro", "score": 50, "text": "Erro na API."},
                "ritmo": {"full_name": "Ritmo da Apresentação", "score": 50, "text": "Erro na API."},
                "didatica": {"full_name": "Didática", "score": 50, "text": "Erro na API."},
                "qualidade_geral": {"full_name": "Qualidade Geral da Aula", "score": 50, "text": "Erro na API."}
            },
            "suggestions": {},
            "scores": {
                "qualidade_tecnica": 50,
                "linguagem_corporal": 50,
                "tom_de_voz": 50,
                "clareza_roteiro": 50,
                "ritmo": 50,
                "didatica": 50,
                "qualidade_geral": 50
            }
        }

def escape_latex(text):
    if text is None:
        return ""
    
    text = str(text)
    text = text.replace('\n', ' ')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('<', '\\textless{}')
    text = text.replace('>', '\\textgreater{}')
    text = text.replace('\'', '\'')
    text = text.replace('"', '``')
    text = text.replace('...', '.\protect\ldots')
    text = text.replace('|', '\\textbar{}')
    return text

def generate_pdf_report(evaluation):
    try:
        print("Criando documento LaTeX...")
        doc = Document('audit_report', documentclass='memoir')

        doc.preamble.append(Command('usepackage', 'fontenc', 'T1'))
        doc.preamble.append(Command('usepackage', 'lmodern'))
        doc.preamble.append(Command('usepackage', 'textcomp'))
        doc.preamble.append(Command('usepackage', 'inputenc', 'utf8'))
        doc.preamble.append(Command('usepackage', 'babel', 'brazilian'))
        doc.preamble.append(Command('usepackage', 'geometry'))
        doc.preamble.append(NoEscape(r'\geometry{a4paper, margin=0.8in}'))
        doc.preamble.append(Command('usepackage', 'titlesec'))
        doc.preamble.append(Command('usepackage', 'fancyhdr'))
        doc.preamble.append(Command('usepackage', 'xcolor'))
        doc.preamble.append(Command('usepackage', 'noto'))
        doc.preamble.append(Command('usepackage', 'hyperref'))
        doc.preamble.append(NoEscape(r'\hypersetup{colorlinks=true, urlcolor=blue, linkcolor=blue}'))

        doc.preamble.append(NoEscape(r'\titleformat{\section}{\normalfont\Large\bfseries\color{blue!80!black}}{\thesection}{1em}{}'))
        doc.preamble.append(NoEscape(r'\titleformat{\subsection}{\normalfont\large\bfseries\color{blue!60!black}}{\thesubsection}{1em}{}'))

        doc.preamble.append(NoEscape(r'\pagestyle{fancy}'))
        doc.preamble.append(NoEscape(r'\fancyhf{}'))
        doc.preamble.append(NoEscape(r'\fancyhead[L]{\color{gray}\small Relatório de Auditoria de Aula}'))
        doc.preamble.append(NoEscape(r'\fancyhead[R]{\color{gray}\small \today}'))
        doc.preamble.append(NoEscape(r'\fancyfoot[C]{\color{gray}\thepage}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\headrulewidth}{0.4pt}'))
        doc.preamble.append(NoEscape(r'\renewcommand{\footrulewidth}{0.4pt}'))

        doc.preamble.append(NoEscape(r'\setmainfont{Noto Serif}'))

        doc.preamble.append(Command('title', NoEscape(r'Relatório de Auditoria de Aula \\ \large Análise Automatizada')))
        doc.preamble.append(Command('author', ''))
        doc.preamble.append(Command('date', r'\today'))

        doc.append(NoEscape(r'\maketitle'))
        doc.append(NoEscape(r'\vspace{1cm}'))

        with doc.create(Section('Resultados da Análise Automática')):
            doc.append('Este relatório apresenta os resultados detalhados da análise automática da aula.')
            doc.append(NoEscape(r'\vspace{0.5cm}'))

            with doc.create(Subsection('Nota Final')):
                doc.append(NoEscape(f'\\textbf{{{evaluation["final_score"]:.1f}}}'))
                doc.append(NoEscape(r'\newline'))

            with doc.create(Subsection('Decisão')):
                decision_color = 'green!60!black' if evaluation['decision'] == 'Aprovada' else 'red!60!black'
                doc.append(NoEscape(f'\\color{{{decision_color}}}{escape_latex(evaluation["decision"])}'))
                doc.append(NoEscape(r'\newline'))

            with doc.create(Subsection('Justificativas')):
                display_to_key_map = {
                    "Qualidade Técnica do Conteúdo Transmitido": "qualidade_tecnica",
                    "Linguagem Corporal do Professor": "linguagem_corporal",
                    "Tom de Voz": "tom_de_voz",
                    "Clareza e Estrutura do Roteiro": "clareza_roteiro",
                    "Ritmo da Apresentação": "ritmo",
                    "Didática": "didatica",
                    "Qualidade Geral da Aula": "qualidade_geral"
                }

                for display_name, internal_key in display_to_key_map.items():
                    score_val = evaluation['scores'].get(internal_key, 50)
                    justification_text = evaluation['justifications'].get(internal_key, {"text": "Avaliação não fornecida pela API."}).get("text", "Avaliação não fornecida pela API.")
                    doc.append(NoEscape(
                        f'\\noindent\\color{{blue!80!black}}\\textbf{{{escape_latex(display_name)}}} '
                        f'(Nota: \\color{{black}}{score_val}): '
                        f'{escape_latex(justification_text)}\\par\\vspace{{0.3cm}}'
                    ))

            if evaluation['decision'] == 'Reprovada':
                with doc.create(Subsection('Sugestões de Melhoria')):
                    doc.append(NoEscape(r'\vspace{0.3cm}'))
                    for display_name, internal_key in display_to_key_map.items():
                        if evaluation['scores'].get(internal_key, 0) < APPROVAL_THRESHOLD:
                            suggestion_text = evaluation['suggestions'].get(internal_key, "Nenhuma sugestão específica fornecida.")
                            doc.append(NoEscape(
                                f'\\noindent\\color{{blue!80!black}}\\textbf{{{escape_latex(display_name)}}}: '
                                f'\\color{{black}}{escape_latex(suggestion_text)}\\par\\vspace{{0.2cm}}'
                            ))

        print("Salvando arquivo LaTeX...")
        doc.generate_tex()

        print("Compilando PDF...")
        tex_file_name = 'audit_report.tex'
        pdf_file_name = 'audit_report.pdf'

        for ext in ['.aux', '.log', '.out', '.toc', '.lof', '.lot', '.fls', '.fdb_latexmk']:
            temp_file = os.path.join(os.getcwd(), pdf_file_name.replace('.pdf', ext))
            if os.path.exists(temp_file):
                os.remove(temp_file)

        start_time = time.time()
        process = subprocess.run(
            ['latexmk', '-pdf', '-xelatex', '-interaction=nonstopmode', tex_file_name],
            cwd=os.getcwd(),
            timeout=300,
            capture_output=True,
            text=True
        )
        end_time = time.time()
        print(f"Tempo de compilação do PDF: {end_time - start_time:.2f} segundos")

        print("\n--- Saída do Latexmk (stdout) ---")
        print(process.stdout)
        print("--- Fim da Saída do Latexmk (stdout) ---\n")

        print("\n--- Saída do Latexmk (stderr) ---")
        print(process.stderr)
        print("--- Fim da Saída do Latexmk (stderr) ---\n")

        if process.returncode != 0:
            print(f"Erro na compilação do PDF: Código de saída {process.returncode}")
            with open("latex_error.log", "w", encoding="utf-8") as f:
                f.write(f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
            raise Exception(f"Falha na compilação do PDF. Verifique 'latex_error.log' para detalhes. Erro: {process.stderr}")

        return pdf_file_name

    except subprocess.TimeoutExpired:
        print("Erro: Compilação do PDF excedeu o tempo limite.")
        return None
    except Exception as e:
        print(f"Erro na geração do PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def send_to_monday(evaluation, pdf_path=None): # Adicionado pdf_path com valor padrão None
    headers = {
        "Authorization": f"{MONDAY_API_TOKEN}",
        "Content-Type": "application/json"
    }

    item_name = f"Aula - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"
    suggestions_text = "; ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in evaluation["suggestions"].items() if v != "Nenhuma sugestão específica fornecida."])
    
    column_values = {
        "name": item_name,
        "project_status": {"label": evaluation["decision"]},
        "numeric_mkrmq7jr": evaluation["final_score"],
        "numeric_mkrm5m2f": evaluation["scores"].get("qualidade_tecnica", 50),
        "numeric_mkrmwdza": evaluation["scores"].get("linguagem_corporal", 50),
        "numeric_mkrme618": evaluation["scores"].get("tom_de_voz", 50),
        "numeric_mkrm2988": evaluation["scores"].get("clareza_roteiro", 50),
        "numeric_mkrmx949": evaluation["scores"].get("ritmo", 50),
        "numeric_mkrmrtyq": evaluation["scores"].get("didatica", 50),
        "numeric_mkrmvz8t": evaluation["scores"].get("qualidade_geral", 50),
        "long_text_mkrmhbk8": suggestions_text.replace('"', '\\"') # Escape quaisquer aspas no texto da sugestão
    }

    column_values_string_for_graphql = json.dumps(column_values).replace('"', '\\"')

    mutation_query = f"""
        mutation {{
            create_item (
                board_id: {MONDAY_BOARD_ID},
                item_name: "{item_name}",
                column_values: "{column_values_string_for_graphql}"
            ) {{
                id
                name
            }}
        }}
    """
    
    data = {"query": mutation_query}

    print("\n--- Request Headers para Monday.com ---")
    print(headers)
    print("\n--- Column Values (antes de serializar e escapar) ---")
    print(json.dumps(column_values, indent=2))
    print("\n--- Final GraphQL Query para Monday.com ---")
    print(mutation_query)
    print("------------------------------------------")

    item_id = None # Inicializa item_id como None
    try:
        print(f"Enviando dados para o Monday.com para a aula '{item_name}'...")
        response = requests.post(MONDAY_API_URL, headers=headers, json=data)
        response.raise_for_status() 
        
        response_data = response.json()
        if "data" in response_data and response_data["data"].get("create_item"):
            item_id = response_data["data"]["create_item"]["id"]
            print(f"Item criado com sucesso no Monday.com! ID: {item_id}")

            # --- Lógica de upload de PDF ---
            if pdf_path and os.path.exists(pdf_path):
                print(f"Anexando PDF '{os.path.basename(pdf_path)}' ao item {item_id}...")
                file_upload_url = "https://api.monday.com/v2/file"

                mutation_upload_file = f"""
                mutation addFileToColumn($file: File!) {{
                  add_file_to_column (
                    item_id: {item_id},
                    column_id: "{MONDAY_FILES_COLUMN_ID}",
                    file: $file
                  ) {{
                    id
                  }}
                }}
                """
                
                file_map_json = json.dumps({"file": ["variables.file"]}) 
                
                files = {
                    'query': (None, mutation_upload_file, 'application/json'), 
                    'map': (None, file_map_json, 'application/json'),          
                    'file': (os.path.basename(pdf_path), open(pdf_path, 'rb'), 'application/pdf') 
                }

                file_upload_headers = {
                    "Authorization": f"{MONDAY_API_TOKEN}",
                }

                file_response = requests.post(file_upload_url, headers=file_upload_headers, files=files)
                file_response.raise_for_status() 
                
                file_response_data = file_response.json()
                if file_response_data.get('data') and file_response_data['data'].get('add_file_to_column'):
                    file_id = file_response_data['data']['add_file_to_column']['id']
                    print(f"PDF anexado à coluna de arquivos do item {item_id} com File ID: {file_id}")
                else:
                    print(f"Erro ao anexar o PDF: {json.dumps(file_response_data, indent=2)}")
            else:
                print("Caminho do PDF não fornecido ou arquivo PDF não encontrado. Não foi possível anexar ao Monday.com.")

            return item_id 
    
        
        else:
            print(f"Erro ao criar item no Monday.com (resposta inesperada): {json.dumps(response_data, indent=2)}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Erro de conexão ou API ao Monday.com: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Resposta da API (erro): {e.response.text}")
        return None
    except Exception as e:
        print(f"Erro inesperado ao enviar para o Monday.com: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    file.save(video_path)
    
    transcription = transcribe_audio(video_path)
    if not transcription:
        return jsonify({'error': 'Falha na transcrição'}), 500
    
    movement_score = analyze_body_language(video_path)
    evaluation = evaluate_lesson(transcription, movement_score)
    pdf_path = generate_pdf_report(evaluation) # Gera o PDF primeiro
    
    # Enviar dados para o monday.com, AGORA PASSANDO O pdf_path
    send_to_monday(evaluation, pdf_path) 
    
    if pdf_path and os.path.exists(pdf_path):
        response = make_response(send_file(pdf_path, as_attachment=True, mimetype='application/pdf'))
        response.headers['Content-Disposition'] = f'attachment; filename={os.path.basename(pdf_path)}'
        return response
    else:
        return jsonify({'message': 'Avaliação concluída, mas PDF não gerado/encontrado.'}), 200

if __name__ == '__main__':
    # Cria a pasta de uploads se não existir
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)