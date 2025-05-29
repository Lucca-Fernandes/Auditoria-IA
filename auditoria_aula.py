import cv2
import os
import whisper
import librosa
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime

# Carregar modelos
nlp = spacy.load("pt_core_news_sm")
sentence_model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Função para limpar arquivos temporários
def clean_temp_files():
    temp_files = ["temp_audio.wav"]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

# Função para extrair áudio do vídeo
def extract_audio(video_path, audio_path="temp_audio.wav"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    if os.path.exists(audio_path):
        os.remove(audio_path)
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}")
    return audio_path

# Função para transcrever áudio
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="pt")
    return result["text"]

# Função para extrair características de áudio
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=y)[0]
    energy_std = np.std(energy)
    silence = librosa.effects.split(y, top_db=20)
    total_duration = len(y) / sr
    pause_duration = sum((end - start) / sr for start, end in silence if (end - start) / sr > 1.0)
    pause_ratio = pause_duration / total_duration if total_duration > 0 else 0
    return [energy_std, pause_ratio]

# Função para extrair características de texto
def extract_text_features(transcription, theme):
    doc = nlp(transcription)
    sentences = list(doc.sents)
    avg_sent_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
    
    # Insegurança
    insecurity_words = ["erro", "ops", "desculpe", "não sei", "deixa eu corrigir", "não está funcionando", "vou tentar de novo", "deu erro", "não compila"]
    insecurity_count = sum(1 for token in doc if token.text.lower() in insecurity_words)
    total_words = len(doc)
    insecurity_ratio = insecurity_count / total_words if total_words > 0 else 0
    
    # Conhecimento técnico
    technical_words = ["programação", "variável", "função", "algoritmo", "código", "projeto", "dados", "sistema", "loop", "condicional", "debug", "classe", "método"]
    technical_count = sum(1 for token in doc if token.text.lower() in technical_words)
    technical_density = technical_count / total_words if total_words > 0 else 0
    
    # Contexto técnico negativo (ajustado por tema)
    negative_technical_phrases = ["deu erro", "não funciona", "está errado", "vou refazer"]
    negative_technical_count = sum(1 for sent in sentences if any(phrase in sent.text.lower() for phrase in negative_technical_phrases))
    negative_technical_ratio = negative_technical_count / len(sentences) if sentences else 0
    if theme == "tratamento de erros":
        negative_technical_ratio *= 0.1  # Reduzir impacto em aulas de tratamento de erros
    
    # Didática
    didactic_phrases = ["por exemplo", "em resumo", "passo a passo", "vamos aprender", "como funciona", "vou explicar"]
    didactic_count = sum(1 for sent in sentences if any(phrase in sent.text.lower() for phrase in didactic_phrases))
    didactic_ratio = didactic_count / len(sentences) if sentences else 0
    
    # Interatividade
    interactivity_phrases = ["alguma dúvida", "o que acham", "vamos juntos", "pergunta", "entenderam"]
    interactivity_count = sum(1 for sent in sentences if "?" in sent.text or any(phrase in sent.text.lower() for phrase in interactivity_phrases))
    interactivity_ratio = interactivity_count / len(sentences) if sentences else 0
    
    # Coerência
    if len(sentences) > 1:
        embeddings = sentence_model.encode([sent.text for sent in sentences], convert_to_tensor=True)
        similarities = [util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item() for i in range(len(embeddings) - 1)]
        coherence_score = np.mean(similarities) if similarities else 0
    else:
        coherence_score = 0
    
    # Repetição no discurso
    words = [token.text.lower() for token in doc]
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    repetition_score = sum(count for count in word_counts.values() if count > 3) / total_words if total_words > 0 else 0
    
    # Desânimo
    discouragement_words = ["difícil", "complicado", "não sei explicar", "estou perdido", "não entendi"]
    discouragement_count = sum(1 for token in doc if token.text.lower() in discouragement_words)
    discouragement_ratio = discouragement_count / total_words if total_words > 0 else 0
    
    # Sentimento geral
    sentiment_scores = sentiment_analyzer(transcription[:512])  # Limitar a 512 tokens
    avg_sentiment = float(sentiment_scores[0]['score']) * 100 if sentiment_scores else 50.0
    
    return [avg_sent_length, insecurity_ratio, technical_density, negative_technical_ratio, didactic_ratio, interactivity_ratio, coherence_score, repetition_score, discouragement_ratio, avg_sentiment]

# Função para extrair características de vídeo
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Erro ao abrir o vídeo.")
    
    sampled_frames = 0
    movement_frames = 0
    prev_frame = None
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = max(1, int(fps))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            sampled_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                movement = np.sum(diff) / (diff.shape[0] * diff.shape[1])
                if movement > 15:
                    movement_frames += 1
            prev_frame = gray.copy()
        frame_count += 1
    
    cap.release()
    movement_ratio = movement_frames / sampled_frames if sampled_frames > 0 else 0
    return [movement_ratio]

# Função para treinar o modelo
def train_model():
    X = [
        [94.85714285714286, 0.0007530120481927711, 0.01, 0.02, 0.001, 0.002, 0.4, 0.05, 0.01, 50.0, 0.039369076, 0.1, 0.10119047619047619],  # aula_1 (Ruim)
        [12.452830188679245, 0.0015151515151515152, 0.02, 0.01, 0.002, 0.003, 0.5, 0.03, 0.005, 60.0, 0.04511572, 0.05, 0.07076923076923076],  # aula_2 (Médio)
        [42.30232558139535, 0.0, 0.03, 0.005, 0.003, 0.004, 0.6, 0.02, 0.0, 70.0, 0.030089283, 0.02, 0.10897435897435898],                   # aula_3 (Bom)
        [86.0, 0.0, 0.04, 0.0, 0.004, 0.005, 0.7, 0.01, 0.0, 75.0, 0.018021856, 0.01, 0.06426332288401254],                               # aula_4 (Bom)
        [60.0, 0.0005, 0.015, 0.01, 0.002, 0.003, 0.55, 0.04, 0.008, 65.0, 0.035, 0.08, 0.09],  # aula_5 (Médio)
        [30.0, 0.0, 0.025, 0.0, 0.003, 0.004, 0.65, 0.02, 0.0, 80.0, 0.025, 0.03, 0.12],         # aula_6 (Bom)
        [100.0, 0.001, 0.005, 0.03, 0.001, 0.002, 0.45, 0.06, 0.015, 45.0, 0.042, 0.12, 0.07],  # aula_7 (Ruim)
    ]
    y = [0, 1, 2, 2, 1, 2, 0]  # 0: Ruim, 1: Médio, 2: Bom
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    print("Modelo treinado com sucesso! Conjunto de dados expandido.")
    return model

# Função para avaliar uma aula
def evaluate_lesson(video_path, model, theme):
    audio_path = extract_audio(video_path)
    transcription = transcribe_audio(audio_path)
    audio_features = extract_audio_features(audio_path)
    text_features = extract_text_features(transcription, theme)
    video_features = extract_video_features(video_path)
    
    features = text_features + audio_features + video_features
    print(f"Características da nova aula: {features}")
    
    prediction = model.predict([features])
    labels = {0: "Ruim", 1: "Médio", 2: "Bom"}
    result = labels[prediction[0]]
    
    # Calcular o score
    try:
        probs = model.predict_proba([features])[0]
        if len(probs) != 3:
            print(f"Aviso: predict_proba retornou {len(probs)} classes ao invés de 3. Ajustando...")
            probs = [probs[i] if i < len(probs) else 0.0 for i in range(3)]
        score = probs[prediction[0]] * 100
        score = max(0, min(100, score))
    except Exception as e:
        print(f"Erro ao calcular o score: {str(e)}. Calculando média dos critérios.")
        criteria_scores = {}
        avg_sent_length, insecurity_ratio, technical_density, negative_technical_ratio, didactic_ratio, interactivity_ratio, coherence_score, repetition_score, discouragement_ratio, avg_sentiment, energy_std, pause_ratio, movement_ratio = features
        criteria_scores["Tom de Voz"] = max(0, min(100, 50 + (energy_std - 0.03) * 3000)) if energy_std >= 0.03 else 40
        criteria_scores["Linguagem Corporal"] = max(0, min(100, 50 + (movement_ratio - 0.05) * 500)) if movement_ratio >= 0.05 else 40
        criteria_scores["Clareza e Estrutura"] = max(0, min(100, 100 - (avg_sent_length - 30) * 1.5)) if avg_sent_length <= 70 else 0
        criteria_scores["Ritmo"] = max(0, min(100, 100 - (pause_ratio - 0.1) * 250)) if pause_ratio <= 0.3 else 0
        criteria_scores["Qualidade Técnica"] = max(0, min(100, (technical_density * 8000) - (negative_technical_ratio * 5000))) if technical_density >= 0.005 else 0
        criteria_scores["Didática"] = max(0, min(100, didactic_ratio * 40000)) if didactic_ratio >= 0.001 else 0
        criteria_scores["Interatividade"] = max(0, min(100, interactivity_ratio * 20000)) if interactivity_ratio >= 0.001 else 0
        criteria_scores["Coerência"] = max(0, min(100, coherence_score * 150)) if coherence_score >= 0.5 else 0
        criteria_scores["Estabilidade Emocional"] = max(0, min(100, avg_sentiment - (discouragement_ratio * 10000))) if discouragement_ratio <= 0.01 else 0
        score = sum(criteria_scores.values()) / len(criteria_scores)

    # Justificativas e sugestões
    avg_sent_length, insecurity_ratio, technical_density, negative_technical_ratio, didactic_ratio, interactivity_ratio, coherence_score, repetition_score, discouragement_ratio, avg_sentiment, energy_std, pause_ratio, movement_ratio = features
    justification = []
    suggestions = []
    criteria_scores = {}
    
    # Avaliação por critério
    criteria_scores["Tom de Voz"] = max(0, min(100, 50 + (energy_std - 0.03) * 3000)) if energy_std >= 0.03 else 40
    criteria_scores["Linguagem Corporal"] = max(0, min(100, 50 + (movement_ratio - 0.05) * 500)) if movement_ratio >= 0.05 else 40
    criteria_scores["Clareza e Estrutura"] = max(0, min(100, 100 - (avg_sent_length - 30) * 1.5)) if avg_sent_length <= 70 else 0
    criteria_scores["Ritmo"] = max(0, min(100, 100 - (pause_ratio - 0.1) * 250)) if pause_ratio <= 0.3 else 0
    criteria_scores["Qualidade Técnica"] = max(0, min(100, (technical_density * 8000) - (negative_technical_ratio * 5000))) if technical_density >= 0.005 else 0
    criteria_scores["Didática"] = max(0, min(100, didactic_ratio * 40000)) if didactic_ratio >= 0.001 else 0
    criteria_scores["Interatividade"] = max(0, min(100, interactivity_ratio * 20000)) if interactivity_ratio >= 0.001 else 0
    criteria_scores["Coerência"] = max(0, min(100, coherence_score * 150)) if coherence_score >= 0.5 else 0
    criteria_scores["Estabilidade Emocional"] = max(0, min(100, avg_sentiment - (discouragement_ratio * 10000))) if discouragement_ratio <= 0.01 else 0

    if result == "Ruim":
        if insecurity_ratio > 0.0005:
            justification.append(f"Insegurança alta ({insecurity_ratio:.6f}) detectada.")
            suggestions.append("Reduza erros ou hesitações no discurso.")
        if avg_sent_length > 50:
            justification.append(f"Sentenças longas ({avg_sent_length:.2f}) podem confundir.")
            suggestions.append("Divida as sentenças em partes menores.")
        if energy_std < 0.03:
            justification.append(f"Tom de voz monótono ({energy_std:.6f}).")
            suggestions.append("Aumente a variação no tom para engajar mais.")
        if movement_ratio < 0.08:
            justification.append(f"Movimento baixo ({movement_ratio:.3f}) na linguagem corporal.")
            suggestions.append("Adicione gestos ou movimentação.")
        if pause_ratio > 0.3:
            justification.append(f"Ritmo irregular com muitas pausas ({pause_ratio:.2f}).")
            suggestions.append("Reduza pausas longas para melhorar a fluidez.")
        if technical_density < 0.005 or negative_technical_ratio > 0.01:
            justification.append(f"Conhecimento técnico fraco (densidade: {technical_density:.3f}, erros: {negative_technical_ratio:.3f}).")
            suggestions.append("Revise o conteúdo técnico e pratique a execução do código.")
        if didactic_ratio < 0.001:
            justification.append(f"Didática limitada ({didactic_ratio:.6f}).")
            suggestions.append("Use mais exemplos ou explicações estruturadas.")
        if interactivity_ratio < 0.001:
            justification.append(f"Baixa interatividade ({interactivity_ratio:.6f}).")
            suggestions.append("Inclua perguntas ou convites à participação.")
        if coherence_score < 0.5:
            justification.append(f"Coerência baixa ({coherence_score:.2f}).")
            suggestions.append("Melhore a conexão entre ideias e sentenças.")
        if repetition_score > 0.05:
            justification.append(f"Alta repetição no discurso ({repetition_score:.3f}).")
            suggestions.append("Evite repetir palavras ou ideias desnecessariamente.")
        if discouragement_ratio > 0.005:
            justification.append(f"Desânimo detectado ({discouragement_ratio:.6f}).")
            suggestions.append("Tente manter um tom mais positivo e confiante.")
    elif result == "Médio":
        if insecurity_ratio > 0.001:
            justification.append(f"Insegurança moderada ({insecurity_ratio:.6f}) detectada.")
            suggestions.append("Minimize hesitações para maior confiança.")
        if avg_sent_length > 30:
            justification.append(f"Sentenças longas ({avg_sent_length:.2f}) podem ser melhoradas.")
            suggestions.append("Simplifique a estrutura do roteiro.")
        if energy_std < 0.04:
            justification.append(f"Tom de voz com pouca variação ({energy_std:.6f}).")
            suggestions.append("Varie mais o tom para dinamismo.")
        if pause_ratio > 0.2:
            justification.append(f"Ritmo pode melhorar ({pause_ratio:.2f}).")
            suggestions.append("Ajuste o ritmo para evitar pausas longas.")
        if technical_density < 0.01 or negative_technical_ratio > 0.005:
            justification.append(f"Conhecimento técnico moderado (densidade: {technical_density:.3f}, erros: {negative_technical_ratio:.3f}).")
            suggestions.append("Aprofunde o conteúdo técnico e evite erros no código.")
        if didactic_ratio < 0.002:
            justification.append(f"Didática pode melhorar ({didactic_ratio:.6f}).")
            suggestions.append("Inclua mais exemplos práticos.")
        if interactivity_ratio < 0.002:
            justification.append(f"Interatividade moderada ({interactivity_ratio:.6f}).")
            suggestions.append("Adicione mais perguntas ou interações.")
        if coherence_score < 0.6:
            justification.append(f"Coerência moderada ({coherence_score:.2f}).")
            suggestions.append("Conecte melhor as ideias apresentadas.")
        if repetition_score > 0.03:
            justification.append(f"Repetição moderada no discurso ({repetition_score:.3f}).")
            suggestions.append("Tente variar mais o vocabulário.")
        if discouragement_ratio > 0.003:
            justification.append(f"Leve desânimo detectado ({discouragement_ratio:.6f}).")
            suggestions.append("Mantenha um tom mais confiante durante a aula.")
    else:  # Bom
        justification.append("Conteúdo bem estruturado e apresentado com confiança.")
        suggestions.append("Mantenha o padrão de qualidade.")

    justification = "; ".join(justification) if justification else "Nenhuma falha técnica significativa."
    suggestion = "; ".join(suggestions) if suggestions else "Nenhuma sugestão de melhoria necessária."

    return {
        "Qualidade Geral": f"{result} (score: {score:.1f}/100)",
        "Justificativa Técnica": justification,
        "Sugestão Geral": suggestion,
        "Criteria Scores": criteria_scores
    }

# Função para gerar relatório em PDF estilizado
def generate_pdf_report(results, lesson_type, theme, output_path="relatorio_aula_nova.pdf"):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.navy
    )
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6,
        textColor=colors.black
    )
    normal_style = ParagraphStyle(
        name='NormalStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        leading=14
    )
    quality_style = ParagraphStyle(
        name='QualityStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        leading=14,
        textColor=colors.green if "Bom" in results["Qualidade Geral"] else colors.orange if "Médio" in results["Qualidade Geral"] else colors.red
    )
    summary_style = ParagraphStyle(
        name='SummaryStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        leading=14,
        textColor=colors.darkblue
    )
    
    elements.append(Paragraph("Relatório de Auditoria de Aula", title_style))
    elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Informações Gerais", heading_style))
    elements.append(Paragraph(f"Tipo de Aula: {lesson_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Tema da Aula: {theme.capitalize()}", normal_style))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Resultados da Análise", heading_style))
    elements.append(Paragraph(f"Análise Geral: {results['Qualidade Geral']}", quality_style))
    elements.append(Paragraph(f"Justificativa Técnica: {results['Justificativa Técnica']}", normal_style))
    elements.append(Paragraph(f"Sugestões de Melhoria: {results['Sugestão Geral']}", normal_style))
    
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Avaliação por Critério", heading_style))
    for criterion, score in results["Criteria Scores"].items():
        color = colors.green if score >= 70 else colors.orange if score >= 50 else colors.red
        score_style = ParagraphStyle(
            name=f'{criterion}Style',
            parent=normal_style,
            textColor=color
        )
        elements.append(Paragraph(f"{criterion}: {score:.1f}/100", score_style))
    
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Resumo da Aula", heading_style))
    summary = ""
    if "Ruim" in results["Qualidade Geral"]:
        summary += "A aula foi reprovada devido a problemas na entrega e conteúdo. "
    elif "Médio" in results["Qualidade Geral"]:
        summary += "A aula teve desempenho médio, com pontos a melhorar. "
    else:
        summary += "A aula foi aprovada, com boa estrutura e apresentação. "
    if results["Criteria Scores"]["Tom de Voz"] < 50:
        summary += "Tom de voz monótono é um ponto negativo. "
    if results["Criteria Scores"]["Linguagem Corporal"] < 50:
        summary += "Falta de movimento na linguagem corporal prejudicou a entrega. "
    if results["Criteria Scores"]["Qualidade Técnica"] >= 70:
        summary += "Conhecimento técnico sólido é um ponto positivo. "
    elif results["Criteria Scores"]["Qualidade Técnica"] < 50:
        summary += "Falta de domínio técnico foi um ponto fraco. "
    if results["Criteria Scores"]["Didática"] >= 70:
        summary += "Explicações claras destacaram a didática. "
    elif results["Criteria Scores"]["Didática"] < 50:
        summary += "Didática limitada dificultou o entendimento. "
    if results["Criteria Scores"]["Interatividade"] >= 70:
        summary += "Boa interatividade engajou os alunos. "
    elif results["Criteria Scores"]["Interatividade"] < 50:
        summary += "Interatividade baixa foi um ponto fraco. "
    if results["Criteria Scores"]["Estabilidade Emocional"] < 50:
        summary += "Desânimo ou insegurança foram evidentes. "
    elements.append(Paragraph(summary, summary_style))
    
    doc.build(elements)
    return output_path

# Função para inferir tipo e tema
def infer_lesson_type_and_theme(transcription):
    doc = nlp(transcription.lower())
    project_keywords = ["projeto", "passo a passo", "implementar"]
    practical_keywords = ["exemplo", "prática", "exercício"]
    error_keywords = ["erro", "exceção", "try", "except"]
    
    project_score = sum(1 for token in doc if token.text in project_keywords)
    practical_score = sum(1 for token in doc if token.text in practical_keywords)
    error_score = sum(1 for token in doc if token.text in error_keywords)
    
    if project_score > 3:
        lesson_type = "orientada a projetos"
    elif practical_score > 3:
        lesson_type = "prática"
    else:
        lesson_type = "teórica"
    
    if error_score > 3:
        theme = "tratamento de erros"
    else:
        theme = "geral"
    
    return lesson_type, theme

# Função principal
def main():
    try:
        clean_temp_files()
        
        model = train_model()
        if not model:
            print("Não foi possível treinar o modelo com os dados fornecidos.")
            return
        
        print("Modelo treinado com sucesso! Agora você pode avaliar novas aulas.")
        
        while True:
            choice = input("Deseja avaliar uma nova aula? (s/n): ").lower()
            if choice == 's':
                video_path = input("Digite o caminho da nova aula (ex.: C:\\Users\\Bolado\\Desktop\\aula_nova.mp4): ")
                video_path = video_path.replace("\\", "\\\\")
                audio_path = extract_audio(video_path)
                print("Áudio extraído com sucesso.")

                transcription = transcribe_audio(audio_path)
                print("Transcrição concluída:", transcription[:100], "...")

                lesson_type, theme = infer_lesson_type_and_theme(transcription)
                print(f"Tipo de aula inferido: {lesson_type}")
                print(f"Tema da aula inferido: {theme}")

                overall_analysis = evaluate_lesson(video_path, model, theme)  # Passar theme aqui
                print("Avaliação geral concluída:", overall_analysis)

                results = {
                    "Qualidade Geral": overall_analysis["Qualidade Geral"],
                    "Justificativa Técnica": overall_analysis["Justificativa Técnica"],
                    "Sugestão Geral": overall_analysis["Sugestão Geral"],
                    "Criteria Scores": overall_analysis["Criteria Scores"]
                }
                report_path = generate_pdf_report(results, lesson_type, theme)
                print(f"Relatório gerado: {report_path}")
            elif choice == 'n':
                print("Encerrando o programa.")
                break
            else:
                print("Opção inválida. Digite 's' para sim ou 'n' para não.")

    except Exception as e:
        print(f"Erro: {str(e)}")
    finally:
        clean_temp_files()

if __name__ == "__main__":
    main()