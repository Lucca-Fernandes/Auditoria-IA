import cv2
import os
import whisper
import librosa
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
from sentence_transformers import SentenceTransformer
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
    return [energy_std]

# Função para extrair características de texto
def extract_text_features(transcription):
    doc = nlp(transcription)
    sentences = list(doc.sents)
    avg_sent_length = np.mean([len(sent) for sent in sentences]) if sentences else 0
    insecurity_words = ["erro", "ops", "desculpe", "não sei"]
    insecurity_count = sum(1 for token in doc if token.text.lower() in insecurity_words)
    total_words = len(doc)
    insecurity_ratio = insecurity_count / total_words if total_words > 0 else 0
    return [avg_sent_length, insecurity_ratio]

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

# Função para treinar o modelo com os dados reais
def train_model():
    X = [
        [0.039369076, 94.85714285714286, 0.0007530120481927711, 0.10119047619047619],  # aula_1 (Ruim)
        [0.04511572, 12.452830188679245, 0.0015151515151515152, 0.07076923076923076],  # aula_2 (Médio)
        [0.030089283, 42.30232558139535, 0.0, 0.10897435897435898],                    # aula_3 (Bom)
        [0.018021856, 86.0, 0.0, 0.06426332288401254]                                 # aula_4 (Bom)
    ]
    y = [0, 1, 2, 2]  # 0: Ruim, 1: Médio, 2: Bom
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("Atenção: Conjunto de dados muito pequeno. Modelo treinado com todos os dados.")
    return model

# Função para avaliar uma aula
def evaluate_lesson(video_path, model):
    audio_path = extract_audio(video_path)
    transcription = transcribe_audio(audio_path)
    audio_features = extract_audio_features(audio_path)
    text_features = extract_text_features(transcription)
    video_features = extract_video_features(video_path)
    
    features = audio_features + text_features + video_features
    print(f"Características da nova aula: {features}")
    
    prediction = model.predict([features])
    labels = {0: "Ruim", 1: "Médio", 2: "Bom"}
    result = labels[prediction[0]]
    score = model.predict_proba([features])[0][prediction[0]] * 100
    
    # Justificativas e sugestões baseadas nas características
    energy_std, avg_sent_length, insecurity_ratio, movement_ratio = features
    justification = []
    suggestions = []
    
    if result == "Ruim":
        if insecurity_ratio > 0.0005:
            justification.append(f"Insegurança alta ({insecurity_ratio:.6f}) detectada.")
            suggestions.append("Reduza erros ou hesitações no discurso.")
        if avg_sent_length > 50:
            justification.append(f"Sentenças longas ({avg_sent_length:.2f}) podem confundir.")
            suggestions.append("Divida as sentenças em partes menores.")
        if energy_std < 0.03:
            justification.append(f"Tom de voz pouco variado ({energy_std:.6f}).")
            suggestions.append("Aumente a variação no tom para engajar mais.")
        if movement_ratio < 0.08:
            justification.append(f"Movimento baixo ({movement_ratio:.3f}) na linguagem corporal.")
            suggestions.append("Adicione gestos ou movimentação.")
    elif result == "Médio":
        if insecurity_ratio > 0.001:
            justification.append(f"Insegurança moderada ({insecurity_ratio:.6f}) detectada.")
            suggestions.append("Minimize hesitações para maior confiança.")
        if avg_sent_length > 20:
            justification.append(f"Sentenças longas ({avg_sent_length:.2f}) podem ser melhoradas.")
            suggestions.append("Simplifique a estrutura do roteiro.")
        if energy_std < 0.04:
            justification.append(f"Tom de voz com pouca variação ({energy_std:.6f}).")
            suggestions.append("Varie mais o tom para dinamismo.")
    else:  # Bom
        justification.append("Conteúdo bem estruturado e apresentado.")
        suggestions.append("Mantenha o padrão de qualidade.")

    justification = "; ".join(justification) if justification else "Nenhuma falha técnica significativa."
    suggestion = "; ".join(suggestions) if suggestions else "Nenhuma sugestão de melhoria necessária."

    return {
        "Qualidade Geral": f"{result} (score: {score:.1f}/100)",
        "Justificativa Técnica": justification,
        "Sugestão Geral": suggestion
    }

# Função para gerar relatório em PDF estilizado
def generate_pdf_report(results, lesson_type, theme, output_path="relatorio_aula_nova.pdf"):
    # Criar documento PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    
    # Estilos
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
        textColor=colors.green if "Bom" in results["Análise Geral"] else colors.orange if "Médio" in results["Análise Geral"] else colors.red
    )
    
    # Cabeçalho
    elements.append(Paragraph("Relatório de Auditoria de Aula", title_style))
    elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Informações da aula
    elements.append(Paragraph("Informações Gerais", heading_style))
    elements.append(Paragraph(f"Tipo de Aula: {lesson_type.capitalize()}", normal_style))
    elements.append(Paragraph(f"Tema da Aula: {theme.capitalize()}", normal_style))
    elements.append(Spacer(1, 12))
    
    # Resultados da análise
    elements.append(Paragraph("Resultados da Análise", heading_style))
    elements.append(Paragraph(f"Análise Geral: {results['Análise Geral']}", quality_style))
    elements.append(Paragraph(f"Justificativa Técnica: {results['Justificativa Técnica']}", normal_style))
    elements.append(Paragraph(f"Sugestões de Melhoria: {results['Sugestão']}", normal_style))
    
    # Construir o PDF
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
        
        # Treinar o modelo com os dados reais imediatamente
        model = train_model()
        if not model:
            print("Não foi possível treinar o modelo com os dados fornecidos.")
            return
        
        print("Modelo treinado com sucesso! Agora você pode avaliar novas aulas.")
        
        # Opção para avaliar uma nova aula
        while True:
            choice = input("Deseja avaliar uma nova aula? (s/n): ").lower()
            if choice == 's':
                video_path = input("Digite o caminho da nova aula (ex.: C:\\Users\\MP10\\Desktop\\aula_nova.mp4): ")
                video_path = video_path.replace("\\", "\\\\")
                audio_path = extract_audio(video_path)
                print("Áudio extraído com sucesso.")

                transcription = transcribe_audio(audio_path)
                print("Transcrição concluída:", transcription[:100], "...")

                lesson_type, theme = infer_lesson_type_and_theme(transcription)
                print(f"Tipo de aula inferido: {lesson_type}")
                print(f"Tema da aula inferido: {theme}")

                overall_analysis = evaluate_lesson(video_path, model)
                print("Avaliação geral concluída:", overall_analysis)

                results = {
                    "Análise Geral": overall_analysis["Qualidade Geral"],
                    "Justificativa Técnica": overall_analysis["Justificativa Técnica"],
                    "Sugestão": overall_analysis["Sugestão Geral"]
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