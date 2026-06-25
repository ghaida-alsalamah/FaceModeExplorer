import os
import cv2
import torch
import gradio as gr
import pandas as pd
from collections import Counter
from insightface.app import FaceAnalysis
from hsemotion.facial_emotions import HSEmotionRecognizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

face_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "genderage"]
)

ctx_id = 0 if DEVICE == "cuda" else -1
face_app.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=(640, 640))

# Patch torch.load for HSEmotion compatibility with PyTorch 2.6+
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

emotion_recognizer = HSEmotionRecognizer(
    model_name="enet_b0_8_best_vgaf",
    device=DEVICE
)


def detect_faces(frame):
    return face_app.get(frame)


def predict_age_gender(face):
    age = int(face.age)

    raw_gender = getattr(face, "sex", getattr(face, "gender", None))
    val = str(raw_gender).strip().upper()

    if val in ["1", "M", "MALE"]:
        gender = "Male"
    elif val in ["0", "F", "FEMALE"]:
        gender = "Female"
    else:
        gender = "Unknown"

    return age, gender


def predict_emotion(frame, face):
    x1, y1, x2, y2 = face.bbox.astype(int)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return "Unknown"

    emotion, _ = emotion_recognizer.predict_emotions(face_crop)
    return emotion


def draw_results(frame, faces, age_gender_results, emotion_results):
    PURPLE = (128, 0, 128)

    for face, (age, gender), emotion in zip(faces, age_gender_results, emotion_results):
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), PURPLE, 2)

        label = f"Age: {age} | {gender} | {emotion}"

        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1
        )

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 28)),
            (x1 + text_width + 10, y1),
            PURPLE,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


def analyze_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)

    ages = []
    genders = []
    emotions = []

    last_faces = []
    last_age_gender = []
    last_emotions = []

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = f"/tmp/annotated_{uuid.uuid4().hex}.mp4"

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip == 0:
            faces = detect_faces(frame)

            current_age_gender = []
            current_emotions = []

            for face in faces:
                age, gender = predict_age_gender(face)
                emotion = predict_emotion(frame, face)

                current_age_gender.append((age, gender))
                current_emotions.append(emotion)

                ages.append(age)
                genders.append(gender)
                emotions.append(emotion)

            last_faces = faces
            last_age_gender = current_age_gender
            last_emotions = current_emotions

        frame = draw_results(frame, last_faces, last_age_gender, last_emotions)
        writer.write(frame)

    cap.release()
    writer.release()

    return output_path, ages, genders, emotions


def show_loading():
    loading_html = """
    <div class="result-card loading-card">
        <h2>Working on your video...</h2>
        <p>The analysis is running now. You can stay on this page while we prepare the results.</p>
        <div class="loader"></div>
    </div>
    """
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        loading_html,
        None,
        None,
        pd.DataFrame(),
        gr.update(value=None, visible=False)
    )


def gradio_analyze(video_path):
    output_path, ages, genders, emotions = analyze_video(video_path)

    if not ages:
        summary = """
        <div class="result-card">
            <h2>No clear faces found</h2>
            <p>Try another video with better lighting or closer faces.</p>
        </div>
        """
        return output_path, summary, None, pd.DataFrame(), gr.update(value=None, visible=False)

    ages = [int(a) for a in ages]
    avg_age = int(round(sum(ages) / len(ages)))

    gender_counts = Counter(genders)
    emotion_counts = Counter(emotions)

    final_gender = gender_counts.most_common(1)[0][0]
    final_emotion = emotion_counts.most_common(1)[0][0]
    total_predictions = len(ages)

    df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "Emotion": emotions
    })

    csv_path = f"/tmp/face_analysis_results_{uuid.uuid4().hex}.csv"
    df.to_csv(csv_path, index=False)

    summary = f"""
    <div class="result-card">
        <h2>Done, your video is ready</h2>
        <p>Here is a clean summary from the analyzed frames.</p>
        <div class="stats-grid">
            <div class="stat-box">
                <span>Average Age</span>
                <strong>{avg_age}</strong>
            </div>
            <div class="stat-box">
                <span>Most Seen Gender</span>
                <strong>{final_gender}</strong>
            </div>
            <div class="stat-box">
                <span>Main Expression</span>
                <strong>{final_emotion}</strong>
            </div>
            <div class="stat-box">
                <span>Face Detections</span>
                <strong>{total_predictions}</strong>
            </div>
        </div>
    </div>
    """

    gender_df = pd.DataFrame({
        "Gender": list(gender_counts.keys()),
        "Count": list(gender_counts.values())
    })

    emotion_df = pd.DataFrame({
        "Emotion": list(emotion_counts.keys()),
        "Count": list(emotion_counts.values())
    }).sort_values(by="Count", ascending=False)

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Gender Mix", "Expressions", "Age Spread")
    )

    fig.add_trace(
        go.Pie(
            labels=gender_df["Gender"],
            values=gender_df["Count"],
            hole=0.55,
            marker=dict(colors=["#6D4C9F", "#B48AD8", "#8E6BBE"]),
            textinfo="label+percent"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Bar(
            x=emotion_df["Emotion"],
            y=emotion_df["Count"],
            marker_color="#8E6BBE",
            text=emotion_df["Count"],
            textposition="auto"
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Histogram(
            x=df["Age"],
            nbinsx=10,
            marker_color="#6D4C9F"
        ),
        row=1,
        col=3
    )

    fig.update_layout(
        height=380,
        paper_bgcolor="#11101A",
        plot_bgcolor="#11101A",
        font=dict(color="#F5F1FF"),
        title=dict(
            text="Video Insights",
            x=0.5,
            font=dict(size=21, color="#E6D7FF")
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=25)
    )

    fig.update_xaxes(color="#F5F1FF", gridcolor="#34264A")
    fig.update_yaxes(color="#F5F1FF", gridcolor="#34264A")

    return output_path, summary, fig, df, gr.update(value=csv_path, visible=True)


def reset_app():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        None,
        None,
        pd.DataFrame(),
        gr.update(value=None, visible=False)
    )


css = """
.gradio-container {
    background: linear-gradient(135deg, #090713, #15111F, #241633) !important;
    color: #F5F1FF !important;
}
#hero {
    text-align: center;
    padding: 22px 16px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(190, 160, 225, 0.15);
    margin-bottom: 16px;
}
#hero h1 {
    font-size: 38px;
    margin: 0 0 6px 0;
    color: #D9C2FF;
}
#hero p {
    font-size: 15px;
    color: #CFC3E8;
}
.upload-card, .result-card {
    padding: 16px;
    border-radius: 16px;
    background: rgba(20, 15, 32, 0.88);
    border: 1px solid rgba(190, 160, 225, 0.16);
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 14px;
}
.stat-box {
    background: linear-gradient(135deg, #20172E, #2F2142);
    padding: 13px;
    border-radius: 14px;
    border: 1px solid rgba(190, 160, 225, 0.14);
}
.stat-box span {
    display: block;
    color: #CDB7EC;
    font-size: 12px;
}
.stat-box strong {
    display: block;
    margin-top: 6px;
    font-size: 22px;
    color: #FFFFFF;
}
button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    background: linear-gradient(90deg, #6D4C9F, #9B6BCB) !important;
    border: none !important;
}
.loader {
    margin-top: 18px;
    width: 100%;
    height: 10px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6D4C9F, #B48AD8, #6D4C9F);
    background-size: 200% 100%;
    animation: loadingMove 1.2s linear infinite;
}
@keyframes loadingMove {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
"""


with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="purple")) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>Face Mood Explorer</h1>
        <p>Upload a video, run the analysis, and view everything in one clean results page.</p>
    </div>
    """)

    with gr.Group(visible=True) as upload_page:
        gr.HTML("""
        <div class="upload-card">
            <h2>Upload your video</h2>
            <p>Choose a clear short clip where faces are visible.</p>
        </div>
        """)
        input_video = gr.Video(label="Upload Video", height=330)
        analyze_btn = gr.Button("Start Analysis", variant="primary")

    with gr.Group(visible=False) as results_page:
        summary_output = gr.HTML()
        output_video = gr.Video(label="Annotated Video", height=360)
        dashboard_plot = gr.Plot(label="Video Insights")

        with gr.Accordion("Detailed Results and Downloads", open=False):
            results_table = gr.Dataframe(
                label="Detailed Results",
                interactive=False
            )
            download_btn = gr.DownloadButton(
                label="Download Results CSV",
                value=None,
                visible=False
            )

        reset_btn = gr.Button("Analyze Another Video")

    analyze_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=[
            upload_page,
            results_page,
            summary_output,
            output_video,
            dashboard_plot,
            results_table,
            download_btn
        ]
    ).then(
        fn=gradio_analyze,
        inputs=input_video,
        outputs=[
            output_video,
            summary_output,
            dashboard_plot,
            results_table,
            download_btn
        ]
    )

    reset_btn.click(
        fn=reset_app,
        inputs=None,
        outputs=[
            upload_page,
            results_page,
            summary_output,
            output_video,
            dashboard_plot,
            results_table,
            download_btn
        ]
    )

demo.queue()
demo.launch()
