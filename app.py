
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image



st.set_page_config(
    page_title="DermaAI | Skin Analysis",
    page_icon="🔬",
    layout="wide"
)



st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
    color: #17324d;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.header {
    background: linear-gradient(120deg, #102a43, #155e75);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 30px;
}

.header h1 {
    color: #ffffff;
    font-size: 38px;
    font-weight: 750;
    margin: 0;
}

.header p {
    color: #d5e8f0;
    font-size: 16px;
    margin-top: 10px;
}

/* Section titles */
.section-title {
    color: #17324d;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e1e9f0;
    border-radius: 16px;
    padding: 18px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #f8fbfd;
    border: 2px dashed #9bbdce;
    border-radius: 12px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    background: #087e8b;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 650;
    width: 100%;
}

.stButton > button:hover {
    background: #075d68;
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #f1f8f8;
    border: 1px solid #d5e9e7;
    border-radius: 12px;
    padding: 16px;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #e8f7f4, #f4fbfa);
    border: 1px solid #b7e4d9;
    border-radius: 14px;
    padding: 22px;
    margin-top: 10px;
}

.prediction-label {
    color: #47766e;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}

.prediction-value {
    color: #12665c;
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

/* Disclaimer */
.disclaimer {
    background: #fff8e8;
    border-left: 5px solid #e6a23c;
    padding: 18px;
    border-radius: 8px;
    color: #805b20;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #8293a3;
    font-size: 13px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "skin_cancer_cnn.h5"
    )

    with open("class_names.json", "r") as f:
        class_names = json.load(f)

    return model, class_names


try:
    model, class_names = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()



input_height = model.input_shape[1]
input_width = model.input_shape[2]




st.markdown("""
<div class="header">
<h1>🔬 DermaAI</h1>
<p>
Intelligent Skin Lesion Classification<br>
A Deep Learning research application powered by CNN.
</p>
</div>
""", unsafe_allow_html=True)




left, right = st.columns(
    [1, 1],
    gap="large"
)



with left:

    st.markdown(
        '<div class="section-title">📁 Image Upload</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        uploaded_file = st.file_uploader(
            "Upload a skin lesion image",
            type=["jpg", "jpeg", "png"],
            key="skin_image"
        )

        if uploaded_file is not None:

            try:
                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    caption="Uploaded Skin Image",
                    use_column_width=True
                )

                st.caption(
                    f"Image dimensions: "
                    f"{image.width} × {image.height}"
                )

            except Exception:
                st.error("Unable to read this image.")
                st.stop()

        else:

            st.info(
                "Choose a JPG, JPEG or PNG image "
                "to start the analysis."
            )




with right:

    st.markdown(
        '<div class="section-title">📊 Analysis Results</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        if uploaded_file is not None:

            predict_button = st.button(
                "🔍 Analyze Skin Image",
                type="primary"
            )

            if predict_button:

                with st.spinner(
                    "AI model is analyzing the image..."
                ):

                    # Resize image
                    processed_image = image.resize(
                        (input_width, input_height)
                    )

                    # Convert image to NumPy array
                    image_array = np.array(
                        processed_image
                    ).astype("float32")

                    # Normalize pixel values
                    image_array = image_array / 255.0

                    # Add batch dimension
                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    # Model prediction
                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    # Check class mapping
                    if len(class_names) != len(predictions):
                        st.error(
                            "Class names do not match "
                            "the model output."
                        )
                        st.stop()

                    # Get predicted class
                    predicted_index = np.argmax(
                        predictions
                    )

                    predicted_class = class_names[
                        predicted_index
                    ]

                    confidence = float(
                        predictions[predicted_index] * 100
                    )

                    # Save results in session state
                    st.session_state["prediction"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "scores": predictions.tolist(),
                        "filename": uploaded_file.name
                    }

        else:

            st.info(
                "Upload an image to display "
                "classification results here."
            )

if "prediction" in st.session_state:

    result = st.session_state["prediction"]

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(f"""
        <div class="prediction-card">
        <div class="prediction-label">PREDICTED CLASS</div>
        <div class="prediction-value">
        {result["class"]}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="prediction-card">
        <div class="prediction-label">MODEL CONFIDENCE</div>
        <div class="prediction-value">
        {result["confidence"]:.2f}%
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Top 3 Predictions")

    predictions = np.array(result["scores"])

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    for index in top_indices:

        score = float(predictions[index])

        st.write(
            f"**{class_names[index]}** — "
            f"{score * 100:.2f}%"
        )

        st.progress(score)

    st.caption(
        f"Analyzed file: {result['filename']}"
    )



st.markdown("""
<div class="disclaimer">
<b>⚠ Medical Disclaimer</b>
<br><br>
This application is developed for educational and research
purposes only. Its predictions are not medical diagnoses
and must not be used to guide treatment decisions.
Please consult a qualified dermatologist for professional
medical evaluation.
</div>
""", unsafe_allow_html=True)



st.markdown("""
<div class="footer">
DERMAAI | SKIN LESION CLASSIFICATION
<br>
Developed using Python • TensorFlow • Streamlit
</div>
""", unsafe_allow_html=True)
