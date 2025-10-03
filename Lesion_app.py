import streamlit as st
from PIL import Image
import pandas as pd
import warnings
import anyio

from app.services.classification_service import classify_image
from app.services.segmentation_service import run_segmentation
from app.services.gradcam_service import generate_gradcam

warnings.filterwarnings("ignore")

st.title("Skin Lesion Diagnostic APP")

with st.form("input_form"):
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    age = st.number_input("Enter age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Select sex", ["Male", "Female"])
    sex_mapping = {"male": 0, "female": 1}
    sex_num = sex_mapping[sex.lower()]

    localisation_list = [
        "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
        "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity"
    ]
    localisation = st.selectbox("Choose a localisation", localisation_list)
    localisation_num = localisation_list.index(localisation)

    task = st.selectbox("Choose task", ["Classification", "Segmentation"])

    submitted = st.form_submit_button("Go")

if submitted and uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=400)

    if task == "Segmentation":
        with st.spinner("Running segmentation..."):
            try:
                img = Image.open(uploaded_file).convert("RGB")
                data = run_segmentation(img)
                overlay_image = data['Overlay_Image']
                st.image(overlay_image, caption="Overlayed Segmentation", width=400)
                st.metric("Lesion Area (pixels)", data.get("Lesion_Area_pixels", 0))
                st.metric("Estimated Diameter (pixels)", round(data.get("Estimated_Diameter_pixels", 0), 2))
            except Exception as e:
                st.error(f"Segmentation failed: {e}")

    else:  # Classification
        payload = {"age": age, "sex": sex_num, "localisation": localisation_num}

        with st.spinner("Running classification..."):
            try:
                # Use anyio to safely run async function in Streamlit
                data = classify_image(uploaded_file, payload)
            except Exception as e:
                st.error(f"Classification failed: {e}")
                data = {}

        # Show classification results
        try:
            CLASSES = [
                "Melanocytic nevi", "Melanoma", "Benign keratosis-like lesions",
                "Basal cell carcinoma", "Actinic keratoses / intraepithelial carcinoma",
                "Vascular lesions", "Dermatofibroma"
            ]

            st.subheader("Classification Results")
            st.markdown(f"**Image Prediction:** {data.get('Image Prediction', 'N/A')}")
            st.markdown(f"**Image Mean Confidence:** {data.get('Image Mean Confidence', 0):.4f}")
            ci = data.get('Image 95% CI', [0, 0])
            st.markdown(f"**Image 95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
            st.markdown(f"**Metadata Prediction:** {data.get('Metadata Prediction', 'N/A')}")
            st.markdown(f"**Combined Prediction:** {data.get('Combined Prediction', 'N/A')}")

            st.subheader("Class Probabilities")
            combined_probs = data.get("Combined Probabilities", [[0]*len(CLASSES)])
            df_probs = pd.DataFrame({"Class": CLASSES, "Probability": combined_probs[0]})
            st.table(df_probs.style.format({"Probability": "{:.4f}"}))
        except Exception as e:
            st.error(f"Displaying results failed: {e}")

        # Grad-CAM overlay
        try:
            img = Image.open(uploaded_file).convert("RGB")
            gradcam_result = generate_gradcam(img)
            if "GradCAM_Image" in gradcam_result:
                st.subheader("Grad-CAM Overlay")
                st.image(gradcam_result["GradCAM_Image"], caption="Grad-CAM Overlay", width=400)
            else:
                st.warning(f"Grad-CAM failed: {gradcam_result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Grad-CAM generation failed: {e}")


            


            





        




