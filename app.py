"""Run with: python -m streamlit run app.py"""

import io
import json

import streamlit as st
from PIL import Image, UnidentifiedImageError

from vehicle_lens.pipeline import VehicleLens, annotate, report

st.set_page_config(page_title="Vehicle Lens", page_icon="✈️", layout="wide")
st.markdown("<h4 style='color:#58bfff;letter-spacing:3px'>VEHICLE LENS</h4>", unsafe_allow_html=True)
st.title("Identify what moves.")
st.write("Find airplanes, boats, and cars. Explore likely models, one vehicle at a time.")
with st.sidebar:
    st.header("Analysis settings")
    threshold = st.slider("Detection confidence", 0.1, 0.95, 0.6, 0.05)
    identify = st.toggle("Estimate vehicle models", value=True)
    st.caption("First use downloads pretrained weights. Inference runs locally afterward.")
    st.caption("Model estimates depend on the candidate catalog. Scores are relative similarities.")


@st.cache_resource
def get_engine(identify):
    return VehicleLens(identify=identify)


upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
if upload:
    try:
        source = Image.open(upload)
        source.load()
        if source.width * source.height > 25_000_000:
            st.error("Choose an image under 25 megapixels.")
            st.stop()
        if st.button("Analyze image", type="primary"):
            with st.spinner("Loading models and analyzing vehicles…"):
                engine = get_engine(identify)
                image, detections = engine.predict(source, threshold)
                output = annotate(image, detections)
                st.session_state["result"] = (upload.getvalue(), image, detections, output)
        saved = st.session_state.get("result")
        if saved and saved[0] == upload.getvalue():
            _, image, detections, output = saved
            cols = st.columns(3)
            for col, category in zip(cols, ["airplane", "boat", "car"]):
                col.metric(category.title() + "s", sum(d.category == category for d in detections))
            left, right = st.columns([2, 1])
            with left:
                st.image(output, caption="Detected vehicles and tentative model estimates")
                buffer = io.BytesIO()
                output.save(buffer, format="PNG")
                st.download_button("Download annotated image", buffer.getvalue(),
                                   "vehicle-lens.png", "image/png")
            with right:
                st.subheader("Vehicle details")
                if not detections:
                    st.info("No vehicles found. Try a closer view or a lower detection threshold.")
                for index, detection in enumerate(detections, 1):
                    with st.expander(f"{index}. {detection.category.title()} · {detection.identification}"):
                        st.image(image.crop(tuple(detection.box)))
                        st.caption(detection.reason)
                        st.write(f"Detection confidence: {detection.confidence:.1%}")
                        for candidate in detection.candidates:
                            st.write(candidate["label"])
                            st.progress(candidate["relative_score"],
                                        text=f"Relative score: {candidate['relative_score']:.1%}")
                st.download_button("Download JSON report", json.dumps(report(image, detections), indent=2),
                                   "vehicle-lens.json", "application/json")
    except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not analyze this image: {error}")
else:
    st.info("Start with a clear aircraft photo, a harbor scene, or a street photo.")
    st.caption("For video: vehicle-lens clip.mp4 --video --stride 15")
