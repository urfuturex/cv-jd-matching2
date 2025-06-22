import streamlit as st
from cv_jd_matching import extract_text_from_pdf, compute_similarity

st.title("CV - JD Matching Demo")

st.write("Upload CV (PDF) và nhập JD (hoặc upload JD PDF) để đánh giá độ phù hợp.")

cv_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
jd_option = st.radio("Nhập JD bằng:", ["Text", "PDF"])

jd_text = ""
if jd_option == "Text":
    jd_text = st.text_area("Nhập nội dung JD tại đây")
else:
    jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd")
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)

if st.button("Đánh giá độ matching"):
    if not cv_file or not jd_text.strip():
        st.warning("Vui lòng upload CV và nhập JD!")
    else:
        with st.spinner("Đang xử lý..."):
            cv_text = extract_text_from_pdf(cv_file)
            score = compute_similarity(cv_text, jd_text)
        percent = round(score * 100, 1)
        st.success(f"Độ matching giữa CV và JD: **{percent}%**")
        if score > 0.7:
            st.info("CV này rất phù hợp với JD!")
        elif score > 0.5:
            st.info("CV này khá phù hợp với JD.")
        else:
            st.info("CV này chưa thực sự phù hợp với JD.")
