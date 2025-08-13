import streamlit as st
# 同樣把之前的函式 import 進來：
from bm25_version1 import search_context, generate_prompt, call_qwen

st.title("台大資工系法規助理")
st.write("請輸入你的問題，我會根據法規資料回答。")

question = st.text_input("學生問題：")
if st.button("送出"):
    with st.spinner("系統思考中…"):
        ctx = search_context(question)
        prompt = generate_prompt(question, ctx)
        answer = call_qwen(prompt)
    st.text_area("助理回答：", value=answer, height=200)