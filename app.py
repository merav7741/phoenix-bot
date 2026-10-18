import re
import streamlit as st

# Brand colors, change these if you want an exact match to the site
PRIMARY = "#E8501C"
PRIMARY_DARK = "#C23B0E"
PRIMARY_SOFT = "#FDEEE6"
INK = "#2B2B33"

st.set_page_config(page_title="הבוט של הפניקס", page_icon="🔥", layout="centered")

st.markdown(
    f"""
    <style>
   html, body, [data-testid="stAppViewContainer"], [data-testid="stChatInput"],
   [data-testid="stChatMessage"], [data-testid="stMarkdownContainer"],
    [data-testid="stVerticalBlock"], [data-testid="stBottom"],
    [data-testid="stElementContainer"] {{ direction: rtl; text-align: right; }}

    [data-testid="stBottom"] > div {{ background: #F6F4F2; }}
    [data-testid="stChatInput"] textarea {{ direction: rtl; text-align: right; }}
    [data-testid="stMarkdownContaKiner"] p {{ text-align: right; }}
    [data-testid="stAppViewContainer"] {{ background: #F6F4F2; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ max-width: 720px; padding-top: 1.2rem; padding-bottom: 5rem; }}

    .hero {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        color: white; border-radius: 18px; padding: 18px 22px; margin-bottom: 14px;
        box-shadow: 0 6px 18px rgba(232, 80, 28, 0.25);
    }}
    .hero h1 {{ font-size: 1.35rem; margin: 0; color: white; padding: 0; }}
    .hero p {{ margin: 4px 0 0 0; font-size: 0.9rem; opacity: 0.92; }}

    [data-testid="stChatMessage"] {{
        background: white; border-radius: 16px; padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 10px;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: {PRIMARY_SOFT};
    }}

    .src {{ margin-top: 10px; font-size: 0.82rem; }}
    .src a {{
        display: inline-block; background: {PRIMARY_SOFT}; color: {PRIMARY_DARK};
        padding: 3px 10px; border-radius: 999px; text-decoration: none; font-weight: 600;
    }}
    .src a:hover {{ background: {PRIMARY}; color: white; }}

    .stButton > button {{
        border-radius: 999px; border: 1px solid {PRIMARY}; color: {PRIMARY_DARK};
        background: white; font-size: 0.85rem; padding: 4px 14px;
    }}
    .stButton > button:hover {{ background: {PRIMARY}; color: white; border-color: {PRIMARY}; }}
    </style>

    <div class="hero">
        <h1>🔥 הבוט שיודע את האתר של הפניקס</h1>
        <p>עונה רק מתוך המידע שבאתר, ותמיד עם מקור</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def get_answer(question, history):
    """Dummy answer. We will replace this with the real RAG pipeline later."""
    return {
        "answer": "זו תשובת דמה. בהמשך כאן תופיע תשובה אמיתית מתוך האתר.",
        "sources": ["https://www.fnx.co.il/claims-customer-service/"],
    }


def render_sources(urls):
    if not urls:
        return ""
    links = " ".join(
        f'<a href="{u}" target="_blank">מקור {i}</a>' for i, u in enumerate(urls, 1)
    )
    return f'<div class="src">{links}</div>'


if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "יש לי נזילה בבית, איך מגישים תביעה?",
    "מה כולל ביטוח צד ג' בדירה?",
    "איך מגישים תביעה בביטוח סיעודי?",
]

clicked = None
if not st.session_state.messages:
    st.markdown("**נסי לשאול, למשל:**")
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            clicked = ex

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

question = st.chat_input("שאלי אותי על ביטוח, תביעות ועוד...") or clicked

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = get_answer(question, st.session_state.messages)
    html = result["answer"] + render_sources(result["sources"])

    st.session_state.messages.append({"role": "assistant", "content": html})
    with st.chat_message("assistant"):
        st.markdown(html, unsafe_allow_html=True)
    if clicked:
        st.rerun()