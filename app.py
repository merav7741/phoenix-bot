from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import streamlit as st

from rag import RAGError, get_answer

PRIMARY = "#F15A24"
NAVY = "#101D3A"
INK = "#17233B"
CLAIMS_CENTER_URL = "https://www.fnx.co.il/claims-customer-service/"

st.set_page_config(
    page_title="הבוט החכם של הפניקס",
    page_icon=".streamlit/blank.svg",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMarkdownContainer"], [data-testid="stVerticalBlock"],
    [data-testid="stElementContainer"], [data-testid="stChatMessage"] {{
        direction: rtl;
        text-align: right;
    }}
    [data-testid="stHeader"] {{ display: none; }}
    [data-testid="stMain"] > div {{ padding-top: 0; }}
    .block-container {{
        max-width: 1440px;
        padding: 0 0.8rem 4rem;
    }}
    .site-utility {{
        background: #0B1227;
        color: #E9ECF4;
        padding: 7px max(calc((100vw - 1240px) / 2), 28px);
        font-size: 0.76rem;
        text-align: left;
        letter-spacing: 0.02em;
    }}
    .site-nav {{
        background: white;
        border-bottom: 1px solid #E9ECF2;
        min-height: 74px;
        padding: 0 max(calc((100vw - 1240px) / 2), 28px);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 32px;
    }}
    .site-logo {{
        color: {NAVY};
        font-size: 1.7rem;
        font-weight: 800;
        white-space: nowrap;
    }}
    .site-logo span {{
        display: inline-block;
        width: 25px;
        height: 4px;
        margin: 0 0 7px 8px;
        border-radius: 3px;
        background: {PRIMARY};
    }}
    .site-links {{
        display: flex;
        align-items: center;
        gap: clamp(14px, 2.5vw, 36px);
        color: #26324A;
        font-size: 0.92rem;
        font-weight: 650;
    }}
    .hero {{
        background: {NAVY};
        color: white;
        min-height: 410px;
        padding: 62px max(calc((100vw - 1240px) / 2), 28px) 54px;
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
        align-items: center;
        gap: 48px;
        overflow: hidden;
        position: relative;
    }}
    .hero-copy {{ position: relative; z-index: 1; }}
    .hero-eyebrow {{
        color: #FF8053;
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 14px;
    }}
    .hero h1 {{
        color: white;
        font-size: clamp(2.4rem, 5vw, 4.3rem);
        line-height: 1.12;
        font-weight: 750;
        margin: 0;
        max-width: 680px;
    }}
    .hero-copy > p:last-child {{
        color: #D4D9E5;
        font-size: 1.12rem;
        line-height: 1.8;
        margin: 20px 0 0;
        max-width: 620px;
    }}
    .hero-art {{
        min-height: 260px;
        position: relative;
        display: grid;
        place-items: center;
    }}
    .hero-orbit {{
        width: min(310px, 68vw);
        aspect-ratio: 1;
        border: 1px solid rgba(255,255,255,0.23);
        border-radius: 50%;
        display: grid;
        place-items: center;
        transform: rotate(-14deg);
    }}
    .hero-orbit::before {{
        content: "";
        width: 76%;
        aspect-ratio: 1;
        border: 2px solid {PRIMARY};
        border-radius: 50%;
        display: block;
    }}
    .hero-spark {{
        position: absolute;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: {PRIMARY};
        right: 16%;
        top: 16%;
        box-shadow: 0 0 26px rgba(241,90,36,0.7);
    }}
    .section-heading {{
        color: {INK};
        font-size: 1.65rem;
        font-weight: 750;
        margin: 48px 0 8px;
    }}
    .section-intro {{
        color: #687287;
        margin: 0 0 20px;
        font-size: 0.98rem;
    }}
    [class*="st-key-topic_card_"] {{
        min-height: 148px;
        padding: 18px 18px;
        background: white;
        border: 1px solid #E5E8EF;
        border-radius: 14px;
        box-shadow: 0 8px 28px rgba(16,29,58,0.055);
    }}
    [class*="st-key-topic_card_"] [data-testid="stMarkdownContainer"] {{
        color: {NAVY};
    }}
    [class*="st-key-topic_card_"] h3 {{
        color: {NAVY};
        font-size: 1.05rem;
        margin: 10px 0 5px;
    }}
    [class*="st-key-topic_card_"] [data-testid="stCaptionContainer"] {{
        color: #6D7688;
        font-size: 0.84rem;
    }}
    .prototype-note {{
        color: #8991A0;
        font-size: 0.76rem;
        text-align: center;
        padding: 38px 0 0;
    }}

    .st-key-chat_launcher,
    .st-key-chat_minimized {{
        position: fixed;
        right: 28px;
        bottom: 24px;
        width: auto;
        z-index: 1000;
    }}
    .st-key-chat_launcher button,
    .st-key-chat_minimized button {{
        min-height: 58px;
        border: 0;
        border-radius: 999px;
        padding: 0 25px;
        background: {PRIMARY};
        color: white;
        font-size: 0.98rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(181, 59, 19, 0.32);
        transition: transform 150ms ease, box-shadow 150ms ease;
    }}
    .st-key-chat_launcher button:hover,
    .st-key-chat_minimized button:hover {{
        background: #E64D19;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 14px 34px rgba(181, 59, 19, 0.38);
    }}
    .st-key-chat_panel {{
        position: fixed;
        z-index: 1001;
        right: 28px;
        bottom: 96px;
        width: min(400px, calc(100vw - 32px));
        height: min(650px, calc(100vh - 124px));
        max-height: calc(100vh - 124px);
        display: flex;
        overflow-x: hidden;
        overflow-y: auto;
        overscroll-behavior: contain;
        scrollbar-width: thin;
        scrollbar-color: #B8C1CF transparent;
        padding: 0 18px 16px;
        background: white;
        border: 1px solid #E3E7EF;
        border-top: 3px solid {PRIMARY};
        border-radius: 18px;
        box-shadow: 0 20px 70px rgba(12, 22, 45, 0.24);
    }}
    .st-key-chat_panel [data-testid="stVerticalBlock"] {{
        gap: 0.55rem;
    }}
    .st-key-chat_header {{
        position: sticky;
        top: 0;
        z-index: 2;
        padding: 13px 0 11px;
        border-bottom: 1px solid #EDF0F4;
        margin-bottom: 8px;
        background: white;
    }}
    .st-key-chat_header .brand-line {{
        color: {PRIMARY};
        font-size: 0.76rem;
        font-weight: 750;
        margin: 0 0 3px;
    }}
    .st-key-chat_header .brand-title {{
        color: {NAVY};
        font-size: 1.05rem;
        font-weight: 750;
        margin: 0;
    }}
    .st-key-chat_header .brand-desc {{
        color: #70798A;
        font-size: 0.79rem;
        margin: 3px 0 0;
    }}
    .st-key-chat_header > [data-testid="stHorizontalBlock"] {{
        align-items: center;
        gap: 6px;
    }}
    .st-key-chat_header > [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.st-key-close_chat),
    .st-key-chat_header > [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.st-key-minimize_chat) {{
        flex: 0 0 42px !important;
        width: 42px !important;
        min-width: 42px !important;
        padding: 3px;
        border: 1px solid #F7B28F;
        border-radius: 12px;
        background: #FFF3EC;
        box-sizing: border-box;
    }}
    .st-key-chat_header > [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.st-key-close_chat) > div,
    .st-key-chat_header > [data-testid="stHorizontalBlock"] > [data-testid="column"]:has(.st-key-minimize_chat) > div {{
        width: 100%;
    }}
    .st-key-chat_notice {{
        margin: 10px 0 12px;
        padding: 10px 12px;
        border: 1px solid #F7B28F;
        border-radius: 11px;
        background: #FFF3EC;
        color: #713313;
    }}
    .st-key-chat_notice [data-testid="stMarkdownContainer"] {{
        color: #713313;
        font-size: 0.77rem;
        line-height: 1.55;
    }}
    .st-key-chat_notice_close button {{
        min-height: 30px;
        width: 30px;
        padding: 0;
        border: 0;
        border-radius: 50%;
        background: transparent;
        color: #8B4B2A;
    }}
    .st-key-chat_notice_close button p {{
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }}
    .st-key-chat_welcome {{
        padding: 15px 10px 13px;
        text-align: center;
    }}
    .st-key-chat_welcome [data-testid="stMarkdownContainer"] {{
        text-align: center;
    }}
    .st-key-welcome_icon [data-testid="stMarkdownContainer"] p {{
        color: {PRIMARY};
        font-size: 2.1rem;
        line-height: 1;
        margin: 0 0 8px;
    }}
    .st-key-chat_welcome h3 {{
        color: {NAVY};
        font-size: 1.05rem;
        margin: 0 0 5px;
    }}
    .st-key-chat_welcome p {{
        color: #738097;
        font-size: 0.8rem;
        line-height: 1.55;
        margin: 0;
    }}
    .st-key-close_chat button,
    .st-key-minimize_chat button {{
        min-height: 36px;
        width: 100%;
        padding: 0;
        border: 0;
        border-radius: 8px;
        color: #9A451F;
        background: transparent;
    }}
    .st-key-close_chat button:hover,
    .st-key-minimize_chat button:hover {{
        border: 0;
        color: #7C3215;
        background: #FFE3D4;
    }}
    .st-key-chat_mode button {{
        min-height: 34px;
        padding: 0.3rem 0.7rem;
        font-size: 0.84rem;
    }}
    .st-key-close_chat button p,
    .st-key-minimize_chat button p {{
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }}
    .st-key-chat_composer {{
        position: sticky;
        bottom: 0;
        z-index: 3;
        padding: 10px 0 2px;
        border-top: 1px solid #E7ECF3;
        background: white;
    }}
    .st-key-chat_messages {{
        padding: 4px 2px;
    }}
    .st-key-chat_panel [data-testid="stChatMessage"] {{
        padding: 10px 12px;
        margin: 5px 0;
        border: 1px solid #E7EAF0;
        border-radius: 13px;
        box-shadow: none;
    }}
    .st-key-chat_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: #F2F6FC;
        border-color: #C9D8EF;
    }}
    .st-key-chat_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background: #FFFFFF;
        border-color: #DCE6F3;
    }}
    .st-key-chat_panel [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {{
        color: #26324A;
        font-size: 0.88rem;
        line-height: 1.65;
    }}
    .st-key-chat_panel [data-testid="stTextInput"] input {{
        direction: rtl;
        border-radius: 11px;
        min-height: 44px;
    }}
    .st-key-chat_send button {{
        min-height: 44px;
        border-radius: 11px;
        background: {PRIMARY};
        border-color: {PRIMARY};
        color: white;
        font-weight: 700;
    }}
    .st-key-chat_send button:hover {{
        background: #E64D19;
        border-color: #E64D19;
        color: white;
    }}
    .st-key-chat_suggestions button {{
        min-height: 36px;
        border-radius: 10px;
        text-align: right;
        font-size: 0.82rem;
        color: {NAVY};
        border-color: #D2DEEF;
        background: #F8FAFD;
    }}
    .st-key-chat_suggestions button p {{
        width: 100%;
        text-align: right;
    }}
    .st-key-chat_suggestions button:hover {{
        border-color: #7895C1;
        color: {NAVY};
    }}
    .st-key-chat_panel [data-testid="stLinkButton"] a {{
        min-height: 34px;
        border-radius: 9px;
        border-color: #D2DEEF;
        color: {NAVY};
        font-size: 0.78rem;
        background: #F8FAFD;
    }}
    .st-key-chat_panel [data-testid="stStatusWidget"] {{
        direction: rtl;
        text-align: right;
        color: #687287;
        font-size: 0.8rem;
    }}

    @media (max-width: 700px) {{
        .site-utility {{ padding: 7px 18px; }}
        .site-nav {{
            min-height: 62px;
            padding: 0 18px;
        }}
        .site-links {{ display: none; }}
        .hero {{
            min-height: 360px;
            grid-template-columns: 1fr;
            padding: 44px 22px 32px;
            gap: 8px;
        }}
        .hero h1 {{ font-size: clamp(2.15rem, 10vw, 3.2rem); }}
        .hero-art {{
            position: absolute;
            left: -60px;
            bottom: -84px;
            opacity: 0.27;
        }}
        .section-heading {{ margin-top: 34px; font-size: 1.35rem; }}
        .st-key-chat_launcher {{ right: 16px; bottom: 16px; }}
        .st-key-chat_panel {{
            right: 12px;
            bottom: 84px;
            width: calc(100vw - 24px);
            max-height: calc(100vh - 102px);
        }}
    }}
    </style>
    <div class="site-utility">מידע ושירות · פרויקט הדגמה</div>
    <header class="site-nav">
        <div class="site-logo"><span></span> הפניקס</div>
        <nav class="site-links" aria-label="ניווט ראשי">
            <span>ביטוחים</span>
            <span>פנסיה וגמל</span>
            <span>השקעות</span>
            <span>תביעות</span>
            <span>שירות ופעולות</span>
        </nav>
    </header>
    <section class="hero">
        <div class="hero-copy">
            <p class="hero-eyebrow">המידע הביטוחי, עכשיו פשוט יותר</p>
            <h1>תשובות ברורות.<br>בשפה שלך.</h1>
            <p>שואלים שאלה על ביטוח, כיסוי או תביעה — ומקבלים תשובה ממידע ציבורי באתר, עם קישור למקור.</p>
        </div>
        <div class="hero-art" aria-hidden="true">
            <div class="hero-orbit"></div>
            <div class="hero-spark"></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h2 class="section-heading">במה נוכל לעזור?</h2>'
    '<p class="section-intro">מידע נגיש על הנושאים החשובים לכם.</p>',
    unsafe_allow_html=True,
)

topics = [
    (":material/directions_car:", "ביטוח רכב", "מידע על כיסויים ותהליכים."),
    (":material/home:", "ביטוח דירה", "תשובות לשאלות על הבית."),
    (":material/health_and_safety:", "ביטוח בריאות", "מידע על שירותים וכיסויים."),
    (":material/description:", "תביעות", "הסבר על הגשת תביעה."),
]
topic_columns = st.columns(4, gap="medium")
for index, (column, (icon, title, description)) in enumerate(
    zip(topic_columns, topics)
):
    with column:
        with st.container(key=f"topic_card_{index}"):
            st.markdown(icon)
            st.markdown(f"### {title}")
            st.caption(description)

CLAIM_GUIDES = {
    "רכב — נזק לרכב": f"{CLAIMS_CENTER_URL}claims-car-insurance/claims-property-car-insurance/",
    "רכב — ביטוח חובה": f"{CLAIMS_CENTER_URL}claims-car-insurance/compulsory-motor-claims/",
    "רכב — צד ג׳": f"{CLAIMS_CENTER_URL}claims-car-insurance-third-party/",
    "דירה": f"{CLAIMS_CENTER_URL}claims-appartment-insurance/",
    "בריאות": f"{CLAIMS_CENTER_URL}claims-health-insurance/",
    "שיניים": f"{CLAIMS_CENTER_URL}claims-dental-insurance/",
    "נסיעות לחו״ל": f"{CLAIMS_CENTER_URL}claims-travel-abroad/",
    "חיים": f"{CLAIMS_CENTER_URL}claims-life-insurance/",
    "תאונות אישיות": f"{CLAIMS_CENTER_URL}claims-personal-accident-insurance/",
    "חיות מחמד": f"{CLAIMS_CENTER_URL}claims-pets/",
    "עסק": f"{CLAIMS_CENTER_URL}claims-business-insurance/",
    "אחריות": f"{CLAIMS_CENTER_URL}claims-liability-insurance/",
    "סיעוד": f"{CLAIMS_CENTER_URL}claims-supportive-insurance/",
    "לא בטוחה / סוג אחר": CLAIMS_CENTER_URL,
}

st.markdown(
    '<p class="prototype-note">אב־טיפוס להדגמה · המידע והתשובות מבוססים על מקורות ציבוריים באתר</p>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_minimized" not in st.session_state:
    st.session_state.chat_minimized = False
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "שאלות"
if "answer_future" not in st.session_state:
    st.session_state.answer_future = None
if "answer_progress_queue" not in st.session_state:
    st.session_state.answer_progress_queue = None
if "answer_progress" not in st.session_state:
    st.session_state.answer_progress = "מחפשת מידע ובודקת את המקורות..."
if "chat_notice_visible" not in st.session_state:
    st.session_state.chat_notice_visible = True


@st.cache_resource
def _answer_executor():
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="phoenix-rag")


def _run_answer(question, history, progress_queue):
    return get_answer(question, history, on_progress=progress_queue.put)


def _queue_question(question=None):
    if st.session_state.get("answer_future") is not None:
        return
    text = (question or st.session_state.get("chat_question", "")).strip()
    if text:
        st.session_state.pending_question = text
        st.session_state.chat_question = ""


def _close_chat():
    future = st.session_state.get("answer_future")
    if future is not None:
        future.cancel()
    st.session_state.messages = []
    st.session_state.chat_open = False
    st.session_state.chat_minimized = False
    st.session_state.chat_question = ""
    st.session_state.answer_future = None
    st.session_state.answer_progress_queue = None
    st.session_state.answer_progress = "מחפשת מידע ובודקת את המקורות..."
    st.session_state.chat_notice_visible = True
    st.session_state.chat_mode = "שאלות"
    st.session_state.pop("claim_guide_selected", None)
    st.session_state.pop("claim_guide_type", None)
    st.session_state.pop("claim_guide_messages", None)
    st.session_state.pop("claim_guide_question", None)
    st.session_state.pop("claim_guide_error", None)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("chat_error", None)
    st.session_state.pop("scroll_after_chat_update", None)


def _minimize_chat():
    st.session_state.chat_open = False
    st.session_state.chat_minimized = True


def _restore_chat():
    st.session_state.chat_open = True
    st.session_state.chat_minimized = False


def _dismiss_chat_notice():
    st.session_state.chat_notice_visible = False


def _start_claim_guide():
    selected_type = st.session_state.get("claim_guide_type")
    if selected_type not in CLAIM_GUIDES:
        st.session_state.claim_guide_error = "בחרי תחום ביטוח מהרשימה."
        return
    st.session_state.claim_guide_selected = selected_type
    st.session_state.claim_guide_messages = []
    st.session_state.claim_guide_question = ""
    st.session_state.pop("claim_guide_error", None)


def _submit_claim_guide_question():
    question = st.session_state.get("claim_guide_question", "").strip()
    selected_type = st.session_state.get("claim_guide_selected")
    source_url = CLAIM_GUIDES.get(selected_type)
    if not source_url:
        st.session_state.claim_guide_error = (
            "בחרי תחום ביטוח לפני שממשיכים."
        )
        return
    if not question:
        st.session_state.claim_guide_error = "כתבי מה קרה או מה תרצי לברר."
        return

    try:
        from rag import get_claim_guide

        chunks = get_claim_guide(source_url, question=question)
    except ImportError:
        st.session_state.claim_guide_error = (
            "רכיב הליווי לא זמין בגרסה הנוכחית. רענני את האפליקציה, "
            "ואם ההודעה נשארת יש לעדכן את קובצי הפרויקט יחד."
        )
    except RAGError as error:
        st.session_state.claim_guide_error = str(error)
    else:
        st.session_state.claim_guide_messages.append(
            {"question": question, "sources": chunks}
        )
        st.session_state.pop("claim_guide_error", None)
    st.session_state.claim_guide_question = ""


def _change_claim_guide_type():
    st.session_state.claim_guide_selected = None
    st.session_state.claim_guide_messages = []
    st.session_state.claim_guide_question = ""
    st.session_state.pop("claim_guide_error", None)


EXAMPLES = [
    "איך מגישים תביעה על נזק בבית?",
    "מה כולל ביטוח צד ג׳?",
    "איך מגישים תביעה בביטוח בריאות?",
]

if not st.session_state.chat_open and st.session_state.chat_minimized:
    with st.container(key="chat_minimized"):
        st.button(
            "המשך בשיחה",
            key="restore_chat",
            icon=":material/chat:",
            type="primary",
            on_click=_restore_chat,
        )
elif not st.session_state.chat_open:
    with st.container(key="chat_launcher"):
        st.button(
            "הבוט החכם של הפניקס",
            key="open_chat",
            icon=":material/chat:",
            type="primary",
            on_click=_restore_chat,
        )
else:
    @st.fragment(
        run_every=0.4 if st.session_state.answer_future is not None else None
    )
    def render_chat_panel():
        future = st.session_state.answer_future
        if future is not None:
            progress_queue = st.session_state.answer_progress_queue
            while True:
                try:
                    st.session_state.answer_progress = progress_queue.get_nowait()
                except Empty:
                    break

            if future.done():
                try:
                    result = future.result()
                except RAGError as error:
                    st.session_state.chat_error = str(error)
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                finally:
                    st.session_state.answer_future = None
                    st.session_state.answer_progress_queue = None
                st.session_state.scroll_after_chat_update = True
                st.rerun(scope="app")

        pending_question = st.session_state.pop("pending_question", "")
        if pending_question and st.session_state.answer_future is None:
            st.session_state.messages.append(
                {"role": "user", "content": pending_question}
            )
            progress_queue = Queue()
            st.session_state.answer_progress_queue = progress_queue
            st.session_state.answer_progress = "מחפשת מידע ובודקת את המקורות..."
            st.session_state.answer_future = _answer_executor().submit(
                _run_answer,
                pending_question,
                list(st.session_state.messages),
                progress_queue,
            )
            st.session_state.scroll_after_chat_update = True
            st.rerun(scope="app")

        busy = st.session_state.answer_future is not None
        with st.container(key="chat_panel"):
            with st.container(key="chat_header"):
                close_column, minimize_column, title_column = st.columns(
                    [1, 1, 8], vertical_alignment="center"
                )
                with close_column:
                    if st.button(
                        "מחיקת השיחה וסגירה",
                        key="close_chat",
                        icon=":material/close:",
                        help="ה-X מוחק את כל ההודעות. פתיחה מחדש תתחיל שיחה חדשה.",
                    ):
                        _close_chat()
                        st.rerun(scope="app")
                with minimize_column:
                    if st.button(
                        "מזעור חלונית הצ׳אט",
                        key="minimize_chat",
                        icon=":material/minimize:",
                        help="מזעור (–) מסתיר את הצ׳אט בלי למחוק את ההודעות.",
                    ):
                        _minimize_chat()
                        st.rerun(scope="app")
                with title_column:
                    st.markdown(
                        """
                        <p class="brand-line">הפניקס · מידע ושירות</p>
                        <p class="brand-title">הבוט החכם של הפניקס</p>
                        <p class="brand-desc">אשמח לעזור למצוא מידע על ביטוחים ותביעות.</p>
                        """,
                        unsafe_allow_html=True,
                    )

            if st.session_state.chat_notice_visible:
                notice_text, notice_close = st.columns(
                    [9, 1], vertical_alignment="center"
                )
                with notice_text:
                    with st.container(key="chat_notice"):
                        st.markdown(
                            "ℹ️ המידע מבוסס על מקורות ציבוריים באתר הפניקס. "
                            "הבוט אינו ניגש לפרטי פוליסה אישיים."
                        )
                with notice_close:
                    with st.container(key="chat_notice_close"):
                        st.button(
                            "סגירת הודעת המידע",
                            key="dismiss_chat_notice",
                            icon=":material/close:",
                            help="סגירת הודעת המידע",
                            on_click=_dismiss_chat_notice,
                        )

            chat_mode = st.segmented_control(
                "איך תרצי להיעזר בבוט?",
                ["שאלות", "ליווי"],
                label_visibility="collapsed",
                key="chat_mode",
                width="content",
            )

            if chat_mode == "שאלות":
                with st.container(key="chat_messages"):
                    if not st.session_state.messages:
                        with st.container(key="chat_welcome"):
                            with st.container(key="welcome_icon"):
                                st.markdown(":material/support_agent:")
                            st.markdown("### שלום, איך אפשר לעזור?")
                            st.caption(
                                "שאלו אותי על ביטוח, כיסוי או תביעה. "
                                "אמצא מידע באתר ואצרף קישור למקור."
                            )

                    for message in st.session_state.messages:
                        avatar = (
                            ":material/person:"
                            if message["role"] == "user"
                            else ":material/support_agent:"
                        )
                        with st.chat_message(message["role"], avatar=avatar):
                            st.markdown(message["content"])
                            for index, source in enumerate(
                                message.get("sources", []), 1
                            ):
                                st.link_button(
                                    f"מקור {index} · {source['title']}",
                                    source["url"],
                                    icon=":material/open_in_new:",
                                    width="stretch",
                                )

                if busy:
                    st.status(
                        f":shimmer[{st.session_state.answer_progress}]",
                        state="running",
                        expanded=False,
                        type="compact",
                    )

                if "chat_error" in st.session_state:
                    st.error(st.session_state.pop("chat_error"))

                if not st.session_state.messages:
                    with st.container(key="chat_suggestions"):
                        st.caption("אפשר להתחיל באחת השאלות:")
                        for index, example in enumerate(EXAMPLES):
                            st.button(
                                example,
                                key=f"suggestion_{index}",
                                width="stretch",
                                disabled=busy,
                                on_click=_queue_question,
                                args=(example,),
                            )

                with st.container(key="chat_composer"):
                    input_column, send_column = st.columns(
                        [5, 1], vertical_alignment="bottom"
                    )
                    with input_column:
                        st.text_input(
                            "השאלה שלך",
                            placeholder="כתבו שאלה על ביטוח או תביעה...",
                            label_visibility="collapsed",
                            key="chat_question",
                            on_change=_queue_question,
                        )
                    with send_column:
                        st.button(
                            "שליחה",
                            key="send_chat",
                            icon=":material/arrow_upward:",
                            width="stretch",
                            disabled=busy,
                            on_click=_queue_question,
                        )
            else:
                with st.container(key="claim_guide_chat"):
                    st.markdown("### היי, אני המלווה של הפניקס")
                    st.caption(
                        "אלווה אותך במידע ציבורי לפני הגשת תביעה. "
                        "אני לא בודק זכאות ולא מחליף את הנחיות הפניקס."
                    )

                    selected_claim_guide = st.session_state.get(
                        "claim_guide_selected"
                    )
                    if selected_claim_guide not in CLAIM_GUIDES:
                        st.markdown(
                            "**באיזה סוג ביטוח תרצי שאלווה אותך?**"
                        )
                        with st.form("claim_guide_start_form"):
                            st.selectbox(
                                "סוג הביטוח",
                                list(CLAIM_GUIDES),
                                key="claim_guide_type",
                            )
                            st.form_submit_button(
                                "המשך",
                                key="start_claim_guide",
                                type="primary",
                                on_click=_start_claim_guide,
                            )
                    else:
                        st.markdown(
                            f"**נושא הליווי: {selected_claim_guide}**"
                        )
                        st.button(
                            "בחירת סוג ביטוח אחר",
                            key="change_claim_guide_type",
                            on_click=_change_claim_guide_type,
                        )

                        if not st.session_state.claim_guide_messages:
                            st.info(
                                "ספרי לי בקצרה מה קרה או מה תרצי לברר. "
                                "אין צורך לכתוב שם, מספר זהות או פרטי פוליסה."
                            )

                        for turn_index, turn in enumerate(
                            st.session_state.claim_guide_messages
                        ):
                            with st.chat_message("user"):
                                st.markdown(turn["question"])
                            with st.chat_message("assistant"):
                                if turn["sources"]:
                                    st.markdown(
                                        "מצאתי קטעים רלוונטיים ממידע רשמי "
                                        "של הפניקס. הנה המקורות לעיון:"
                                    )
                                    for index, chunk in enumerate(
                                        turn["sources"], 1
                                    ):
                                        st.markdown(f"**קטע {index}:**")
                                        st.markdown(chunk["text"])
                                        if chunk["crawled_at"]:
                                            st.caption(
                                                "המידע נאסף מהאתר בתאריך "
                                                f"{chunk['crawled_at']}."
                                            )
                                        st.link_button(
                                            f"פתיחת המקור הרשמי {index}",
                                            chunk["url"],
                                            icon=":material/open_in_new:",
                                            key=(
                                                "claim_guide_source_"
                                                f"{turn_index}_{index}"
                                            ),
                                        )
                                    st.caption(
                                        "אפשר להמשיך לשאול על אותו תחום. "
                                        "המידע כללי ואינו קובע אם תביעה תאושר."
                                    )
                                else:
                                    st.warning(
                                        "לא מצאתי קטע מתאים במקורות שאונדקסו "
                                        "לתחום הזה. לא אנחש; אפשר לנסח את "
                                        "השאלה אחרת או לפתוח את מרכז התביעות."
                                    )
                                    st.link_button(
                                        "פתיחת מרכז התביעות של הפניקס",
                                        CLAIMS_CENTER_URL,
                                        icon=":material/open_in_new:",
                                    )

                        if "claim_guide_error" in st.session_state:
                            st.error(st.session_state.claim_guide_error)

                        with st.form("claim_guide_question_form"):
                            st.text_area(
                                "מה קרה או מה תרצי לשאול?",
                                placeholder=(
                                    "לדוגמה: הייתה נזילה בדירה, איזה מידע "
                                    "כדאי לבדוק לפני הגשת תביעה?"
                                ),
                                key="claim_guide_question",
                            )
                            st.form_submit_button(
                                "שליחה למלווה",
                                key="submit_claim_guide_question",
                                type="primary",
                                on_click=_submit_claim_guide_question,
                            )

            if st.session_state.pop("scroll_after_chat_update", False):
                st.html(
                    """
                    <script>
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        const panel = document.querySelector(".st-key-chat_panel");
                        if (panel) {
                            panel.scrollTo({
                                top: panel.scrollHeight,
                                behavior: "smooth"
                            });
                        }
                    }));
                    </script>
                    """,
                    unsafe_allow_javascript=True,
                )

    render_chat_panel()
