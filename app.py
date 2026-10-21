from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import streamlit as st

from rag import RAGError, get_answer

PRIMARY = "#F15A24"
NAVY = "#101D3A"
INK = "#17233B"

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

    .st-key-chat_launcher {{
        position: fixed;
        right: 28px;
        bottom: 24px;
        width: auto;
        z-index: 1000;
    }}
    .st-key-chat_launcher button {{
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
    .st-key-chat_launcher button:hover {{
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
        max-height: min(650px, calc(100vh - 124px));
        overflow-y: auto;
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
    .st-key-close_chat button {{
        min-height: 36px;
        width: 36px;
        padding: 0;
        border: 1px solid #E5E8EF;
        border-radius: 50%;
        color: #5C6678;
        background: white;
    }}
    .st-key-close_chat button p {{
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

st.markdown(
    '<p class="prototype-note">אב־טיפוס להדגמה · המידע והתשובות מבוססים על מקורות ציבוריים באתר</p>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
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
    st.session_state.chat_open = False


def _dismiss_chat_notice():
    st.session_state.chat_notice_visible = False


EXAMPLES = [
    "איך מגישים תביעה על נזק בבית?",
    "מה כולל ביטוח צד ג׳?",
    "איך מגישים תביעה בביטוח בריאות?",
]

if not st.session_state.chat_open:
    with st.container(key="chat_launcher"):
        if st.button(
            "הבוט החכם של הפניקס",
            key="open_chat",
            icon=":material/chat:",
            type="primary",
        ):
            st.session_state.chat_open = True
            st.rerun()
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
                close_column, title_column = st.columns(
                    [1, 9], vertical_alignment="center"
                )
                with close_column:
                    st.button(
                        "סגירת חלונית הצ׳אט",
                        key="close_chat",
                        icon=":material/close:",
                        help="סגירת חלונית הצ׳אט",
                        on_click=_close_chat,
                    )
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
