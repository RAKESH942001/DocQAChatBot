import streamlit as st
import requests
from datetime import datetime
import time

BACKEND_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="Document Q&A Chat", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "upload_time" not in st.session_state:
    st.session_state.upload_time = None

def time_ago(t):
    diff = datetime.now() - t
    seconds = int(diff.total_seconds())
    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        return f"{seconds // 60} min ago"
    elif seconds < 86400:
        return f"{seconds // 3600} hr ago"
    return t.strftime('%d %b %Y %H:%M')

hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

chat_bubble_style = """
    <style>
    .chat-bubble {
        max-width: 70%;
        padding: 0.7em 1em;
        border-radius: 1.2em;
        margin-bottom: 0.3em;
        font-size: 1.1em;
        display: inline-block;
        word-break: break-word;
    }
    .left {
        background-color: #f1f0f0;
        color: #222;
        text-align: left;
        margin-right: auto;
        margin-left: 0;
    }
    .right {
        background-color: #0084ff;
        color: #fff;
        text-align: right;
        margin-left: auto;
        margin-right: 0;
    }
    </style>
"""
st.markdown(chat_bubble_style, unsafe_allow_html=True)

st.markdown("## Document Q&A Chat")
st.markdown("#### Upload a PDF with the + button and start chatting")

with st.expander("Attach PDF"):
    new_pdf = st.file_uploader("Upload a new PDF", type=["pdf"])
    if new_pdf and (st.session_state.current_pdf is None or new_pdf.name != st.session_state.current_pdf.name):
        st.session_state.current_pdf = new_pdf
        st.session_state.upload_time = datetime.now()
        st.session_state.messages.append({
            "role": "user",
            "type": "file",
            "file_name": new_pdf.name,
            "time": st.session_state.upload_time
        })

if st.session_state.current_pdf:
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        st.session_state.current_pdf.seek(0)
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_input,
            "time": datetime.now()
        })
        start_time = time.time()
        with st.spinner("Thinking..."):
            response = requests.post(
                BACKEND_URL,
                files={"pdf": st.session_state.current_pdf},
                data={"question": user_input}
            )
        elapsed = time.time() - start_time
        print(f"Backend request took {elapsed:.2f} seconds.")
        if response.status_code == 200:
            result = response.json()
            bot_reply = result.get("answer", "I couldn't find an answer.")
            print(f"Answer from backend: {bot_reply}")
            st.session_state.messages.append({
                "role": "bot",
                "type": "text",
                "content": bot_reply,
                "time": datetime.now()
            })
        else:
            st.error(f"Server error: {response.status_code}")

for msg in st.session_state.messages:
    align = "right" if msg["role"] == "user" else "left"
    align_text = "flex-end" if align == "right" else "flex-start"
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: {align_text}; margin-bottom: 0.5em;'>
            <div class='chat-bubble {align}'>{'<b>' + msg['file_name'] + ' uploaded</b>' if msg['type']=='file' else msg['content']}</div>
            <div style='font-size:0.8em; color: #888; margin-top: 2px; text-align: {align}; width: max-content; min-width: 60px;'>{time_ago(msg['time'])}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.markdown('### Debug: Message Roles')
    for i, msg in enumerate(st.session_state.messages):
        st.write(f"Message {i+1}: role = {msg['role']}")