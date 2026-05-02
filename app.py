"""
Streamlit UI — 토익 튜터 에이전트
"""

import os
import streamlit as st
from core.agent import TutorAgent

# Streamlit Cloud secrets → 환경변수로 주입
for key, val in st.secrets.items():
    os.environ.setdefault(key, val)

st.set_page_config(page_title="토익 튜터 AI", page_icon="📚", layout="centered")

st.title("📚 토익 튜터 AI")
st.caption("Part 5 문법/어휘 문제를 풀며 튜터의 도움을 받아보세요.")

# ── 세션 상태 초기화 ──────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = TutorAgent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 토익 Part 5 튜터입니다.\n\n**'문제 줘'** 라고 입력하면 문제를 출제해드릴게요."}
    ]

# ── 사이드바 — 학습 현황 ──────────────────────────────────────

with st.sidebar:
    st.header("📊 학습 현황")

    db = st.session_state.agent._db
    current = st.session_state.agent._current

    if current:
        st.subheader("현재 문제")
        st.write(f"**ID**: {current.question_id}")
        st.write(f"**유형**: {current.sub_type}")
        st.write(f"**문법**: {current.grammar_tag}")
        st.write(f"**난이도**: {'⭐' * current.difficulty}")

        attempts = db.get_attempts(current.question_id)
        if attempts:
            st.write(f"**시도 횟수**: {len(attempts)}")
            st.write(f"**현재 상태**: {attempts[-1].state}")

    st.divider()

    weak = db.weak_grammar_tags(top_n=5)
    if weak:
        st.subheader("취약 문제")
        for w in weak:
            st.write(f"- {w['question_id']} (오답 {w['wrong']}회)")

    if st.button("대화 초기화"):
        st.session_state.messages = [
            {"role": "assistant", "content": "대화가 초기화됐습니다. **'문제 줘'** 로 시작하세요."}
        ]
        st.session_state.agent._current = None
        st.rerun()

# ── 채팅 히스토리 출력 ────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 입력 처리 ─────────────────────────────────────────────────

if user_input := st.chat_input("답변 입력 (예: 1, 2, 3, 4) 또는 '문제 줘'"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("튜터가 응답 중..."):
            response = st.session_state.agent.chat(user_input)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
