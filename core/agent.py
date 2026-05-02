"""
Tutor Agent — 메인 오케스트레이터

역할:
  1. 학습자 메시지 의도 파악 (답변 제출 / 새 문제 요청 / 정답 요청)
  2. HybridRetriever로 문제 추천
  3. TutoringStateMachine으로 상태 전이 + 시스템 프롬프트 생성
  4. Claude API 호출 → 튜터 응답 반환
"""

import os
import re
from dotenv import load_dotenv
import anthropic

from core.retriever import HybridRetriever, CypherRetriever, QuestionResult
from core.learner_db import LearnerDB
from core.state_machine import TutoringStateMachine, TutoringState

load_dotenv()

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS   = 1024


def _detect_choice(text: str) -> str | None:
    """메시지에서 보기 번호(1~4) 추출"""
    text = text.strip()
    # "1번", "①", "1", "답은 2" 등 다양한 형태 처리
    m = re.search(r"[①②③④]|[1-4]번|^[1-4]$|답.{0,3}([1-4])", text)
    if not m:
        return None
    raw = m.group()
    mapping = {"①": "1", "②": "2", "③": "3", "④": "4"}
    for k, v in mapping.items():
        if k in raw:
            return v
    digit = re.search(r"[1-4]", raw)
    return digit.group() if digit else None


def _is_reveal_request(text: str) -> bool:
    keywords = ["정답", "알려줘", "모르겠", "포기", "힌트 말고", "답 줘", "답을 줘"]
    return any(k in text for k in keywords)


def _is_new_question_request(text: str) -> bool:
    keywords = ["문제", "풀고 싶", "새 문제", "다른 문제", "문제 줘", "시작"]
    return any(k in text for k in keywords)


HISTORY_LIMIT = 10  # 유지할 최대 대화 턴 수 (user+assistant 쌍 기준)


class TutorAgent:
    def __init__(self):
        self._client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._hybrid   = HybridRetriever()
        self._cypher   = CypherRetriever()
        self._db       = LearnerDB()
        self._fsm      = TutoringStateMachine(self._db)
        self._current: QuestionResult | None = None
        self._history: list[dict] = []  # Claude API용 대화 이력

    # ── 퍼블릭 API ────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """학습자 메시지를 받아 튜터 응답 반환"""

        # 1) 새 문제 요청
        if self._current is None or _is_new_question_request(user_message):
            response = self._start_new_question(user_message)
            self._history = []  # 새 문제 시작 시 이력 초기화
            self._append_history(user_message, response)
            return response

        # 2) 정답 직접 요청
        if _is_reveal_request(user_message):
            ctx = self._fsm.process(self._current, chosen=None, force_reveal=True)
            response = self._call_claude(ctx.system_prompt, user_message)
            self._append_history(user_message, response)
            return response

        # 3) 보기 번호 제출
        chosen = _detect_choice(user_message)
        if chosen:
            ctx = self._fsm.process(self._current, chosen=chosen)
            response = self._call_claude(ctx.system_prompt, user_message)
            if ctx.is_correct or ctx.state == TutoringState.REVEAL:
                response += "\n\n---\n다음 문제를 풀고 싶으면 **'문제 줘'** 라고 입력하세요."
            self._append_history(user_message, response)
            return response

        # 4) 자유 질문 — 현재 문제 컨텍스트 포함해서 응답
        response = self._call_claude(self._free_tutor_prompt(user_message), user_message)
        self._append_history(user_message, response)
        return response

    def recommend_question(self, query: str = "", grammar_tag: str | None = None,
                           max_difficulty: int | None = None) -> QuestionResult:
        """학습 이력 기반 문제 추천"""
        solved_ids = [r["question_id"] for r in self._db.weak_grammar_tags(top_n=50)]

        if query:
            results = self._hybrid.search(
                query=query,
                grammar_tag=grammar_tag,
                max_difficulty=max_difficulty,
                exclude_ids=solved_ids or None,
                top_k=1,
            )
            if results:
                return results[0]

        # fallback — Cypher로 난이도 낮은 문제부터
        results = self._cypher.search(
            grammar_tag=grammar_tag,
            max_difficulty=max_difficulty or 3,
            exclude_ids=solved_ids or None,
            limit=1,
        )
        return results[0] if results else self._cypher.search(limit=1)[0]

    # ── 내부 메서드 ───────────────────────────────────────────

    def _start_new_question(self, user_message: str) -> str:
        self._current = self.recommend_question(query=user_message)
        q = self._current

        choices_str = "\n".join(f"  {k}) {v}" for k, v in q.choices.items())
        return (
            f"**[Part 5 — {q.sub_type} / 난이도 {q.difficulty}]**\n\n"
            f"{q.question}\n\n"
            f"{choices_str}\n\n"
            f"보기 번호(1~4)를 입력하세요."
        )

    def _append_history(self, user_msg: str, assistant_msg: str) -> None:
        self._history.append({"role": "user",      "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})
        # 오래된 이력 제거 (HISTORY_LIMIT 쌍 유지)
        if len(self._history) > HISTORY_LIMIT * 2:
            self._history = self._history[-(HISTORY_LIMIT * 2):]

    def _call_claude(self, system_prompt: str, current_user_msg: str) -> str:
        # 이전 대화 이력 + 현재 메시지 조합
        messages = self._history + [{"role": "user", "content": current_user_msg}]
        message = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=messages,
        )
        return message.content[0].text

    def _free_tutor_prompt(self, user_message: str) -> str:
        q = self._current
        if q:
            choices_str = "\n".join(f"  {k}) {v}" for k, v in q.choices.items())
            question_ctx = f"""
현재 학습자가 풀고 있는 문제:
[문제] {q.question}
[보기]
{choices_str}
[문법 포인트] {q.grammar_tag} / [유형] {q.sub_type}
"""
        else:
            question_ctx = "현재 진행 중인 문제 없음."

        return f"""당신은 토익 전문 튜터입니다.
{question_ctx}
지시사항:
- 위 문제 맥락을 바탕으로 학습자의 질문에 친절하게 답변하세요.
- 정답은 직접 알려주지 마세요. 힌트나 설명으로 유도하세요.
- 답변은 3~5문장 이내로 간결하게 하세요.
"""

    def close(self):
        self._hybrid.close()
        self._cypher.close()
        self._db.close()


# ── 간단 테스트 ───────────────────────────────────────────────

if __name__ == "__main__":
    agent = TutorAgent()

    print("=== Tutor Agent 테스트 ===\n")

    # 1) 문제 요청
    print("[학습자] 문제 줘")
    resp = agent.chat("문제 줘")
    print(f"[튜터]\n{resp}\n")

    # 2) 오답 제출
    q = agent._current
    wrong = "1" if q.answer_num != "1" else "2"
    print(f"[학습자] {wrong}")
    resp = agent.chat(wrong)
    print(f"[튜터]\n{resp}\n")

    # 3) 정답 요청
    print("[학습자] 모르겠어, 정답 알려줘")
    resp = agent.chat("모르겠어, 정답 알려줘")
    print(f"[튜터]\n{resp}\n")

    agent.close()
