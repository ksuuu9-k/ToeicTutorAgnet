"""
Tutoring State Machine

상태 전이: PROBE → HINT → SCAFFOLD → REVEAL
각 상태에서 Claude에게 넘길 system prompt 조각을 생성
"""

from enum import Enum
from dataclasses import dataclass
from core.learner_db import LearnerDB
from core.retriever import QuestionResult


class TutoringState(str, Enum):
    PROBE    = "PROBE"     # 1번째 틀림 — 탐색 질문
    HINT     = "HINT"      # 2번째 틀림 — 핵심 힌트
    SCAFFOLD = "SCAFFOLD"  # 3번째 틀림 — 단계별 유도
    REVEAL   = "REVEAL"    # 4번+ 또는 직접 요청 — 정답 공개


ERROR_TYPES = [
    "어휘_부재",
    "패러프레이징_미인식",
    "문법_미숙지",
    "함정_낚임",
    "독해_속도",
]


@dataclass
class TutoringContext:
    question: QuestionResult
    state: TutoringState
    attempt: int           # 이번 시도 번호
    chosen: str | None     # 학습자가 고른 보기 번호
    is_correct: bool
    error_type: str | None
    system_prompt: str     # Claude에게 전달할 튜터 지시문


class TutoringStateMachine:
    def __init__(self, db: LearnerDB):
        self._db = db

    # ── 상태 결정 ──────────────────────────────────────────────

    def current_state(self, question_id: str) -> TutoringState:
        """DB 기준 현재 상태 반환 (아직 시도 없으면 PROBE)"""
        count = self._db.attempt_count(question_id)
        return self._state_from_count(count)

    def _state_from_count(self, wrong_count: int) -> TutoringState:
        if wrong_count == 0:
            return TutoringState.PROBE
        elif wrong_count == 1:
            return TutoringState.HINT
        elif wrong_count == 2:
            return TutoringState.SCAFFOLD
        else:
            return TutoringState.REVEAL

    # ── 메인 진입점 ────────────────────────────────────────────

    def process(
        self,
        question: QuestionResult,
        chosen: str | None,
        force_reveal: bool = False,
    ) -> TutoringContext:
        """
        학습자의 답변을 받아 상태를 전이하고 TutoringContext 반환

        chosen       : 학습자가 선택한 보기 번호 문자열 ("1"~"4")
        force_reveal : 학습자가 "정답 알려줘"라고 직접 요청한 경우
        """
        is_correct = (chosen == question.answer_num)
        prev_count = self._db.attempt_count(question.question_id)
        attempt    = prev_count + 1

        if force_reveal or is_correct:
            state = TutoringState.REVEAL
        else:
            state = self._state_from_count(prev_count)

        error_type = None
        if not is_correct and chosen:
            error_type = self._infer_error_type(question, chosen)

        self._db.record(
            question_id=question.question_id,
            attempt=attempt,
            chosen=chosen,
            is_correct=is_correct,
            state=state.value,
            error_type=error_type,
        )

        system_prompt = self._build_prompt(question, state, chosen, error_type, is_correct)

        return TutoringContext(
            question=question,
            state=state,
            attempt=attempt,
            chosen=chosen,
            is_correct=is_correct,
            error_type=error_type,
            system_prompt=system_prompt,
        )

    # ── 오답 원인 추론 ─────────────────────────────────────────

    def _infer_error_type(self, question: QuestionResult, chosen: str) -> str:
        """
        문제 메타데이터 기반 오답 원인 추론
        (실제 서비스에선 Claude로 분류 가능 — 여기선 규칙 기반 추정)
        """
        distractor = question.distractor_pattern.lower()

        if "어휘" in distractor or "단어" in distractor:
            return "어휘_부재"
        if "패러프레이즈" in distractor or "유사" in distractor:
            return "패러프레이징_미인식"
        if question.sub_type == "문법형":
            return "문법_미숙지"
        if "함정" in distractor or "낚" in distractor:
            return "함정_낚임"
        if question.sub_type == "연어형":
            return "패러프레이징_미인식"
        return "문법_미숙지"

    # ── 시스템 프롬프트 생성 ────────────────────────────────────

    def _build_prompt(
        self,
        q: QuestionResult,
        state: TutoringState,
        chosen: str | None,
        error_type: str | None,
        is_correct: bool,
    ) -> str:
        chosen_text = q.choices.get(chosen, "") if chosen else ""
        correct_text = q.choices.get(q.answer_num, "")

        base = f"""당신은 토익 전문 튜터입니다.
학습자가 다음 문제를 풀었습니다.

[문제]
{q.question}

[보기]
{self._format_choices(q.choices)}

학습자 선택: {chosen}번 — {chosen_text}
정답: {q.answer_num}번 — {correct_text}
"""

        if is_correct:
            return base + self._prompt_correct(q)

        if state == TutoringState.PROBE:
            return base + self._prompt_probe(q, error_type)
        elif state == TutoringState.HINT:
            return base + self._prompt_hint(q, error_type)
        elif state == TutoringState.SCAFFOLD:
            return base + self._prompt_scaffold(q, error_type)
        else:
            return base + self._prompt_reveal(q)

    def _format_choices(self, choices: dict) -> str:
        return "\n".join(f"  {k}) {v}" for k, v in choices.items())

    def _prompt_correct(self, q: QuestionResult) -> str:
        return f"""
학습자가 정답을 맞혔습니다.

지시사항:
- 정답임을 간단히 축하해주세요.
- 왜 정답인지 핵심 이유를 1~2문장으로 설명하세요.
- 관련 문법 포인트({q.grammar_tag})를 한 줄로 정리해주세요.
- 응원의 말로 마무리하세요.
"""

    def _prompt_probe(self, q: QuestionResult, error_type: str | None) -> str:
        grammar_hint = f"이 문제의 핵심 문법 포인트는 '{q.grammar_tag}'입니다." if q.grammar_tag else ""
        return f"""
학습자가 첫 번째 시도에서 틀렸습니다. [PROBE 단계]
추정 오답 원인: {error_type or "미분류"}

지시사항:
- 답을 절대 알려주지 마세요.
- 틀렸다는 사실을 부드럽게 전달하세요.
- 문장 구조나 핵심 단서에 주목하도록 유도하는 탐색 질문을 1개만 하세요.
- {grammar_hint}
- 예시) "이 문장에서 시간을 나타내는 표현을 찾아볼 수 있을까요?"
"""

    def _prompt_hint(self, q: QuestionResult, error_type: str | None) -> str:
        vocab_hint = f"핵심 어휘: {', '.join(q.choices.values()) if q.choices else ''}"
        error_guide = {
            "어휘_부재":           "관련 어휘의 뜻을 짧게 힌트로 주세요.",
            "패러프레이징_미인식": "비슷한 의미의 다른 표현(패러프레이즈)을 힌트로 주세요.",
            "문법_미숙지":         f"'{q.grammar_tag}' 규칙의 핵심 포인트를 한 줄 힌트로 주세요.",
            "함정_낚임":           "오답을 고르게 만드는 함정 패턴을 경고해주세요.",
            "독해_속도":           "문장에서 정답 결정에 필요한 핵심 구절을 짚어주세요.",
        }.get(error_type or "", "핵심 키워드 힌트를 주세요.")

        return f"""
학습자가 두 번째 시도에서도 틀렸습니다. [HINT 단계]
추정 오답 원인: {error_type or "미분류"}

지시사항:
- 답을 직접 알려주지 마세요.
- {error_guide}
- {vocab_hint}
- 힌트는 2~3문장 이내로 간결하게 주세요.
"""

    def _prompt_scaffold(self, q: QuestionResult, error_type: str | None) -> str:
        return f"""
학습자가 세 번째 시도에서도 틀렸습니다. [SCAFFOLD 단계]
추정 오답 원인: {error_type or "미분류"}
문법 포인트: {q.grammar_tag}

지시사항:
- 답을 직접 말하지 않되, 정답에 매우 가까이 유도하세요.
- 아래 단계로 사고를 유도하세요:
  1. 문장의 주어/동사/목적어를 확인하게 하세요.
  2. '{q.grammar_tag}' 규칙을 적용하는 방법을 단계별로 설명하세요.
  3. 각 보기를 하나씩 소거법으로 검토하도록 안내하세요.
- 마지막에 "이제 다시 한번 골라볼까요?"로 마무리하세요.
"""

    def _prompt_reveal(self, q: QuestionResult) -> str:
        explanation_text = "\n".join(
            f"  {k}번 {v}" for k, v in q.explanation.items()
        )
        return f"""
정답을 공개할 단계입니다. [REVEAL 단계]

지시사항:
- 정답({q.answer_num}번: {q.choices.get(q.answer_num, "")})을 명확히 알려주세요.
- 각 보기에 대한 설명을 제공하세요:
{explanation_text}
- '{q.grammar_tag}' 문법 포인트를 정리해주세요.
- 학습자가 다음에 비슷한 문제를 맞힐 수 있도록 핵심 전략을 한 줄로 정리하세요.
- 격려의 말로 마무리하세요.
"""


# ── 간단 테스트 ───────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from core.retriever import QuestionResult

    with open("toeic_100_with_metadata.json", encoding="utf-8") as f:
        raw = json.load(f)

    sample = list(raw.values())[0]
    question = QuestionResult(
        question_id=sample["question_id"],
        question=sample["question"],
        choices=sample["choices"],
        answer=sample["answer"],
        answer_num=sample["answer_num"],
        sub_type=sample.get("sub_type", ""),
        grammar_tag=sample.get("grammar_tag", ""),
        difficulty=sample.get("difficulty", 3),
        distractor_pattern=sample.get("distractor_pattern", ""),
        explanation=sample.get("explanation", {}),
    )

    db  = LearnerDB(":memory:")  # 테스트용 인메모리 DB
    fsm = TutoringStateMachine(db)

    print(f"문제: {question.question[:60]}...")
    print(f"정답 번호: {question.answer_num}\n")

    wrong_choice = "1" if question.answer_num != "1" else "2"

    for i, label in enumerate(["1차(오답)", "2차(오답)", "3차(오답)", "4차(오답)"]):
        ctx = fsm.process(question, chosen=wrong_choice)
        print(f"=== {label} → [{ctx.state}] ===")
        print(ctx.system_prompt[:300])
        print()

    print("=== 정답 제출 ===")
    ctx = fsm.process(question, chosen=question.answer_num)
    print(f"[{ctx.state}] 정답: {ctx.is_correct}")
    print(ctx.system_prompt[:300])

    db.close()
