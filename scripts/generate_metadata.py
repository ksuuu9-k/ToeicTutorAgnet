"""
토익 100개 문제 메타데이터 자동 생성 스크립트
================================================
사용법:
  1. pip install anthropic python-dotenv
  2. .env 파일 생성 후 API 키 입력 (아래 참고)
  3. python generate_metadata.py

.env 파일 예시:
  ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx

- 중간에 중단해도 checkpoint.json 기반으로 이어서 실행 가능
- 완료 파일: toeic_100_with_metadata.json
"""

import json
import time
import os
import sys
import anthropic
from dotenv import load_dotenv

# .env 로드 (파일 없어도 오류 없이 진행, 환경변수 직접 설정도 허용)
load_dotenv()

# API 키 검증
if not os.getenv("ANTHROPIC_API_KEY"):
    print("[오류] ANTHROPIC_API_KEY가 설정되지 않았습니다.")
    print("       .env 파일에 ANTHROPIC_API_KEY=sk-ant-xxx 형식으로 추가하세요.")
    sys.exit(1)

# ── 설정 ──────────────────────────────────────────────────────
INPUT_FILE  = "toeic_100.json"        # 100개 추출 파일
OUTPUT_FILE = "toeic_100_with_metadata.json"
CHECKPOINT  = "checkpoint.json"
SLEEP_SEC   = 0.5                     # API 호출 간격
MODEL       = "claude-sonnet-4-5"
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 토익 전문 분석가입니다.
토익 Part 5 문제를 받으면 아래 JSON 형식으로만 응답하세요.
다른 텍스트, 설명, 마크다운 코드블록 없이 순수 JSON만 반환하세요.

반환 형식:
{
  "sub_type": "문법형 | 어휘의미형 | 연어형 중 하나",
  "grammar_tag": "해당 문법 개념 (예: 시제, 품사-명사, 품사-부사, 대명사-재귀, 전치사, 접속사, 관계사 등. 어휘형이면 null)",
  "difficulty": 1~5 사이 정수,
  "distractor_pattern": "오답 함정 패턴 한 줄 설명 (한국어)",
  "explanation": {
    "correct": "정답인 이유 (한국어, 1~2문장)",
    "wrong_1": "1번 보기가 틀린 이유 (한국어, 1문장)",
    "wrong_2": "2번 보기가 틀린 이유 (한국어, 1문장)",
    "wrong_3": "3번 보기가 틀린 이유 (한국어, 1문장)",
    "wrong_4": "4번 보기가 틀린 이유 (한국어, 1문장)"
  },
  "key_vocab": ["문제에서 중요한 어휘 1~3개 (영어)"],
  "paraphrase_hints": ["정답 단어의 유사 표현 1~2개 (영어). 없으면 빈 배열 []"],
  "related_grammar_id": "G-{grammar_tag} 형식 문자열. 어휘형이면 null"
}

sub_type 판단 기준:
- 문법형: 보기가 같은 단어의 다른 형태 (시제/품사/격 변화)
- 어휘의미형: 보기가 완전히 다른 단어, 의미로 선택
- 연어형: 보기가 비슷한 의미지만 특정 단어와 어울리는 조합으로 선택

difficulty 기준:
1: 매우 쉬움 (기초 문법, 빈출 어휘)
2: 쉬움
3: 보통
4: 어려움 (고급 어휘, 복잡한 문법)
5: 매우 어려움 (킬러 문제, 연어 고난도)"""


def build_prompt(q: dict) -> str:
    return (
        f"문제: {q['question']}\n\n"
        f"보기:\n"
        f"1. {q['1']}\n"
        f"2. {q['2']}\n"
        f"3. {q['3']}\n"
        f"4. {q['4']}\n\n"
        f"정답: {q['anwser']}"
    )


def safe_parse(raw: str) -> dict:
    """마크다운 코드블록 제거 후 JSON 파싱"""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                return json.loads(p)
    return json.loads(raw)


def get_answer_num(q: dict) -> str | None:
    for i in ["1", "2", "3", "4"]:
        if q[i] == q["anwser"]:
            return i
    return None


def build_item(qid: str, original: dict, meta: dict) -> dict:
    """원본 문제 + 생성된 메타데이터 합치기"""
    answer_num = get_answer_num(original)
    exp = meta.get("explanation", {})

    # wrong_1~4를 실제 보기 번호에 매핑
    choice_exp = {}
    wrong_idx = 1
    for i in ["1", "2", "3", "4"]:
        if i == answer_num:
            choice_exp[i] = exp.get("correct", "")
        else:
            choice_exp[i] = exp.get(f"wrong_{wrong_idx}", "")
            wrong_idx += 1

    return {
        "question_id": f"P5-{int(qid):04d}",
        "part": 5,
        "question": original["question"],
        "choices": {
            "1": original["1"],
            "2": original["2"],
            "3": original["3"],
            "4": original["4"],
        },
        "answer": original["anwser"],
        "answer_num": answer_num,
        # ── 메타데이터 ──────────────────────────
        "sub_type":          meta.get("sub_type", ""),
        "grammar_tag":       meta.get("grammar_tag"),
        "difficulty":        meta.get("difficulty", 3),
        "distractor_pattern": meta.get("distractor_pattern", ""),
        "explanation":       choice_exp,
        "key_vocab":         meta.get("key_vocab", []),
        "paraphrase_hints":  meta.get("paraphrase_hints", []),
        "related_grammar_id": meta.get("related_grammar_id"),
    }


def main():
    # ── 100개 파일 없으면 원본에서 자동 추출 ──────────────
    if not os.path.exists(INPUT_FILE):
        src = "toeic_test.json"
        if not os.path.exists(src):
            print(f"[오류] {src} 파일을 찾을 수 없습니다.")
            return
        with open(src, encoding="utf-8") as f:
            full = json.load(f)
        sample = {k: full[k] for k in list(full.keys())[:100]}
        with open(INPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"[준비] {INPUT_FILE} 자동 생성 완료 (100개)")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # ── 체크포인트 로드 ────────────────────────────────────
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[재개] 체크포인트 로드: {len(results)}개 완료, {len(data)-len(results)}개 남음")
    else:
        results = {}

    client = anthropic.Anthropic()
    total  = len(data)
    errors = []

    for qid, q in data.items():
        if qid in results:
            continue

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(q)}],
            )
            meta = safe_parse(resp.content[0].text)
            results[qid] = build_item(qid, q, meta)

            done = len(results)
            item = results[qid]
            print(
                f"[{done:3d}/{total}] #{qid:>3} | "
                f"{item['sub_type']:<8} | "
                f"난이도 {item['difficulty']} | "
                f"정답: {item['answer']}"
            )

            # 10개마다 체크포인트 저장
            if done % 10 == 0:
                with open(CHECKPOINT, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"  💾 체크포인트 저장 ({done}개)")

            time.sleep(SLEEP_SEC)

        except Exception as e:
            print(f"  ✗ #{qid} 오류: {e}")
            errors.append(qid)

    # ── 최종 저장 ──────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 체크포인트 정리
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\n✅ 완료: {len(results)}개 → {OUTPUT_FILE}")
    if errors:
        print(f"⚠️  오류 문제 ID: {errors}")
        print("   → 다시 실행하면 체크포인트 기반으로 해당 문제만 재처리됩니다.")


if __name__ == "__main__":
    main()