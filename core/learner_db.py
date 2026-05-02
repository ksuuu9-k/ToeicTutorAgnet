"""
학습 이력 DB — SQLite

테이블: learner_history
용도 : 시도 횟수/오답 유형 추적 → State Machine 상태 결정에 사용
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime

DB_PATH = "learner.db"

DDL = """
CREATE TABLE IF NOT EXISTS learner_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT    NOT NULL,
    attempt     INTEGER NOT NULL,
    chosen      TEXT,
    is_correct  INTEGER NOT NULL,
    error_type  TEXT,
    state       TEXT    NOT NULL,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class AttemptRecord:
    id: int
    question_id: str
    attempt: int
    chosen: str | None
    is_correct: bool
    error_type: str | None
    state: str
    timestamp: str


class LearnerDB:
    def __init__(self, db_path: str = DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(DDL)
        self._conn.commit()

    def record(
        self,
        question_id: str,
        attempt: int,
        chosen: str | None,
        is_correct: bool,
        state: str,
        error_type: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO learner_history
               (question_id, attempt, chosen, is_correct, error_type, state)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (question_id, attempt, chosen, int(is_correct), error_type, state),
        )
        self._conn.commit()

    def get_attempts(self, question_id: str) -> list[AttemptRecord]:
        rows = self._conn.execute(
            "SELECT * FROM learner_history WHERE question_id = ? ORDER BY attempt ASC",
            (question_id,),
        ).fetchall()
        return [
            AttemptRecord(
                id=r["id"],
                question_id=r["question_id"],
                attempt=r["attempt"],
                chosen=r["chosen"],
                is_correct=bool(r["is_correct"]),
                error_type=r["error_type"],
                state=r["state"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    def attempt_count(self, question_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM learner_history WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        return row["cnt"]

    def is_solved(self, question_id: str) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM learner_history WHERE question_id = ? AND is_correct = 1",
            (question_id,),
        ).fetchone()
        return row["cnt"] > 0

    def dominant_error_type(self, question_id: str) -> str | None:
        """해당 문제에서 가장 많이 나온 오답 유형"""
        row = self._conn.execute(
            """SELECT error_type, COUNT(*) AS cnt
               FROM learner_history
               WHERE question_id = ? AND error_type IS NOT NULL
               GROUP BY error_type ORDER BY cnt DESC LIMIT 1""",
            (question_id,),
        ).fetchone()
        return row["error_type"] if row else None

    def weak_grammar_tags(self, top_n: int = 5) -> list[dict]:
        """전체 학습 이력 기준 오답률 높은 grammar_tag 반환 (Tutor Agent 문제 추천용)"""
        rows = self._conn.execute(
            """SELECT h.question_id,
                      COUNT(*) AS total,
                      SUM(CASE WHEN h.is_correct = 0 THEN 1 ELSE 0 END) AS wrong
               FROM learner_history h
               GROUP BY h.question_id
               HAVING wrong > 0
               ORDER BY wrong DESC
               LIMIT ?""",
            (top_n,),
        ).fetchall()
        return [{"question_id": r["question_id"], "wrong": r["wrong"]} for r in rows]

    def close(self):
        self._conn.close()
