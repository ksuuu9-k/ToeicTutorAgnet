"""
하이브리드 리트리버 — Vector + Cypher

VectorRetriever  : 질문 텍스트 임베딩 유사도로 관련 문제 검색
CypherRetriever  : 문법/난이도/유형 조건 기반 정확 탐색
HybridRetriever  : Cypher로 후보 필터 → Vector로 유사도 재정렬
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class QuestionResult:
    question_id: str
    question: str
    choices: dict
    answer: str
    answer_num: str
    sub_type: str
    grammar_tag: str
    difficulty: int
    distractor_pattern: str
    explanation: dict
    score: float = 0.0  # Vector 유사도 점수 (Cypher 전용 시 0)

    def to_dict(self) -> dict:
        return {
            "question_id":        self.question_id,
            "question":           self.question,
            "choices":            self.choices,
            "answer":             self.answer,
            "answer_num":         self.answer_num,
            "sub_type":           self.sub_type,
            "grammar_tag":        self.grammar_tag,
            "difficulty":         self.difficulty,
            "distractor_pattern": self.distractor_pattern,
            "explanation":        self.explanation,
            "score":              self.score,
        }


def _parse_node(node, score: float = 0.0) -> QuestionResult:
    import json
    return QuestionResult(
        question_id=node["question_id"],
        question=node["question"],
        choices=json.loads(node.get("choices", "{}")),
        answer=node["answer"],
        answer_num=node["answer_num"],
        sub_type=node.get("sub_type", ""),
        grammar_tag=node.get("grammar_tag", ""),
        difficulty=node.get("difficulty", 3),
        distractor_pattern=node.get("distractor_pattern", ""),
        explanation=json.loads(node.get("explanation", "{}")),
        score=score,
    )


class VectorRetriever:
    """임베딩 유사도 기반 문제 검색"""

    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self._openai = OpenAI(api_key=OPENAI_API_KEY)

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return resp.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> list[QuestionResult]:
        embedding = self._embed(query)
        with self._driver.session() as session:
            rows = session.run("""
                MATCH (node:Question)
                WITH node, vector.similarity.cosine(node.embedding, $embedding) AS score
                RETURN node, score
                ORDER BY score DESC
                LIMIT $top_k
            """, top_k=top_k, embedding=embedding).data()
        return [_parse_node(r["node"], r["score"]) for r in rows]

    def close(self):
        self._driver.close()


class CypherRetriever:
    """조건 기반 정확 탐색"""

    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def search(
        self,
        grammar_tag: str | None = None,
        sub_type: str | None = None,
        max_difficulty: int | None = None,
        exclude_ids: list[str] | None = None,
        limit: int = 5,
    ) -> list[QuestionResult]:
        """
        grammar_tag     : 예) "시제", "품사-소유격"
        sub_type        : "문법형" | "어휘의미형" | "연어형"
        max_difficulty  : 1~5
        exclude_ids     : 이미 푼 문제 question_id 목록
        """
        filters = []
        params: dict = {"limit": limit}

        if grammar_tag:
            filters.append("q.grammar_tag = $grammar_tag")
            params["grammar_tag"] = grammar_tag

        if sub_type:
            filters.append("q.sub_type = $sub_type")
            params["sub_type"] = sub_type

        if max_difficulty is not None:
            filters.append("q.difficulty <= $max_difficulty")
            params["max_difficulty"] = max_difficulty

        if exclude_ids:
            filters.append("NOT q.question_id IN $exclude_ids")
            params["exclude_ids"] = exclude_ids

        where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

        query = f"""
            MATCH (q:Question)
            {where_clause}
            RETURN q
            ORDER BY q.difficulty ASC
            LIMIT $limit
        """
        with self._driver.session() as session:
            rows = session.run(query, **params).data()
        return [_parse_node(r["q"]) for r in rows]

    def get_by_grammar(self, grammar_id: str, limit: int = 5) -> list[QuestionResult]:
        """GrammarRule 노드 경유 — related_grammar_id 기준 탐색"""
        with self._driver.session() as session:
            rows = session.run("""
                MATCH (q:Question)-[:TESTS]->(g:GrammarRule {grammar_id: $grammar_id})
                RETURN q
                ORDER BY q.difficulty ASC
                LIMIT $limit
            """, grammar_id=grammar_id, limit=limit).data()
        return [_parse_node(r["q"]) for r in rows]

    def close(self):
        self._driver.close()


class HybridRetriever:
    """
    Cypher로 후보군 필터링 → Vector 유사도로 재정렬

    사용 예)
    retriever = HybridRetriever()
    results = retriever.search(
        query="수동태 문제 풀고 싶어",
        grammar_tag="수동태",
        max_difficulty=3,
        top_k=5,
    )
    """

    def __init__(self):
        self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self._openai = OpenAI(api_key=OPENAI_API_KEY)

    def _embed(self, text: str) -> list[float]:
        resp = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return resp.data[0].embedding

    def search(
        self,
        query: str,
        grammar_tag: str | None = None,
        sub_type: str | None = None,
        max_difficulty: int | None = None,
        exclude_ids: list[str] | None = None,
        top_k: int = 5,
        cypher_pool: int = 20,  # Cypher로 뽑을 후보 수
    ) -> list[QuestionResult]:
        embedding = self._embed(query)

        filters = []
        params: dict = {"limit": cypher_pool}

        if grammar_tag:
            filters.append("q.grammar_tag = $grammar_tag")
            params["grammar_tag"] = grammar_tag

        if sub_type:
            filters.append("q.sub_type = $sub_type")
            params["sub_type"] = sub_type

        if max_difficulty is not None:
            filters.append("q.difficulty <= $max_difficulty")
            params["max_difficulty"] = max_difficulty

        if exclude_ids:
            filters.append("NOT q.question_id IN $exclude_ids")
            params["exclude_ids"] = exclude_ids

        where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

        # Cypher 후보 → 해당 노드들 중 Vector 유사도 계산
        cypher = f"""
            MATCH (q:Question)
            {where_clause}
            WITH q LIMIT $limit
            WITH q, vector.similarity.cosine(q.embedding, $embedding) AS score
            RETURN q, score
            ORDER BY score DESC
            LIMIT $top_k
        """
        params["embedding"] = embedding
        params["top_k"] = top_k

        with self._driver.session() as session:
            rows = session.run(cypher, **params).data()

        return [_parse_node(r["q"], r["score"]) for r in rows]

    def close(self):
        self._driver.close()


# ── 간단 테스트 ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vector Retriever 테스트 ===")
    vr = VectorRetriever()
    results = vr.search("수동태 관련 문제", top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.question_id} | {r.grammar_tag} | 난이도 {r.difficulty}")
        print(f"         {r.question[:60]}...")
    vr.close()

    print("\n=== Cypher Retriever 테스트 ===")
    cr = CypherRetriever()
    results = cr.search(grammar_tag="시제", max_difficulty=3, limit=3)
    for r in results:
        print(f"  {r.question_id} | {r.grammar_tag} | 난이도 {r.difficulty}")
        print(f"         {r.question[:60]}...")
    cr.close()

    print("\n=== Hybrid Retriever 테스트 ===")
    hr = HybridRetriever()
    results = hr.search("동사 시제 문제", grammar_tag="시제", max_difficulty=3, top_k=3)
    for r in results:
        print(f"  [{r.score:.3f}] {r.question_id} | {r.grammar_tag} | 난이도 {r.difficulty}")
        print(f"         {r.question[:60]}...")
    hr.close()
