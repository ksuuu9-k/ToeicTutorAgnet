"""
OpenAI text-embedding-3-small로 Question 임베딩 생성 후
Neo4j Question 노드에 저장 + Vector 인덱스 구축
"""

import json
import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "toeic_100_with_metadata.json")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
BATCH_SIZE      = 20  # OpenAI API 한 번에 처리할 문제 수


def build_embed_text(item: dict) -> str:
    """임베딩할 텍스트 구성 — 검색 품질을 높이기 위해 핵심 정보를 합침"""
    choices = item.get("choices", {})
    choices_str = " / ".join(f"{k}: {v}" for k, v in choices.items())

    key_vocab = ", ".join(item.get("key_vocab", []))
    paraphrase = ", ".join(item.get("paraphrase_hints", []))

    parts = [
        item["question"],
        f"보기: {choices_str}",
        f"유형: {item.get('sub_type', '')}",
        f"문법: {item.get('grammar_tag', '')}",
    ]
    if key_vocab:
        parts.append(f"핵심어휘: {key_vocab}")
    if paraphrase:
        parts.append(f"패러프레이즈: {paraphrase}")

    return " | ".join(parts)


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [e.embedding for e in response.data]


def create_vector_index(session):
    """Neo4j Vector 인덱스 생성 (이미 있으면 스킵)"""
    session.run("""
        CREATE VECTOR INDEX question_embedding IF NOT EXISTS
        FOR (q:Question) ON (q.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: $dim,
                `vector.similarity_function`: 'cosine'
            }
        }
    """, dim=EMBEDDING_DIM)


def save_embeddings(session, batch: list[dict]):
    """Question 노드에 embedding 속성 저장"""
    session.run("""
        UNWIND $batch AS item
        MATCH (q:Question {question_id: item.question_id})
        SET q.embedding = item.embedding
    """, batch=batch)


def main():
    print("데이터 로딩...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = list(raw.values())
    print(f"  총 {len(items)}개 문제")

    client = OpenAI(api_key=OPENAI_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Vector 인덱스 생성
        with driver.session() as session:
            print("\nVector 인덱스 생성 중...")
            create_vector_index(session)
            print("  완료")

        # 배치 단위로 임베딩 생성 및 저장
        print(f"\n임베딩 생성 중 (배치 크기: {BATCH_SIZE})...")
        total = len(items)

        for start in range(0, total, BATCH_SIZE):
            batch_items = items[start:start + BATCH_SIZE]
            texts = [build_embed_text(item) for item in batch_items]

            embeddings = get_embeddings(client, texts)

            save_batch = [
                {"question_id": item["question_id"], "embedding": emb}
                for item, emb in zip(batch_items, embeddings)
            ]

            with driver.session() as session:
                save_embeddings(session, save_batch)

            end = min(start + BATCH_SIZE, total)
            print(f"  [{end}/{total}] 완료")

            if end < total:
                time.sleep(0.5)  # API rate limit 여유

        # 결과 검증
        print("\n결과 확인:")
        with driver.session() as session:
            count = session.run("""
                MATCH (q:Question)
                WHERE q.embedding IS NOT NULL
                RETURN count(q) AS cnt
            """).single()["cnt"]
            print(f"  임베딩 저장된 Question: {count}개")

            # Vector 인덱스 상태 확인
            indexes = session.run("SHOW INDEXES YIELD name, type, state WHERE type = 'VECTOR'").data()
            for idx in indexes:
                print(f"  Vector Index [{idx['name']}] — {idx['state']}")

    finally:
        driver.close()

    print("\n임베딩 구축 완료!")


if __name__ == "__main__":
    main()
