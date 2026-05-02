"""
toeic_100_with_metadata.json → Neo4j Aura 적재 스크립트

노드: Question, GrammarRule, Vocab, Paraphrase
엣지: TESTS, CONTAINS_VOCAB, HAS_PARAPHRASE, SIMILAR_TO
"""

import json
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "toeic_100_with_metadata.json")


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 노드 생성 ────────────────────────────────────────────────

def create_questions(tx, items):
    query = """
    UNWIND $items AS item
    MERGE (q:Question {question_id: item.question_id})
    SET q.part              = item.part,
        q.question          = item.question,
        q.answer            = item.answer,
        q.answer_num        = item.answer_num,
        q.sub_type          = item.sub_type,
        q.grammar_tag       = item.grammar_tag,
        q.difficulty        = item.difficulty,
        q.distractor_pattern = item.distractor_pattern,
        q.choices           = item.choices_json,
        q.explanation       = item.explanation_json
    """
    tx.run(query, items=items)


def create_grammar_rules(tx, rules):
    query = """
    UNWIND $rules AS rule
    MERGE (g:GrammarRule {grammar_id: rule.grammar_id})
    SET g.name = rule.name
    """
    tx.run(query, rules=rules)


def create_vocabs(tx, vocabs):
    query = """
    UNWIND $vocabs AS v
    MERGE (:Vocab {word: v})
    """
    tx.run(query, vocabs=vocabs)


def create_paraphrases(tx, paraphrases):
    query = """
    UNWIND $paraphrases AS p
    MERGE (:Paraphrase {word: p})
    """
    tx.run(query, paraphrases=paraphrases)


# ── 엣지 생성 ────────────────────────────────────────────────

def create_tests_edges(tx, links):
    """Question -[:TESTS]-> GrammarRule"""
    query = """
    UNWIND $links AS link
    MATCH (q:Question {question_id: link.question_id})
    MATCH (g:GrammarRule {grammar_id: link.grammar_id})
    MERGE (q)-[:TESTS]->(g)
    """
    tx.run(query, links=links)


def create_vocab_edges(tx, links):
    """Question -[:CONTAINS_VOCAB]-> Vocab"""
    query = """
    UNWIND $links AS link
    MATCH (q:Question {question_id: link.question_id})
    MATCH (v:Vocab {word: link.word})
    MERGE (q)-[:CONTAINS_VOCAB]->(v)
    """
    tx.run(query, links=links)


def create_paraphrase_edges(tx, links):
    """Vocab -[:HAS_PARAPHRASE]-> Paraphrase
    key_vocab의 첫 번째 단어를 대표 어휘로 사용해 paraphrase_hints와 연결
    """
    query = """
    UNWIND $links AS link
    MATCH (v:Vocab {word: link.vocab})
    MATCH (p:Paraphrase {word: link.paraphrase})
    MERGE (v)-[:HAS_PARAPHRASE]->(p)
    """
    tx.run(query, links=links)


def create_similar_to_edges(tx):
    """같은 grammar_tag를 가진 Question 간 SIMILAR_TO 엣지"""
    query = """
    MATCH (q1:Question), (q2:Question)
    WHERE q1.grammar_tag = q2.grammar_tag
      AND q1.question_id < q2.question_id
    MERGE (q1)-[:SIMILAR_TO]->(q2)
    """
    tx.run(query)


# ── 데이터 변환 ──────────────────────────────────────────────

def prepare_data(raw: dict):
    questions      = []
    grammar_rules  = {}
    all_vocabs     = set()
    all_paraphrases = set()
    tests_links    = []
    vocab_links    = []
    paraphrase_links = []

    for _, item in raw.items():
        qid = item["question_id"]

        questions.append({
            "question_id":       qid,
            "part":              item["part"],
            "question":          item["question"],
            "answer":            item["answer"],
            "answer_num":        item["answer_num"],
            "sub_type":          item.get("sub_type", ""),
            "grammar_tag":       item.get("grammar_tag", ""),
            "difficulty":        item.get("difficulty", 3),
            "distractor_pattern": item.get("distractor_pattern", ""),
            "choices_json":      json.dumps(item.get("choices", {}), ensure_ascii=False),
            "explanation_json":  json.dumps(item.get("explanation", {}), ensure_ascii=False),
        })

        # GrammarRule
        gid = item.get("related_grammar_id", "")
        if gid:
            grammar_rules[gid] = item.get("grammar_tag", gid.replace("G-", ""))
            tests_links.append({"question_id": qid, "grammar_id": gid})

        # Vocab
        for word in item.get("key_vocab", []):
            all_vocabs.add(word)
            vocab_links.append({"question_id": qid, "word": word})

        # Paraphrase — key_vocab[0]을 대표 어휘로 연결
        key_vocabs = item.get("key_vocab", [])
        for phrase in item.get("paraphrase_hints", []):
            all_paraphrases.add(phrase)
            if key_vocabs:
                paraphrase_links.append({"vocab": key_vocabs[0], "paraphrase": phrase})

    grammar_list = [{"grammar_id": gid, "name": name} for gid, name in grammar_rules.items()]

    return {
        "questions":        questions,
        "grammar_rules":    grammar_list,
        "vocabs":           list(all_vocabs),
        "paraphrases":      list(all_paraphrases),
        "tests_links":      tests_links,
        "vocab_links":      vocab_links,
        "paraphrase_links": paraphrase_links,
    }


# ── 메인 ─────────────────────────────────────────────────────

def main():
    print("데이터 로딩...")
    raw  = load_data()
    data = prepare_data(raw)

    print(f"  Questions     : {len(data['questions'])}")
    print(f"  GrammarRules  : {len(data['grammar_rules'])}")
    print(f"  Vocabs        : {len(data['vocabs'])}")
    print(f"  Paraphrases   : {len(data['paraphrases'])}")

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        with driver.session() as session:
            print("\n노드 생성 중...")
            session.execute_write(create_questions,     data["questions"])
            session.execute_write(create_grammar_rules, data["grammar_rules"])
            session.execute_write(create_vocabs,        data["vocabs"])
            session.execute_write(create_paraphrases,   data["paraphrases"])
            print("  노드 생성 완료")

            print("엣지 생성 중...")
            session.execute_write(create_tests_edges,      data["tests_links"])
            session.execute_write(create_vocab_edges,      data["vocab_links"])
            session.execute_write(create_paraphrase_edges, data["paraphrase_links"])
            session.execute_write(create_similar_to_edges)
            print("  엣지 생성 완료")

            # 결과 검증
            print("\n적재 결과 확인:")
            for label in ["Question", "GrammarRule", "Vocab", "Paraphrase"]:
                count = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()["cnt"]
                print(f"  ({label}) : {count}개")

            for rel in ["TESTS", "CONTAINS_VOCAB", "HAS_PARAPHRASE", "SIMILAR_TO"]:
                count = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt").single()["cnt"]
                print(f"  [:{rel}] : {count}개")

    finally:
        driver.close()

    print("\n적재 완료!")


if __name__ == "__main__":
    main()
