"""
Test script for all FastAPI endpoints
Make sure the server is running before executing:
    uvicorn app.main:app --reload
Then run:
    python app/test_api.py
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
DIVIDER = "=" * 60


def print_section(title):
    print(f"\n{DIVIDER}\n{title}\n{DIVIDER}")


def test_root():
    print_section("ROOT ENDPOINT")
    r = requests.get(f"{BASE_URL}/")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))


def test_health():
    print_section("HEALTH CHECK")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))


def test_ask():
    print_section("SINGLE QUESTION (/ask)")
    test_questions = [
        "How do I reset my password?",
        "I want to delete my account",
        "What is your refund policy?",
        "How to contact support?",
        "How much does premium cost?",
        "Is my data secure?",
        "Can I cancel my subscription?",
    ]

    for question in test_questions:
        payload = {"question": question}
        start = time.time()
        r = requests.post(f"{BASE_URL}/ask", json=payload)
        elapsed = (time.time() - start) * 1000

        print(f"\nQ: {question}")
        print(f"   Status: {r.status_code} ({elapsed:.0f}ms)")
        if r.status_code == 200:
            data = r.json()
            print(f"   Intent:     {data['intent']} (conf: {data['confidence']:.3f})")
            print(f"   Answer:     {data['answer'][:80]}...")
        else:
            print(f"   Error: {r.text}")


def test_batch_ask():
    print_section("BATCH QUESTIONS (/batch-ask)")
    payload = {
        "questions": [
            "How do I reset my password?",
            "I want to delete my account",
            "What is your refund policy?",
            "How to contact support?"
        ]
    }
    r = requests.post(f"{BASE_URL}/batch-ask", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Processed: {data['total_processed']} | Total time: {data['total_time_ms']:.1f}ms")
        for i, result in enumerate(data['results'], 1):
            print(f"\n  {i}. Q: {result['question']}")
            print(f"     A: {result['answer'][:60]}...")
            print(f"     Intent: {result['intent']} (conf: {result['confidence']:.3f})")
    else:
        print(f"Error: {r.text}")


def test_model_info():
    print_section("MODEL INFO (/model-info)")
    r = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Type:         {data['model_type']}")
        print(f"  Input dim:    {data['input_dim']}")
        print(f"  Hidden layers:{data['hidden_layers']}")
        print(f"  Classes:      {data['num_classes']}")
        print(f"  Parameters:   {data['total_parameters']:,}")
        print(f"  Device:       {data['device']}")


def test_intents():
    print_section("AVAILABLE INTENTS (/intents)")
    r = requests.get(f"{BASE_URL}/intents")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Total: {data['count']}")
        for intent in data['intents']:
            print(f"  - {intent}")


def test_faq_stats():
    print_section("FAQ STATISTICS (/faq-stats)")
    r = requests.get(f"{BASE_URL}/faq-stats")
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Total FAQs:    {data['total_faqs']}")
        print(f"Total intents: {data['total_intents']}")
        print("Distribution:")
        for intent, count in data['intent_distribution'].items():
            print(f"  {intent}: {count}")


def test_feedback():
    print_section("FEEDBACK (/feedback)")
    payload = {
        "question": "How do I reset my password?",
        "predicted_intent": "password_reset",
        "correct_intent": "password_reset",
        "confidence": 0.95,
        "comment": "Correct prediction, great!"
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        print(json.dumps(r.json(), indent=2))


def test_edge_cases():
    print_section("EDGE CASES")

    # Very short question
    r = requests.post(f"{BASE_URL}/ask", json={"question": "help"})
    print(f"Short query 'help' -> status {r.status_code} | intent: {r.json().get('intent', 'N/A')}")

    # Long question
    long_q = "I have been a loyal customer for many years and I am wondering if it is possible for me to get a full refund for my recent purchase because I was not satisfied with the product"
    r = requests.post(f"{BASE_URL}/ask", json={"question": long_q})
    print(f"Long query   -> status {r.status_code} | intent: {r.json().get('intent', 'N/A')}")

    # Empty question (should fail validation)
    r = requests.post(f"{BASE_URL}/ask", json={"question": "   "})
    print(f"Empty query  -> status {r.status_code} (expected 422)")


def run_all_tests():
    print(f"\n{'*' * 60}")
    print("  RUNNING ALL FASTAPI ENDPOINT TESTS")
    print(f"{'*' * 60}")

    tests = [
        test_root,
        test_health,
        test_model_info,
        test_intents,
        test_faq_stats,
        test_ask,
        test_batch_ask,
        test_feedback,
        test_edge_cases,
    ]

    for test in tests:
        try:
            test()
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to {BASE_URL}.")
            print("   Start the server first: uvicorn app.main:app --reload")
            break
        except Exception as e:
            print(f"\n❌ Error in {test.__name__}: {e}")

    print(f"\n{'*' * 60}\n  ALL TESTS COMPLETE\n{'*' * 60}\n")


if __name__ == "__main__":
    run_all_tests()
