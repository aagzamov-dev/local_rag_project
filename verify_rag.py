import sys
import os

# Add the api directory to python path
sys.path.append(os.path.join(os.getcwd(), "api"))

from rag_core import finalize_answer, FALLBACK


def test_fallback():
    print("Testing Fallback...")
    # Mock retrieved items
    retrieved = [
        {
            "metadata": {"source": "test_doc.txt", "chunk_id": "1"},
            "id": "1",
            "content": "foo",
        }
    ]

    # Case 1: Exact mismatch
    # If the model says exactly fallback
    res = finalize_answer(FALLBACK, retrieved)
    assert res == FALLBACK, f"Expected fallback, got {res}"

    # Case 2: Case insensitive with extra whitespace
    res = finalize_answer(f"  {FALLBACK.lower()}  ", retrieved)
    assert res == FALLBACK, f"Expected fallback, got {res}"

    # Case 3: Embedded fallback
    res = finalize_answer(f"I tried but {FALLBACK} sorry", retrieved)
    assert res == FALLBACK, f"Expected fallback, got {res}"

    print("Fallback tests passed!")


def test_success():
    print("Testing Success...")
    retrieved = [
        {
            "metadata": {"source": "doc1.txt", "chunk_id": "1"},
            "id": "1",
            "content": "foo",
        },
        {
            "metadata": {"source": "doc2.txt", "chunk_id": "2"},
            "id": "2",
            "content": "bar",
        },
        {
            "metadata": {"source": "doc1.txt", "chunk_id": "1"},
            "id": "1",
            "content": "foo",
        },  # duplicate
    ]

    # Case 1: Normal answer with brackets
    model_output = "The answer is 42 [doc1.txt#1]. "
    res = finalize_answer(model_output, retrieved)

    expected_text = "The answer is 42."
    expected_sources = "Sources: [doc1.txt#1], [doc2.txt#2]"

    assert expected_text in res, f"Expected text '{expected_text}' not in '{res}'"
    assert expected_sources in res, (
        f"Expected sources '{expected_sources}' not in '{res}'"
    )

    # deduplication check
    assert res.count("[doc1.txt#1]") == 1, "Duplicate source found in output"

    print("Success tests passed!")


if __name__ == "__main__":
    test_fallback()
    test_success()
