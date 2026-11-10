
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.generation.answer_generator import generate_answer


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer("   ", [])


def test_generate_answer_handles_no_documents():
    answer = generate_answer(
        "Why is the server overheating?",
        [],
    )

    assert "could not find relevant evidence" in answer.lower()


@patch("src.generation.answer_generator.ChatGoogleGenerativeAI")
@patch("src.generation.answer_generator.PROMPT")
def test_generate_answer_uses_supplied_evidence(
    mock_prompt,
    mock_llm_class,
):
    expected_answer = (
        "The server may overheat if its fans fail "
        "[sample.txt, Chunk ID: 0]."
    )

    mock_llm = MagicMock()
    mock_llm_class.return_value = mock_llm

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(
        content=expected_answer
    )

    mock_prompt.__or__.return_value = mock_chain

    document = Document(
        page_content="A server may overheat when cooling fans fail.",
        metadata={
            "source": "sample.txt",
            "chunk_id": 0,
        },
    )

    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        answer = generate_answer(
            "Why is the server overheating?",
            [document],
        )

    assert "fans fail" in answer
    assert "Chunk ID: 0" in answer

    mock_llm_class.assert_called_once()
    mock_chain.invoke.assert_called_once()


@patch("src.generation.answer_generator.ChatGoogleGenerativeAI")
@patch("src.generation.answer_generator.PROMPT")
def test_generator_reports_when_evidence_is_insufficient(
    mock_prompt,
    mock_llm_class,
):
    mock_llm_class.return_value = MagicMock()

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(
        content=(
            "The provided evidence does not contain enough "
            "information to answer this question reliably."
        )
    )

    mock_prompt.__or__.return_value = mock_chain

    document = Document(
        page_content="Regular maintenance includes checking disk space.",
        metadata={
            "source": "sample.txt",
            "chunk_id": 3,
        },
    )

    with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
        answer = generate_answer(
            "Who manufactured this server?",
            [document],
        )

    assert "does not contain enough information" in answer.lower()
    mock_chain.invoke.assert_called_once()
