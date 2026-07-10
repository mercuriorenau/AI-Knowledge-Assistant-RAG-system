from backend.services.faithfulness import is_faithful


def test_empty_context_requires_refusal():
    assert is_faithful(
        "I don't know. No relevant information found in the indexed documents.",
        "",
        context_empty=True,
    )
    assert not is_faithful("Acme was founded in 1999.", "", context_empty=True)


def test_grounded_answer_overlaps_context():
    context = "Acme Robotics was founded in 1999 in Austin, Texas."
    assert is_faithful(
        "According to the documents, Acme Robotics was founded in 1999.",
        context,
        context_empty=False,
    )
    assert not is_faithful(
        "The moon is made of green cheese and unicorns.",
        context,
        context_empty=False,
    )
