from tenerife_assistant.memory import ConversationMemory, count_message_tokens

LONG = "La Orotava tiene un casco histórico precioso con la Casa de los Balcones. " * 8


def test_budget_triggers_summary_and_keeps_recent_turns():
    calls = []

    def summarizer(previous, turns):
        calls.append(len(turns))
        return "El usuario planea visitar La Orotava."

    memory = ConversationMemory(max_tokens=300, keep_last_turns=1, summarizer=summarizer)
    for i in range(4):
        memory.add_turn(f"pregunta {i}", LONG)
    assert calls, "debería haberse llamado al resumidor"
    assert memory.summary == "El usuario planea visitar La Orotava."
    assert memory.turns[-1].user == "pregunta 3"
    assert memory.tokens <= 300


def test_sliding_window_without_summarizer():
    memory = ConversationMemory(max_tokens=250, keep_last_turns=1, summarizer=None)
    for i in range(5):
        memory.add_turn(f"pregunta {i}", LONG)
    assert memory.tokens <= 250
    assert [t.user for t in memory.turns] == ["pregunta 4"]


def test_failing_summarizer_falls_back_to_window():
    def broken(previous, turns):
        raise RuntimeError("API caída")

    memory = ConversationMemory(max_tokens=250, keep_last_turns=1, summarizer=broken)
    for i in range(4):
        memory.add_turn(f"pregunta {i}", LONG)
    assert memory.tokens <= 250 and memory.summary == ""


def test_messages_format_and_token_count():
    memory = ConversationMemory(max_tokens=10_000)
    memory.add_turn("hola", "¡hola! ¿en qué te ayudo?")
    messages = memory.as_messages()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert count_message_tokens(messages) == memory.tokens
    memory.clear()
    assert memory.as_messages() == [] and memory.tokens == 0
