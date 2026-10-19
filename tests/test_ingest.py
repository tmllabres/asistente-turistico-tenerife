from tenerife_assistant.ingest import Chunk, build_chunks, clean_text


def test_parse_sections_finds_every_place(sections):
    titles = [s.title for s in sections]
    assert len(sections) == 13
    for expected in ["Santa Cruz de Tenerife", "La Orotava", "El Teide", "Masca", "Los Cristianos"]:
        assert expected in titles
    assert {s.zone for s in sections} == {"Zona Norte", "Zona Sur", "Otras Vainas"}


def test_parenthetical_comment_removed_but_names_kept(sections):
    titles = [s.title for s in sections]
    assert "Sitios para comer" in titles
    assert any(t.startswith("Isla Baja (Icod de los Vinos") for t in titles)


def test_pages_are_tracked(sections):
    orotava = next(s for s in sections if s.title == "La Orotava")
    assert orotava.pages[0] == 8 and 9 in orotava.pages


def test_clean_text_removes_broken_link_markers():
    raw = "Plaza de España [vídeo - ubicación] y Plaza Weyler - ubicación –; aparcar (ubicación)."
    cleaned = clean_text(raw)
    assert "ubicación" not in cleaned and "vídeo" not in cleaned
    assert cleaned.startswith("Plaza de España y Plaza Weyler;")


def test_structural_chunks_never_mix_places(chunks, sections):
    titles = {s.title for s in sections}
    assert all(c.section in titles for c in chunks)
    # Ningún chunk supera el límite salvo que sea un único bloque indivisible.
    assert all(len(c.text) <= 1000 or "\n" not in c.text for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_fixed_chunks_overlap(sections):
    fixed = build_chunks(sections, "fixed", size=400, overlap=100)
    assert len(fixed) > 10
    first, second = fixed[0].text, fixed[1].text
    assert first[-40:].split()[-1] in second  # el final de uno reaparece al inicio del siguiente


def test_chunk_roundtrip_and_citation(chunks):
    chunk = chunks[0]
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk
    assert chunk.citation.startswith("Guía de Tenerife · Zona Norte › Santa Cruz de Tenerife")
    assert chunk.embedding_text.startswith(chunk.header)
