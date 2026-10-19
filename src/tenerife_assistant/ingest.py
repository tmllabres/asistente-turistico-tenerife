"""Lectura de la guía en PDF y división en chunks.

La guía tiene una estructura muy clara que se puede leer del PDF:

- Títulos de zona en negrita y pegados al margen (ZONA NORTE, ZONA SUR, OTRAS VAINAS).
- Un lugar por cada viñeta "•" (Santa Cruz de Tenerife, La Orotava...).
- Puntos de interés con viñeta "o" y restaurantes con viñeta "▪".

Aprovecho esa estructura para hacer un chunking estructural: cada fragmento es
de un solo lugar y lleva delante su ruta (Zona Norte › La Orotava). También
implemento el chunking clásico de tamaño fijo con solapamiento, para poder
comparar los dos en la evaluación.
"""

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

# Versión del troceado. Se guarda en el manifiesto del índice para que, si
# cambio la limpieza o el chunking, el índice guardado en disco se vuelva a crear.
CHUNKER_VERSION = "1.2"

# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------


@dataclass
class Line:
    """Línea de texto del PDF con su posición y su estilo."""

    text: str
    page: int
    x0: float
    size: float
    bold: bool


@dataclass
class Block:
    """Párrafo o elemento de lista dentro de un lugar.

    level es 1 para párrafos normales, 2 para viñetas "o" o guiones y 3 para
    viñetas "▪". cont_indent es la sangría de sus líneas de continuación: sirve
    para saber si la línea siguiente sigue en el mismo bloque.
    """

    level: int
    text: str
    pages: list[int]
    cont_indent: float


@dataclass
class Section:
    """Un lugar de la guía (una viñeta "•") con todos sus bloques."""

    zone: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def pages(self) -> list[int]:
        """Páginas que ocupa el lugar (ordenadas y sin repetir)."""
        return sorted({p for b in self.blocks for p in b.pages})

    @property
    def text(self) -> str:
        """Texto completo del lugar, un bloque por línea."""
        return "\n".join(b.text for b in self.blocks)


@dataclass
class Chunk:
    """Fragmento de la guía que se guarda en el índice.

    Attributes:
        chunk_id: identificador (por ejemplo "s07-2" o "f012").
        text: texto limpio del fragmento.
        zone: zona de la guía.
        section: lugar (viñeta "•") al que pertenece.
        pages: páginas del PDF donde aparece.
        strategy: estrategia de chunking con la que se creó.
    """

    chunk_id: str
    text: str
    zone: str
    section: str
    pages: tuple[int, ...]
    strategy: str

    @property
    def header(self) -> str:
        """Ruta del fragmento, por ejemplo "Zona Norte › La Orotava"."""
        if self.section:
            return f"{self.zone} › {self.section}"
        return self.zone

    @property
    def page_label(self) -> str:
        """Páginas en texto: "pág. 9" o "págs. 9-11"."""
        if not self.pages:
            return "pág. ?"
        first, last = min(self.pages), max(self.pages)
        return f"pág. {first}" if first == last else f"págs. {first}-{last}"

    @property
    def citation(self) -> str:
        """Cita que se le enseña al usuario."""
        return f"Guía de Tenerife · {self.header} ({self.page_label})"

    @property
    def embedding_text(self) -> str:
        """Texto que se convierte en embedding: la ruta + el contenido.

        Añadir la ruta ayuda a que "¿qué ver en La Orotava?" encuentre
        fragmentos en los que no se repite el nombre del lugar.
        """
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        """Convierte el chunk en un diccionario (para guardarlo en JSON)."""
        data = asdict(self)
        data["pages"] = list(self.pages)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Crea un chunk a partir del diccionario de to_dict()."""
        data = dict(data)
        data["pages"] = tuple(data["pages"])
        return cls(**data)


# ---------------------------------------------------------------------------
# Limpieza del texto
# ---------------------------------------------------------------------------

# Al pasar el Word a PDF se perdieron los enlaces, pero quedaron las etiquetas
# ("[vídeo - ubicación]", "(ubicación)", "[web]"...). No aportan nada y meten
# ruido en los embeddings, así que las quito.
_LINK_PATTERNS = [
    r"\[\s*v[ií]deo\s*-\s*ubicaci[oó]n\s*\]",
    r"\[\s*(?:v[ií]deo|ubicaci[oó]n|web|tripadvisor)\s*\]",
    r"\(\s*ubicaci[oó]n\s*\)",
    r"\s[-–]\s*ubicaci[oó]n\s*[-–]?",
    r"\(entradas aqu[ií]\)",
]
_LINK_RE = re.compile("|".join(_LINK_PATTERNS), flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Quita las marcas de enlaces rotos y arregla los espacios.

    Args:
        text: texto tal cual sale del PDF.

    Returns:
        Texto limpio, sin espacios dobles ni espacios antes de los signos de puntuación.
    """
    text = _LINK_RE.sub(" ", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    text = re.sub(r"([;,])\1+", r"\1", text)
    text = re.sub(r"\s+[–-]\s*([.,;])", r"\1", text)  # "de hecho) –." -> "de hecho)."
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Lectura del PDF
# ---------------------------------------------------------------------------


def file_sha256(path) -> str:
    """Calcula el hash SHA-256 de un fichero (si cambia el PDF, cambia el hash)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lines(pdf_path) -> list[Line]:
    """Lee todas las líneas de texto del PDF con su posición y su estilo.

    Args:
        pdf_path: ruta de la guía en PDF.

    Returns:
        Las líneas en orden de lectura (sin las vacías).

    Raises:
        FileNotFoundError: si el PDF no existe.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"No encuentro la guía en {pdf_path}")

    lines: list[Line] = []
    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            layout = page.get_text("dict", sort=False)
            for block in layout["blocks"]:
                if block.get("type") != 0:  # los bloques de tipo 1 son imágenes
                    continue
                for raw_line in block["lines"]:
                    spans = raw_line["spans"]
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    first = spans[0]
                    lines.append(
                        Line(
                            text=text,
                            page=page_number,
                            x0=raw_line["bbox"][0],
                            size=first["size"],
                            bold="Bold" in first["font"] or bool(first["flags"] & 16),
                        )
                    )
    return lines


def _is_zone_heading(line: Line) -> bool:
    """Comprueba si una línea es un título de zona (negrita, mayúsculas y pegada al margen)."""
    return line.bold and line.x0 < 85 and line.text.isupper() and line.size < 14


def parse_sections(lines: list[Line]) -> list[Section]:
    """Reconstruye la estructura zona → lugar → bloques a partir de las líneas.

    Args:
        lines: líneas devueltas por read_lines().

    Returns:
        Lista de lugares (uno por cada viñeta "•" de la guía).
    """
    sections: list[Section] = []
    zone = "Introducción"
    current: Section | None = None
    pending_dash = False

    def new_block(level: int, text: str, page: int, indent: float) -> None:
        if current is not None:
            current.blocks.append(Block(level, text, [page], indent))

    for line in lines:
        text = line.text

        if line.size >= 14:  # título del documento ("TENERIFE – ..."), lo salto
            continue
        if _is_zone_heading(line):
            if text.strip().upper() == "SITIOS QUE VER":
                continue
            zone = text.strip().title()
            current = None
            continue
        if text.startswith("•"):
            title = text.lstrip("•").strip().rstrip(":").strip()
            # "(sobre todo os podré decir de zona norte)" es un comentario, no parte
            # del nombre; los paréntesis con nombres propios (Isla Baja) se conservan.
            title = re.sub(r"\s*\((?![^)]*[A-ZÁÉÍÓÚ])[^)]*\)", "", title)
            current = Section(zone=zone, title=title)
            sections.append(current)
            continue
        if current is None:
            continue

        if text == "-":  # guion suelto: el texto viene en la línea siguiente
            pending_dash = True
            continue
        if pending_dash:
            new_block(2, f"- {text}", line.page, line.x0 - 2)
            pending_dash = False
        elif text.startswith("o ") and line.x0 > 115:
            new_block(2, text[2:].strip(), line.page, line.x0 + 15)
        elif text.startswith("▪"):
            new_block(3, text.lstrip("▪").strip(), line.page, line.x0 + 15)
        else:
            # Línea normal: si tiene la sangría del bloque anterior es su continuación;
            # si no, empieza un párrafo nuevo.
            last = current.blocks[-1] if current.blocks else None
            if last is not None and line.x0 >= last.cont_indent - 6:
                last.text = f"{last.text} {text}"
                if line.page not in last.pages:
                    last.pages.append(line.page)
            elif last is not None and last.level == 1 and line.x0 < 120:
                last.text = f"{last.text} {text}"
                if line.page not in last.pages:
                    last.pages.append(line.page)
            else:
                new_block(1, text, line.page, line.x0 - 2)

    for section in sections:
        for block in section.blocks:
            block.text = clean_text(block.text)
        section.blocks = [b for b in section.blocks if b.text]
    return sections


def load_guide(pdf_path) -> list[Section]:
    """Lee el PDF y devuelve la lista de lugares."""
    return parse_sections(read_lines(pdf_path))


# ---------------------------------------------------------------------------
# Estrategias de chunking
# ---------------------------------------------------------------------------

# Corta después de . ! ? cuando la frase siguiente empieza por mayúscula.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡(])")


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Parte un texto demasiado largo por frases, sin pasar de max_chars."""
    sentences = _SENTENCE_RE.split(text)
    parts: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) <= max_chars or not buffer:
            buffer = candidate
        else:
            parts.append(buffer)
            buffer = sentence
    if buffer:
        parts.append(buffer)
    return parts


def chunk_structural(sections: list[Section], max_chars: int = 900) -> list[Chunk]:
    """Chunking estructural: junta bloques del mismo lugar hasta llegar a max_chars.

    Nunca mezcla dos lugares en el mismo fragmento. Solo parte un bloque si por
    sí solo pasa del límite (y entonces lo parte por frases).

    Args:
        sections: lugares devueltos por parse_sections().
        max_chars: tamaño máximo aproximado de cada fragmento.

    Returns:
        Lista de chunks con su zona, lugar y páginas.
    """
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(sections, start=1):
        pieces: list[tuple[str, list[int]]] = []
        for block in section.blocks:
            if len(block.text) > max_chars:
                pieces.extend((p, block.pages) for p in _split_long_text(block.text, max_chars))
            else:
                pieces.append((block.text, block.pages))

        # Voy llenando grupos hasta que el siguiente trozo ya no cabe.
        groups: list[tuple[list[str], set[int]]] = []
        texts: list[str] = []
        pages: set[int] = set()
        for text, block_pages in pieces:
            size = sum(len(t) + 1 for t in texts) + len(text)
            if texts and size > max_chars:
                groups.append((texts, pages))
                texts, pages = [], set()
            texts.append(text)
            pages.update(block_pages)
        if texts:
            groups.append((texts, pages))

        for g_idx, (group_texts, group_pages) in enumerate(groups, start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"s{s_idx:02d}-{g_idx}",
                    text="\n".join(group_texts),
                    zone=section.zone,
                    section=section.title,
                    pages=tuple(sorted(group_pages)),
                    strategy=f"structural-{max_chars}",
                )
            )
    return chunks


def chunk_fixed(sections: list[Section], size: int = 600, overlap: int = 100) -> list[Chunk]:
    """Chunking de tamaño fijo con solapamiento (para comparar).

    Junta todo el texto de la guía y lo corta en ventanas de `size` caracteres
    que se solapan `overlap` caracteres, cortando siempre en un espacio. A cada
    ventana le asigno el lugar en el que empieza.

    Args:
        sections: lugares devueltos por parse_sections().
        size: tamaño de la ventana en caracteres.
        overlap: caracteres que se repiten entre una ventana y la siguiente.

    Returns:
        Lista de chunks.
    """
    if overlap >= size:
        raise ValueError("overlap debe ser menor que size")

    # Texto completo + dónde empieza y acaba cada bloque, para saber luego
    # a qué lugar y a qué páginas pertenece cada ventana.
    full_text = ""
    spans: list[tuple[int, int, Section, list[int]]] = []
    for section in sections:
        for block in section.blocks:
            start = len(full_text)
            full_text += block.text + "\n"
            spans.append((start, len(full_text), section, block.pages))

    def meta_at(pos: int) -> tuple[Section, list[int]]:
        for start, end, section, pages in spans:
            if start <= pos < end:
                return section, pages
        return spans[-1][2], spans[-1][3]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(full_text):
        end = min(start + size, len(full_text))
        if end < len(full_text):
            space = full_text.rfind(" ", start + size // 2, end)
            end = space if space != -1 else end
        piece = full_text[start:end].strip()
        if piece:
            section, _ = meta_at(start)
            pages = {p for s0, e0, _, pgs in spans if s0 < end and e0 > start for p in pgs}
            idx += 1
            chunks.append(
                Chunk(
                    chunk_id=f"f{idx:03d}",
                    text=piece.replace("\n", " "),
                    zone=section.zone,
                    section=section.title,
                    pages=tuple(sorted(pages)),
                    strategy=f"fixed-{size}-{overlap}",
                )
            )
        if end >= len(full_text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(sections: list[Section], strategy: str = "structural", **kwargs) -> list[Chunk]:
    """Genera los chunks con la estrategia indicada.

    Args:
        sections: lugares de la guía.
        strategy: "structural" o "fixed".
        **kwargs: parámetros de la estrategia (max_chars, o size y overlap).

    Returns:
        Lista de chunks.
    """
    if strategy == "structural":
        return chunk_structural(sections, **kwargs)
    if strategy == "fixed":
        return chunk_fixed(sections, **kwargs)
    raise ValueError(f"Estrategia de chunking desconocida: {strategy}")
