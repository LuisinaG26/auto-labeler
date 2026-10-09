"""El código no debe conocer ningún proyecto concreto (ver sección 6.3 del plan)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("autolabel", "services")
# Palabras específicas del proyecto de prueba. Si aparecen en el código, hay un supuesto fijo.
FORBIDDEN = re.compile(r"tomat|\bripe\b|\bunripe\b|maduro|semimaduro|\bverde\b", re.IGNORECASE)


def test_no_project_specific_words_in_code():
    offenders = []
    for folder in SCANNED:
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in {".py", ".txt", ".md", ".yaml", ".yml", ".j2"}:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not offenders, "Texto específico de un proyecto en el código:\n" + "\n".join(offenders)
