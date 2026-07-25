"""Kural tabanli sorgu ayristirici - POC'un ilk asamasi. Arayuz
(ParsedQuery) sabit kalacak sekilde tasarlandi; LLM tabanli ayristiriciya
gecişte yalnizca bu dosyanin ici degisir, search/query.py ve sonrasi
dokunulmaz."""
import re
from dataclasses import dataclass, field

# Uretimdeki "alan katalogu"nun POC karsiligi: kavram -> (kolon, operator, deger)
# TR + EN esanlamlilar ayni pattern'de birlesik: bench'te dil karsilastirmasi
# (Faz 1) filtre varligina gore degil yalnizca semantik kaliteye gore
# ayrissin diye - Ingilizce sorgu Turkce esdegeriyle ayni filtreyi almali.
CONCEPT_MAP = {
    r"otobüs|\bbus(es)?\b": ("bus_count", ">=", 1),
    r"kamyon|\btrucks?\b": ("truck_count", ">=", 1),
    r"araba|araç|otomobil|\bcars?\b|\bvehicles?\b": ("car_count", ">=", 1),
    r"insan|adam|yaya|kişi|\bperson\b|\bpeople\b|\bpedestrians?\b": ("person_count", ">=", 1),
    r"gece|\bnight\b": ("is_night", "=", 1),
    r"gündüz|\bday(time)?\b": ("is_night", "=", 0),
    r"kalabalık|\bcrowd(ed)?\b": ("person_count", ">=", 10),
}

# "en az N X" / "at least N X" - sayisal esik sorgulari. Deger regex'in
# yakaladigi ilk dolu grup; kolon esanlamli TR/EN varyantlar ayni kolona gider.
NUMERIC_MAP = [
    (r"en az (\d+) araba|en az (\d+) araç|at least (\d+) cars?", "car_count"),
    (r"en az (\d+) kişi|en az (\d+) yaya|at least (\d+) (?:people|persons?|pedestrians?)",
     "person_count"),
]

# Bu kelimeler kolonlasmiyor, semantik artiga duser (yine de semantic her
# zaman TAM sorguyu alir - bkz. parse()). Belgeleme amacli, kodda kullanilmiyor.
SEMANTIC_ONLY_HINTS = ["yürüyen", "koşan", "dönen", "bekleyen", "hızlı",
                        "kavşak", "otopark", "yol kenarı",
                        "walking", "running", "turning", "waiting"]


@dataclass
class ParsedQuery:
    filters: list = field(default_factory=list)
    semantic: str = ""


def parse(q: str) -> ParsedQuery:
    p = ParsedQuery(semantic=q)
    for pattern, cond in CONCEPT_MAP.items():
        if re.search(pattern, q, re.IGNORECASE):
            p.filters.append(cond)
    for pattern, col in NUMERIC_MAP:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            n = next(g for g in m.groups() if g is not None)
            p.filters.append((col, ">=", int(n)))
    return p
