"""Kural tabanli sorgu ayristirici - POC'un ilk asamasi. Arayuz
(ParsedQuery) sabit kalacak sekilde tasarlandi; LLM tabanli ayristiriciya
gecişte yalnizca bu dosyanin ici degisir, search/query.py ve sonrasi
dokunulmaz."""
import re
from dataclasses import dataclass, field

# Uretimdeki "alan katalogu"nun POC karsiligi: kavram -> (kolon, operator, deger)
CONCEPT_MAP = {
    r"otobüs": ("bus_count", ">=", 1),
    r"kamyon": ("truck_count", ">=", 1),
    r"araba|araç|otomobil": ("car_count", ">=", 1),
    r"insan|adam|yaya|kişi": ("person_count", ">=", 1),
    r"gece": ("is_night", "=", 1),
    r"gündüz": ("is_night", "=", 0),
    r"kalabalık": ("person_count", ">=", 10),
}

# Bu kelimeler kolonlasmiyor, semantik artiga duser (yine de semantic her
# zaman TAM sorguyu alir - bkz. parse()). Belgeleme amacli, kodda kullanilmiyor.
SEMANTIC_ONLY_HINTS = ["yürüyen", "koşan", "dönen", "bekleyen", "hızlı",
                        "kavşak", "otopark", "yol kenarı"]


@dataclass
class ParsedQuery:
    filters: list = field(default_factory=list)
    semantic: str = ""


def parse(q: str) -> ParsedQuery:
    p = ParsedQuery(semantic=q)
    for pattern, cond in CONCEPT_MAP.items():
        if re.search(pattern, q, re.IGNORECASE):
            p.filters.append(cond)
    return p
