# wiki_analyzers/transliterator.py
"""Tatar Latin → Cyrillic transliteration."""

import re

CHAR_MAP = {
    "a":"а", "b":"б", "c":"ц", "d":"д", "e":"е", "f":"ф", "g":"г",
    "h":"х", "i":"и", "j":"җ", "k":"к", "l":"л", "m":"м", "n":"н",
    "o":"о", "p":"п", "q":"к", "r":"р", "s":"с", "t":"т", "u":"у",
    "v":"в", "w":"в", "x":"х", "y":"й", "z":"з",
    "ä":"ә", "ə":"ә", "ç":"ч", "ğ":"г", "ö":"ө", "ş":"ш", "ü":"ү", "ı":"ы", "ñ":"ң",
    "A":"А", "B":"Б", "C":"Ц", "D":"Д", "E":"Е", "F":"Ф", "G":"Г",
    "H":"Х", "I":"И", "J":"Җ", "K":"К", "L":"Л", "M":"М", "N":"Н",
    "O":"О", "P":"П", "Q":"К", "R":"Р", "S":"С", "T":"Т", "U":"У",
    "V":"В", "W":"В", "X":"Х", "Y":"Й", "Z":"З",
    "Ä":"Ә", "Ə":"Ә", "Ç":"Ч", "Ğ":"Г", "Ö":"Ө", "Ş":"Ш", "Ü":"Ү", "İ":"İ", "Ñ":"Ң"
}

DIGRAPHS = {
    "ng":"ң", "NG":"Ң", "Ng":"Ң",
    "iy":"ый", "İY":"ЫЙ", "Iy":"Ый",
    "ya":"я", "YA":"Я", "Ya":"Я",
    "yo":"ё", "YO":"Ё", "Yo":"Ё",
    "yu":"ю", "YU":"Ю", "Yu":"Ю",
    "ye":"е", "YE":"Е", "Ye":"Е",
    "ch":"ч", "CH":"Ч", "Ch":"Ч",
    "sh":"ш", "SH":"Ш", "Sh":"Ш",
    "ts":"ц", "TS":"Ц", "Ts":"Ц"
}

def transliterate(text: str) -> str:
    """Transliterate Tatar Latin text to Cyrillic."""
    # Process digraphs first
    for lat, cyr in DIGRAPHS.items():
        text = text.replace(lat, cyr)
    # Then character by character
    result = []
    for ch in text:
        result.append(CHAR_MAP.get(ch, ch))
    return ''.join(result)