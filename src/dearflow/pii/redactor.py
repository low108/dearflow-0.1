"""Finds personal data in text and replaces it with stable placeholders.

Deterministic on purpose, like DeerFlow's: fixed regexes, plus a checksum where
the identifier format defines one. No model call, no stored mapping.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dearflow.config import PiiRedactionConfig

# --- checksums: the gate for numeric identifiers --------------------------------


def luhn_valid(value: str) -> bool:
    """Luhn checksum over the digits of `value` (separators ignored). Used for card numbers."""
    digits = [int(ch) for ch in value if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for offset, digit in enumerate(reversed(digits)):
        if offset % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


_CN_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_ID_CHECK_DIGITS = "10X98765432"


def cn_resident_id_valid(value: str) -> bool:
    """GB 11643 checksum for the 18-character Chinese resident ID."""
    body = value[:17]
    if not body.isdigit():
        return False
    year, month, day = int(body[6:10]), int(body[10:12]), int(body[12:14])
    if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31):
        return False
    total = sum(int(digit) * weight for digit, weight in zip(body, _CN_ID_WEIGHTS))
    return _CN_ID_CHECK_DIGITS[total % 11] == value[17].upper()


def cpf_valid(value: str) -> bool:
    """Brazilian CPF: two mod-11 check digits."""
    digits = [int(ch) for ch in value if ch.isdigit()]
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for boundary in (9, 10):
        total = sum(digit * weight for weight, digit in enumerate(reversed(digits[:boundary]), start=2))
        if (total * 10) % 11 % 10 != digits[boundary]:
            return False
    return True


def national_id_valid(value: str) -> bool:
    """CN resident ID and CPF are checksum-validated; CUIT and RFC are format-only."""
    if "." in value:
        return cpf_valid(value)
    if len(value) == 18 and value[:17].isdigit():
        return cn_resident_id_valid(value)
    return True


# --- detectors -------------------------------------------------------------------


@dataclass(frozen=True)
class Detector:
    name: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

_API_KEY = re.compile(
    r"\b(?:"
    r"sk-[A-Za-z0-9_-]{20,}"  # OpenAI-style
    r"|AKIA[0-9A-Z]{16}"  # AWS access key id
    r"|gh[pousr]_[A-Za-z0-9]{30,}"  # GitHub token
    r"|github_pat_[A-Za-z0-9_]{20,}"  # GitHub fine-grained token
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack token
    r"|AIza[0-9A-Za-z_-]{35}"  # Google API key
    r")\b"
)

# Digit-anchored patterns use digit lookarounds, not \b: CJK characters count as word
# characters, so \b would miss "身份证110105..." (a Chinese label glued to the number).
_CREDIT_CARD = re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{1,7}(?!\d)|(?<!\d)\d{13,19}(?!\d)")

_PHONE = re.compile(
    r"(?<!\d)\+\d{1,3}(?:[ \-]?\d{1,4}){3,6}(?!\d)"  # international +CC form
    r"|(?<!\d)1[3-9]\d{9}(?!\d)"  # CN mobile
    r"|\(\d{3}\) ?\d{3}[-.]?\d{4}(?!\d)"  # US formatted
)

_NATIONAL_ID = re.compile(
    r"(?<![\dXx])\d{17}[\dXx](?![\dXx])"  # CN resident ID (checksum)
    r"|(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)"  # CPF (checksum)
    r"|(?<!\d)\d{2}-\d{8}-\d(?!\d)"  # CUIT (format only)
    r"|(?<![A-Z0-9Ñ&])[A-ZÑ&]{4}\d{6}[0-9A-Z]{3}(?![A-Z0-9Ñ&])"  # RFC with homoclave (format only)
)

# ORDER MATTERS. Emails and API keys have unambiguous formats, so they go first. Checksummed
# national IDs come before cards: an 18-digit resident ID can also pass Luhn and must not be
# eaten as a card. Phones go last and only see digits the stronger detectors left alone.
DETECTORS: tuple[Detector, ...] = (
    Detector("email", _EMAIL),
    Detector("api_key", _API_KEY),
    Detector("national_id", _NATIONAL_ID, national_id_valid),
    Detector("credit_card", _CREDIT_CARD, luhn_valid),
    Detector("phone", _PHONE, lambda value: 8 <= len(re.sub(r"\D", "", value)) <= 15),
)

MIN_SECRET_LENGTH = 16


def placeholder(category: str, value: str, secret: str) -> str:
    """`[EMAIL_abcdefghijklmnopqrstuvwxyz]`: a keyed, value-derived stand-in.

    - Same value, same token, every time (an HMAC of the value), so the model can still
      tell that two mentions are the same person, across turns, with no stored mapping.
    - Different secrets give different tokens, and without the secret a token can't be
      recomputed from a guessed value (phone numbers are too easy to guess unkeyed).
    - Letters only: a token never contains a digit run, so no later detector can match
      inside an earlier placeholder.
    """
    if not secret:
        raise ValueError("placeholders need a non-empty secret")
    digest = hmac.new(secret.encode(), f"{category}\x00{value}".encode(), hashlib.sha256).digest()
    number = int.from_bytes(digest[:16], "big")  # 128 bits
    letters = []
    for _ in range(27):
        number, remainder = divmod(number, 26)
        letters.append(chr(ord("a") + remainder))
    return f"[{category.upper()}_{''.join(letters)}]"


class Redactor:
    """Applies the enabled detectors, in order, to text. Stateless, so safe to share."""

    def __init__(self, detectors: Sequence[Detector], secret: str) -> None:
        self._detectors = tuple(detectors)
        self._secret = secret

    @property
    def detector_names(self) -> list[str]:
        return [d.name for d in self._detectors]

    def redact(self, text: str) -> str:
        for detector in self._detectors:
            text = detector.pattern.sub(self._replacement(detector), text)
        return text

    def redact_content(self, content: object) -> tuple[object, bool]:
        """Redact message content of either shape (string or list of blocks).

        Returns `(content, changed)`. The input is never modified. Only `text` fields of
        blocks are touched; a non-text block that carries a `text` field is still covered.
        """
        if isinstance(content, str):
            redacted = self.redact(content)
            return redacted, redacted != content
        if not isinstance(content, list):
            return content, False
        blocks: list = []
        changed = False
        for block in content:
            if isinstance(block, str):
                new = self.redact(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text = self.redact(block["text"])
                new = {**block, "text": text} if text != block["text"] else block
            else:
                new = block
            changed = changed or new != block
            blocks.append(new)
        return blocks, changed

    def _replacement(self, detector: Detector) -> Callable[[re.Match[str]], str]:
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if detector.validator is not None and not detector.validator(value):
                return value  # looks like one, fails the checksum: leave it alone
            return placeholder(detector.name, value, self._secret)

        return replace


def make_redactor(config: PiiRedactionConfig | None, secret: str | None = None) -> Redactor | None:
    """A redactor for the enabled detectors, or None when redaction is off.

    `secret` is the already-resolved key (the config may hold `$ENV_VAR`). It must be long
    enough: a short key lets someone brute-force low-entropy values such as phone numbers.
    """
    if config is None or not config.enabled:
        return None
    key = secret if secret is not None else config.token_secret
    if not key or len(key.strip()) < MIN_SECRET_LENGTH:
        raise ValueError(f"pii_redaction.token_secret must be at least {MIN_SECRET_LENGTH} characters")
    return Redactor([d for d in DETECTORS if getattr(config, f"redact_{d.name}")], key)
