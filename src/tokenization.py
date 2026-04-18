"""Character-level tokenizer for Hebrew ASR (causal LM).

Vocab:
  - Special tokens: [PAD], [UNK], [BOS], [EOS], [EOA]
  - Hebrew letters: alef-tav (including final forms)
  - ASCII lowercase + digits + punctuation + space
  - Hebrew punctuation: maqaf, geresh, gershayim

Normalizer:
  - NFKC
  - Lowercase
  - StripAccents
"""

from __future__ import annotations

import string
import unicodedata
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[EOA]"]


def build_vocab() -> dict[str, int]:
    hebrew = [
        chr(cp)
        for cp in range(0x05D0, 0x05EB)
        if unicodedata.category(chr(cp)) == "Lo"
    ]
    # Hebrew punctuation — see https://en.wikipedia.org/wiki/Unicode_and_HTML_for_the_Hebrew_alphabet
    hebrew_punct = [
        "\u05BE",  # maqaf (Hebrew hyphen)
        "\u05F3",  # geresh
        "\u05F4",  # gershayim
    ]
    chars = (
        list(string.ascii_lowercase)
        + list(string.digits)
        + list(string.punctuation)
        + [" "]
        + hebrew
        + hebrew_punct
    )

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for c in chars:
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def build_tokenizer() -> Tokenizer:
    vocab = build_vocab()

    tokenizer = Tokenizer(WordPiece(vocab, unk_token="[UNK]", continuing_subword_prefix="##"))
    tokenizer.normalizer = Sequence([NFKC(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Split(pattern=Regex("[\\s\\S]"), behavior="isolated")
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL_TOKENS])

    return tokenizer


def save_tokenizer(path: str | Path) -> None:
    build_tokenizer().save(str(path))


def id_to_token(tokenizer) -> dict[int, str]:
    return {v: k for k, v in tokenizer.get_vocab().items()}


@lru_cache(maxsize=None)
def load_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=build_tokenizer(),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
