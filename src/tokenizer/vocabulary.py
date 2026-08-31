"""Token-to-id vocabulary used by the tokenizer."""

from collections.abc import Iterable, Iterator, Mapping

from .types import *


class Vocabulary:
    """Bidirectional, append-only token vocabulary.

    Tokens are assigned consecutive integer ids.  A mapping may be supplied
    when ids must be loaded from an existing vocabulary; otherwise tokens are
    assigned in iterable order.
    """

    def __init__(
        self,
        tokens: Iterable[TokenData] | Mapping[TokenData, Token] = tuple(
            bytes([token]) for token in range(256)
        ),
    ) -> None:
        self.data_to_token: dict[TokenData, Token] = {}
        self.token_to_data: dict[Token, TokenData] = {}

        if isinstance(tokens, Mapping):
            for token_data, token in tokens.items():
                self._insert(token_data, token)
        else:
            for token_data in tokens:
                self.add_token(token_data)

    def _insert(self, token_data: TokenData, token: Token) -> None:
        if not isinstance(token, int) or token < 0:
            raise ValueError("token_data ids must be non-negative integers")
        if token_data in self.data_to_token:
            raise ValueError(f"token_data already exists: {token_data!r}")
        if token in self.token_to_data:
            raise ValueError(f"token_data id already exists: {token}")
        self.data_to_token[token_data] = token
        self.token_to_data[token] = token_data

    def add_token(self, token_data: TokenData) -> Token:
        """Add *token_data* and return its id; return its existing id unchanged."""
        existing = self.data_to_token.get(token_data)
        if existing is not None:
            return existing
        token = max(self.token_to_data, default=-1) + 1
        self._insert(token_data, token)
        return token

    def add_pair(self, pair: Pair) -> Token:
        token_data = b"".join([self.get_token_data(token) for token in pair])
        return self.add_token(token_data)

    def get_token(self, token: TokenData) -> Token:
        """Return the id for *token*, raising ``KeyError`` if absent."""
        return self.data_to_token[token]

    def get_token_data(self, token: Token) -> TokenData:
        """Return the token_data for *token*, raising ``KeyError`` if absent."""
        return self.token_to_data[token]

    def __len__(self) -> int:
        return len(self.data_to_token)

    def __contains__(self, token_data: object) -> bool:
        return token_data in self.data_to_token

    def __getitem__(self, token_data: TokenData) -> Token:
        return self.get_token(token_data)

    def __iter__(self) -> Iterator[TokenData]:
        return iter(self.data_to_token)

    def __repr__(self) -> str:
        return f"Vocabulary({self.data_to_token!r})"
