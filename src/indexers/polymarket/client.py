from collections.abc import Generator
from typing import Union

from src.common.client import HttpClient
from src.indexers.polymarket.models import Market

GAMMA_API_URL = "https://gamma-api.polymarket.com"


class PolymarketClient:
    def __init__(self, gamma_url: str = GAMMA_API_URL):
        self.gamma_url = gamma_url
        self.http = HttpClient(rate_limit=10)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def close(self):
        self.http.close()

    def get_markets(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Market]:
        params = {"limit": limit, "offset": offset, **kwargs}
        data: Union[dict, list] = self.http.get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return [Market.from_dict(m) for m in data]
        return [Market.from_dict(m) for m in data.get("markets", data)]

    def iter_markets(self, limit: int = 500, offset: int = 0) -> Generator[tuple[list[Market], int], None, None]:
        current_offset = offset

        while True:
            markets = self.get_markets(limit=limit, offset=current_offset)

            if not markets:
                yield [], -1
                break

            next_offset = current_offset + len(markets)
            yield markets, next_offset

            if len(markets) < limit:
                break

            current_offset = next_offset
