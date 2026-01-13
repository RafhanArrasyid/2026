import asyncio
import json
import os
from typing import Any, Optional

import ccxt
import pandas as pd

from config import Config


class BinanceSyncWrapper:
    """Membungkus ccxt sync agar bisa dipakai async via to_thread.

    Catatan penting:
    - ccxt sync client tidak didesain aman untuk akses paralel lintas thread.
    - Maka semua network call diproteksi oleh lock (serial), supaya stabil 24/7.
    """

    def __init__(self, config: dict[str, Any]):
        self.client = ccxt.binance(config)
        self._lock = asyncio.Lock()

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    async def _call(self, fn, *args, **kwargs):
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def load_markets(self):
        return await self._call(self.client.load_markets)

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        return await self._call(self.client.fetch_ohlcv, symbol, timeframe, limit=limit)

    async def fetch_ticker(self, symbol):
        return await self._call(self.client.fetch_ticker, symbol)

    async def fetch_balance(self):
        return await self._call(self.client.fetch_balance)

    async def fetch_positions(self, symbols=None):
        return await self._call(self.client.fetch_positions, symbols)

    async def fetch_open_orders(self, symbol=None):
        return await self._call(self.client.fetch_open_orders, symbol)

    async def fetch_order(self, id, symbol=None):
        return await self._call(self.client.fetch_order, id, symbol)

    async def create_order(self, symbol, type, side, amount, price=None, params=None):
        if params is None:
            params = {}
        return await self._call(self.client.create_order, symbol, type, side, amount, price, params)

    async def cancel_order(self, id, symbol):
        return await self._call(self.client.cancel_order, id, symbol)

    async def cancel_all_orders(self, symbol):
        if hasattr(self.client, 'cancel_all_orders'):
            return await self._call(self.client.cancel_all_orders, symbol)
        # fallback manual: fetch_open_orders + cancel
        orders = await self.fetch_open_orders(symbol)
        for o in orders:
            try:
                await self.cancel_order(o['id'], symbol)
            except Exception:
                pass
        return True

    async def set_leverage(self, leverage: int, symbol: str):
        # Tidak semua market support; lindungi dengan try.
        if hasattr(self.client, 'set_leverage'):
            return await self._call(self.client.set_leverage, leverage, symbol)
        return None

    async def close(self):
        # ccxt sync tidak butuh close
        return None


class ExchangeLoader:
    def __init__(self):
        self.exchange: Optional[BinanceSyncWrapper] = None
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self.paper_state_file = os.path.join(self._base_dir, "paper_wallet.json")
        self._markets_loaded = False
        self._init_exchange()
        self._init_paper_wallet()

    def _init_exchange(self):
        config = {
            'apiKey': Config.API_KEY,
            'secret': Config.SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            },
        }
        self.exchange = BinanceSyncWrapper(config)

    def _init_paper_wallet(self):
        if Config.TRADING_MODE != "PAPER":
            return
        if not os.path.exists(self.paper_state_file):
            initial_data = {
                "balance": float(Config.PAPER_INITIAL_BALANCE),
                "positions": {},
                "history": [],
            }
            with open(self.paper_state_file, 'w') as f:
                json.dump(initial_data, f, indent=4)
            print(f"[LOADER] Paper Wallet created. Balance: ${Config.PAPER_INITIAL_BALANCE}")

    async def ensure_markets(self):
        if not self.exchange:
            return
        if self._markets_loaded and getattr(self.exchange, 'markets', None):
            return
        try:
            await self.exchange.load_markets()
            self._markets_loaded = True
        except Exception:
            # Kalau load_markets gagal, kita lanjut; beberapa fungsi masih bisa jalan.
            self._markets_loaded = False

    async def get_balance(self) -> float:
        if Config.TRADING_MODE == "LIVE":
            try:
                bal = await self.exchange.fetch_balance()
                # Binance futures kadang meletakkan USDT di total / free / used
                if isinstance(bal, dict):
                    total = bal.get('total', {}) or {}
                    free = bal.get('free', {}) or {}
                    used = bal.get('used', {}) or {}
                    usdt = float(total.get('USDT', 0) or 0)
                    if usdt == 0:
                        usdt = float(free.get('USDT', 0) or 0) + float(used.get('USDT', 0) or 0)
                    return float(usdt)
                return 0.0
            except Exception:
                return 0.0

        # PAPER
        try:
            with open(self.paper_state_file, 'r') as f:
                data = json.load(f)
            return float(data.get("balance", 0.0))
        except Exception:
            return 0.0

    async def fetch_candles(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        try:
            await self.ensure_markets()
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df.dropna(inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            last = ticker.get('last') if isinstance(ticker, dict) else None
            if last is None:
                last = (ticker.get('close') if isinstance(ticker, dict) else None)
            return float(last) if last is not None else None
        except Exception:
            return None

    async def fetch_positions(self):
        try:
            await self.ensure_markets()
            if hasattr(self.exchange, 'fetch_positions'):
                return await self.exchange.fetch_positions()
        except Exception:
            pass
        return []

    async def fetch_open_orders(self, symbol: Optional[str] = None):
        try:
            await self.ensure_markets()
            return await self.exchange.fetch_open_orders(symbol)
        except Exception:
            return []

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        try:
            return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception:
            return float(amount)

    def price_to_precision(self, symbol: str, price: float) -> float:
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception:
            return float(price)

    def get_market_limits(self, symbol: str) -> dict[str, Any]:
        try:
            m = (self.exchange.markets or {}).get(symbol, {})
            return {
                'amount_min': (m.get('limits', {}).get('amount', {}) or {}).get('min'),
                'cost_min': (m.get('limits', {}).get('cost', {}) or {}).get('min'),
            }
        except Exception:
            return {'amount_min': None, 'cost_min': None}

    async def close_connection(self):
        if self.exchange:
            await self.exchange.close()
