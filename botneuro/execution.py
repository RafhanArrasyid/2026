import json
import os
import uuid
from datetime import datetime
from typing import Optional

from config import Config


class ExecutionHandler:
    """Eksekusi order.

    Perubahan versi harden:
    - LIVE: setelah entry ter-fill, bot langsung pasang SL/TP (reduceOnly) agar posisi selalu terlindungi.
    - LIVE: ada timeout untuk limit order. Jika tidak fill, dibatalkan.
    - Semua operasi dibungkus try/except agar loop utama tidak crash.
    """

    def __init__(self, loader):
        self.loader = loader

    async def place_entry(self, symbol: str, side: str, qty: float, price: float, sl: float, tp: float):
        if qty <= 0:
            return None

        # Normalize precision
        qty_p = self.loader.amount_to_precision(symbol, float(qty))
        price_p = self.loader.price_to_precision(symbol, float(price))
        sl_p = self.loader.price_to_precision(symbol, float(sl))
        tp_p = self.loader.price_to_precision(symbol, float(tp))

        if Config.TRADING_MODE == "LIVE":
            return await self._place_live_entry(symbol, side, qty_p, price_p, sl_p, tp_p)
        return self._place_paper_entry(symbol, side, qty_p, price_p, sl_p, tp_p)

    # ==========================
    # LIVE (BINANCE FUTURES)
    # ==========================
    async def _place_live_entry(self, symbol: str, side: str, qty: float, price: float, sl: float, tp: float):
        try:
            # set leverage (ignore error jika tidak support)
            try:
                await self.loader.exchange.set_leverage(int(Config.LEVERAGE), symbol)
            except Exception:
                pass

            # 1) kirim limit entry
            entry_order = await self.loader.exchange.create_order(symbol, 'limit', side, qty, price)
            entry_id = entry_order.get('id')

            # 2) tunggu fill (timeout)
            filled_qty = 0.0
            status = (entry_order.get('status') or '').lower()
            if status in ("closed", "filled"):
                filled_qty = float(entry_order.get('filled') or qty)
            else:
                # polling fetch_order
                import asyncio
                import time
                start = time.time()
                while time.time() - start < float(Config.ENTRY_ORDER_TIMEOUT_SEC):
                    await asyncio.sleep(float(Config.ENTRY_ORDER_POLL_SEC))
                    try:
                        o = await self.loader.exchange.fetch_order(entry_id, symbol)
                        st = (o.get('status') or '').lower()
                        filled_qty = float(o.get('filled') or 0.0)
                        if st in ("closed", "filled"):
                            break
                    except Exception:
                        continue

                # kalau belum filled, cancel
                if filled_qty <= 0:
                    try:
                        await self.loader.exchange.cancel_order(entry_id, symbol)
                    except Exception:
                        pass
                    return None

            # normalize filled qty
            filled_qty = self.loader.amount_to_precision(symbol, float(filled_qty))
            if filled_qty <= 0:
                return None

            # 3) pasang SL/TP reduceOnly
            sl_id, tp_id = await self._place_protective_orders(symbol, side, filled_qty, sl, tp)
            if sl_id is None or tp_id is None:
                # jika gagal pasang proteksi, tutup posisi agar tidak naked
                await self.close_position_live(symbol, side, filled_qty, reason="Protective order failed")
                return None

            return {
                "symbol": symbol,
                "side": side,
                "qty": float(filled_qty),
                "entry_price": float(price),
                "sl": float(sl),
                "tp": float(tp),
                "entry_order_id": entry_id,
                "sl_order_id": sl_id,
                "tp_order_id": tp_id,
                "status": "OPEN",
            }

        except Exception:
            return None

    async def _place_protective_orders(self, symbol: str, entry_side: str, qty: float, sl: float, tp: float):
        """Pasang SL/TP sebagai order reduceOnly.

        Catatan: Implementasi ccxt/binance futures bervariasi. Kita coba beberapa tipe/param.
        Return (sl_order_id, tp_order_id) atau (None, None).
        """
        opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'

        # Common params Binance futures
        base_params = {
            'reduceOnly': True,
            'timeInForce': 'GTC',
            'workingType': 'MARK_PRICE',
        }

        sl_id = None
        tp_id = None

        # --- STOP LOSS ---
        for order_type in ("STOP_MARKET", "stop_market", "STOP", "stop"):
            try:
                o = await self.loader.exchange.create_order(
                    symbol,
                    order_type,
                    opposite,
                    qty,
                    None,
                    {**base_params, 'stopPrice': float(sl)},
                )
                sl_id = o.get('id')
                if sl_id:
                    break
            except Exception:
                continue

        # --- TAKE PROFIT ---
        for order_type in ("TAKE_PROFIT_MARKET", "take_profit_market", "TAKE_PROFIT", "take_profit"):
            try:
                o = await self.loader.exchange.create_order(
                    symbol,
                    order_type,
                    opposite,
                    qty,
                    None,
                    {**base_params, 'stopPrice': float(tp)},
                )
                tp_id = o.get('id')
                if tp_id:
                    break
            except Exception:
                continue

        return sl_id, tp_id

    async def cancel_order(self, symbol: str, order_id: str):
        if Config.TRADING_MODE != "LIVE":
            return
        try:
            await self.loader.exchange.cancel_order(order_id, symbol)
        except Exception:
            pass

    async def close_position_live(self, symbol: str, entry_side: str, qty: float, reason: str = "Manual Close"):
        """Close posisi via market reduceOnly."""
        try:
            opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'
            await self.loader.exchange.create_order(
                symbol,
                'market',
                opposite,
                float(qty),
                None,
                {'reduceOnly': True},
            )
        except Exception:
            pass

    # ==========================
    # PAPER
    # ==========================
    def _place_paper_entry(self, symbol: str, side: str, qty: float, price: float, sl: float, tp: float):
        order_id = str(uuid.uuid4())[:8]
        trade_data = {
            "id": order_id,
            "symbol": symbol,
            "status": "OPEN",
            "side": side,
            "qty": float(qty),
            "entry_price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._update_paper_wallet(trade_data, add=True)
        return trade_data

    def _update_paper_wallet(self, trade_data, add: bool = True):
        try:
            with open(self.loader.paper_state_file, 'r') as f:
                data = json.load(f)
            if add:
                data['positions'][trade_data['id']] = trade_data
            with open(self.loader.paper_state_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    # ==========================
    # COMMON
    # ==========================
    async def close_position(self, trade_id: str, symbol: str, price: float, reason: str = "TP", *, side: Optional[str] = None, qty: Optional[float] = None):
        """Menutup posisi.

        - PAPER: update saldo & pindahkan ke history.
        - LIVE: market close jika side+qty diberikan.
        """
        if Config.TRADING_MODE == "LIVE":
            if side is not None and qty is not None:
                await self.close_position_live(symbol, side, qty, reason=reason)
            return

        # PAPER
        try:
            with open(self.loader.paper_state_file, 'r') as f:
                data = json.load(f)
            if trade_id not in data.get('positions', {}):
                return
            pos = data['positions'][trade_id]
            q = float(pos['qty'])
            entry = float(pos['entry_price'])
            side_p = pos.get('side', 'buy')

            if side_p in ('buy', 'LONG'):
                pnl = (float(price) - entry) * q
            else:
                pnl = (entry - float(price)) * q

            data['balance'] = float(data.get('balance', 0.0)) + float(pnl)
            pos['exit_price'] = float(price)
            pos['exit_reason'] = reason
            pos['pnl'] = float(pnl)
            pos['close_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pos['status'] = "CLOSED"
            data.setdefault('history', []).append(pos)
            del data['positions'][trade_id]
            with open(self.loader.paper_state_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass
