import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Callable

from config import Config


class ExecutionHandler:
    """Eksekusi order.

    Perubahan versi harden:
    - LIVE: setelah entry ter-fill, bot langsung pasang SL/TP (reduceOnly) agar posisi selalu terlindungi.
    - LIVE: ada timeout untuk limit order. Jika tidak fill, dibatalkan.
    - Semua operasi dibungkus try/except agar loop utama tidak crash.
    """

    def __init__(self, loader, on_pending_entry: Callable | None = None, on_pending_clear: Callable | None = None):
        self.loader = loader
        self.logger = logging.getLogger("neurobot.execution")
        self._orders_by_symbol: dict[str, set[str]] = {}
        self._on_pending_entry = on_pending_entry
        self._on_pending_clear = on_pending_clear

    def _log(self, msg: str, level: str = "INFO"):
        try:
            if level == "ERROR":
                self.logger.error(msg)
            elif level == "WARN":
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
        except Exception:
            pass

    def _track_order_id(self, symbol: str, order_id: Optional[str]):
        if not order_id:
            return
        sid = str(order_id)
        self._orders_by_symbol.setdefault(symbol, set()).add(sid)

    def _untrack_order_id(self, symbol: str, order_id: Optional[str]):
        if not order_id:
            return
        sid = str(order_id)
        ids = self._orders_by_symbol.get(symbol)
        if not ids:
            return
        ids.discard(sid)
        if not ids:
            self._orders_by_symbol.pop(symbol, None)

    async def _cancel_tracked_orders(self, symbol: str):
        order_ids = list(self._orders_by_symbol.get(symbol, set()))
        for oid in order_ids:
            try:
                await self.loader.exchange.cancel_order(oid, symbol)
                self._untrack_order_id(symbol, oid)
            except Exception as e:
                self._log(f"Cancel tracked order failed {symbol}:{oid}: {e}", "WARN")

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
            entry_id = None
            # set leverage (ignore error jika tidak support)
            try:
                await self.loader.exchange.set_leverage(int(Config.LEVERAGE), symbol)
            except Exception:
                pass

            # 1) kirim limit entry
            entry_params = self._entry_params(side)
            entry_order = await self.loader.exchange.create_order(symbol, 'limit', side, qty, price, entry_params)
            entry_id = entry_order.get('id')
            self._track_order_id(symbol, entry_id)
            self._notify_pending_entry(symbol, side, entry_id, price, sl, tp, qty)

            # 2) tunggu fill (timeout)
            filled_qty = float(entry_order.get('filled') or 0.0)
            status = (entry_order.get('status') or '').lower()
            final_statuses = {"closed", "filled", "canceled", "cancelled", "expired"}

            if status not in ("closed", "filled"):
                # polling fetch_order
                import asyncio
                import time
                start = time.time()
                while time.time() - start < float(Config.ENTRY_ORDER_TIMEOUT_SEC):
                    await asyncio.sleep(float(Config.ENTRY_ORDER_POLL_SEC))
                    try:
                        o = await self.loader.exchange.fetch_order(entry_id, symbol)
                        status = (o.get('status') or '').lower()
                        filled_qty = float(o.get('filled') or 0.0)
                        if status in ("closed", "filled"):
                            break
                    except Exception:
                        continue

            # kalau belum filled, cancel
            if filled_qty <= 0:
                try:
                    await self.loader.exchange.cancel_order(entry_id, symbol)
                    self._untrack_order_id(symbol, entry_id)
                except Exception as e:
                    self._log(f"Cancel entry failed {symbol}:{entry_id}: {e}", "WARN")
                    await self._cancel_tracked_orders(symbol)
                self._notify_pending_clear(symbol, side, entry_id)
                return None

            # jika partial fill dan order masih terbuka, cancel sisa agar tidak terisi tanpa proteksi
            orig_qty = float(qty)
            if filled_qty < orig_qty and status not in final_statuses:
                cancel_ok = False
                try:
                    await self.loader.exchange.cancel_order(entry_id, symbol)
                    self._untrack_order_id(symbol, entry_id)
                    cancel_ok = True
                except Exception as e:
                    self._log(f"Cancel partial entry failed {symbol}:{entry_id}: {e}", "WARN")
                    cancel_ok = False

                if cancel_ok:
                    try:
                        o = await self.loader.exchange.fetch_order(entry_id, symbol)
                        status = (o.get('status') or status).lower()
                        filled_qty = max(filled_qty, float(o.get('filled') or 0.0))
                    except Exception:
                        pass
                else:
                    await self._cancel_tracked_orders(symbol)
                    await self.close_position_live(symbol, side, filled_qty, reason="Entry cancel failed")
                    self._notify_pending_clear(symbol, side, entry_id)
                    return None

            # normalize filled qty
            filled_qty = self.loader.amount_to_precision(symbol, float(filled_qty))
            if filled_qty <= 0:
                self._notify_pending_clear(symbol, side, entry_id)
                return None
            self._untrack_order_id(symbol, entry_id)

            # 3) pasang SL/TP reduceOnly
            sl_id, tp_id = await self._place_protective_orders(symbol, side, filled_qty, sl, tp)
            if sl_id is None or tp_id is None:
                # jika gagal pasang proteksi, batalkan order yang sempat dibuat agar tidak orphan
                if sl_id:
                    await self.cancel_order(symbol, sl_id)
                if tp_id:
                    await self.cancel_order(symbol, tp_id)
                # jika gagal pasang proteksi, tutup posisi agar tidak naked
                self._log(f"Protective order failed {symbol} SL:{sl_id} TP:{tp_id}", "ERROR")
                await self.close_position_live(symbol, side, filled_qty, reason="Protective order failed")
                self._notify_pending_clear(symbol, side, entry_id)
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

        except Exception as e:
            self._log(f"Entry flow error {symbol}: {e}", "ERROR")
            self._notify_pending_clear(symbol, side, entry_id)
            return None

    async def _place_protective_orders(self, symbol: str, entry_side: str, qty: float, sl: float, tp: float):
        """Pasang SL/TP sebagai order reduceOnly.

        Catatan: Implementasi ccxt/binance futures bervariasi. Kita coba beberapa tipe/param.
        Return (sl_order_id, tp_order_id) atau (None, None).
        """
        opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'

        base_params = self._protective_params(entry_side=entry_side)

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
                    self._track_order_id(symbol, sl_id)
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
                    self._track_order_id(symbol, tp_id)
                    break
            except Exception:
                continue

        return sl_id, tp_id


    def _notify_pending_entry(self, symbol: str, side: str, entry_id: Optional[str], entry: float, sl: float, tp: float, qty: float):
        if not self._on_pending_entry or not entry_id:
            return
        try:
            self._on_pending_entry(symbol, side, str(entry_id), float(entry), float(sl), float(tp), float(qty))
        except Exception:
            return

    def _notify_pending_clear(self, symbol: str, side: str, entry_id: Optional[str]):
        if not self._on_pending_clear or not entry_id:
            return
        try:
            self._on_pending_clear(symbol, side, str(entry_id))
        except Exception:
            return

    def _position_side(self, entry_side: str) -> Optional[str]:
        if not getattr(Config, 'HEDGE_MODE', False):
            return None
        return "LONG" if entry_side.lower() == "buy" else "SHORT"

    def _entry_params(self, entry_side: str) -> dict:
        params = {}
        position_side = self._position_side(entry_side)
        if position_side:
            params['positionSide'] = position_side
        return params

    def _protective_params(self, entry_side: Optional[str] = None, existing_order: Optional[dict] = None) -> dict:
        params = {
            'reduceOnly': True,
            'timeInForce': 'GTC',
            'workingType': 'MARK_PRICE',
        }
        position_side = None
        if existing_order:
            info = existing_order.get('info', {}) or {}
            position_side = info.get('positionSide') or existing_order.get('positionSide')
        if not position_side and entry_side:
            position_side = self._position_side(entry_side)
        if position_side:
            params['positionSide'] = position_side
        return params

    async def _place_stop_loss_order(self, symbol: str, entry_side: str, qty: float, sl: float, order_type_hint: Optional[str] = None, existing_order: Optional[dict] = None):
        opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'
        base_params = self._protective_params(entry_side=entry_side, existing_order=existing_order)

        order_types = []
        if order_type_hint:
            order_types.append(order_type_hint)
        order_types.extend(["STOP_MARKET", "stop_market", "STOP", "stop"])

        seen = set()
        for order_type in order_types:
            if not order_type or order_type in seen:
                continue
            seen.add(order_type)
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
                    return sl_id
            except Exception:
                continue
        return None


    async def _place_take_profit_order(self, symbol: str, entry_side: str, qty: float, tp: float, order_type_hint: Optional[str] = None, existing_order: Optional[dict] = None):
        opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'
        base_params = self._protective_params(entry_side=entry_side, existing_order=existing_order)

        order_types = []
        if order_type_hint:
            order_types.append(order_type_hint)
        order_types.extend(["TAKE_PROFIT_MARKET", "take_profit_market", "TAKE_PROFIT", "take_profit"])

        seen = set()
        for order_type in order_types:
            if not order_type or order_type in seen:
                continue
            seen.add(order_type)
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
                    return tp_id
            except Exception:
                continue
        return None

    async def ensure_protective_orders(
        self,
        symbol: str,
        entry_side: str,
        qty: float,
        sl: Optional[float],
        tp: Optional[float],
        sl_order: Optional[dict] = None,
        tp_order: Optional[dict] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if Config.TRADING_MODE != "LIVE":
            return None, None
        try:
            qty_p = float(self.loader.amount_to_precision(symbol, float(qty)))
        except Exception:
            qty_p = float(qty)
        if qty_p <= 0:
            return None, None
        sl_id = None
        tp_id = None
        if sl_order is None and sl is not None:
            try:
                sl_val = float(sl)
            except Exception:
                sl_val = None
            if sl_val and sl_val > 0:
                sl_id = await self._place_stop_loss_order(symbol, entry_side, qty_p, sl_val, existing_order=sl_order)
                if sl_id:
                    self._track_order_id(symbol, sl_id)
        if tp_order is None and tp is not None:
            try:
                tp_val = float(tp)
            except Exception:
                tp_val = None
            if tp_val and tp_val > 0:
                tp_id = await self._place_take_profit_order(symbol, entry_side, qty_p, tp_val, existing_order=tp_order)
                if tp_id:
                    self._track_order_id(symbol, tp_id)
        return sl_id, tp_id

    async def update_sl_to_breakeven(self, symbol: str, entry_side: str, qty: float, new_sl: float, sl_order: Optional[dict] = None) -> bool:
        if Config.TRADING_MODE != "LIVE":
            return False

        sl_order_id = sl_order.get('id') if sl_order else None
        order_type = None
        sl_side = None
        amount = float(qty)
        params = self._protective_params(entry_side=entry_side, existing_order=sl_order)
        if sl_order:
            order_type = sl_order.get('type') or (sl_order.get('info', {}) or {}).get('type')
            sl_side = sl_order.get('side') or (sl_order.get('info', {}) or {}).get('side')
            amount = float(sl_order.get('amount') or (sl_order.get('info', {}) or {}).get('origQty') or qty)

        if sl_side is None:
            sl_side = 'sell' if entry_side.lower() == 'buy' else 'buy'

        exchange_has = getattr(self.loader.exchange, 'has', {}) or {}
        if sl_order_id and exchange_has.get('editOrder'):
            try:
                await self.loader.exchange.edit_order(
                    sl_order_id,
                    symbol,
                    order_type or "STOP_MARKET",
                    sl_side,
                    amount,
                    None,
                    {**params, 'stopPrice': float(new_sl)},
                )
                return True
            except Exception:
                pass

        new_sl_id = await self._place_stop_loss_order(symbol, entry_side, amount, new_sl, order_type_hint=order_type, existing_order=sl_order)
        if not new_sl_id:
            return False
        self._track_order_id(symbol, new_sl_id)

        if sl_order_id:
            try:
                await self.loader.exchange.cancel_order(sl_order_id, symbol)
                self._untrack_order_id(symbol, sl_order_id)
            except Exception:
                # rollback new SL to avoid double protection if old SL can't be canceled
                try:
                    await self.loader.exchange.cancel_order(new_sl_id, symbol)
                    self._untrack_order_id(symbol, new_sl_id)
                except Exception:
                    pass
                return False

        return True

    async def cancel_order(self, symbol: str, order_id: str):
        if Config.TRADING_MODE != "LIVE":
            return
        try:
            await self.loader.exchange.cancel_order(order_id, symbol)
            self._untrack_order_id(symbol, order_id)
        except Exception as e:
            self._log(f"Cancel order failed {symbol}:{order_id}: {e}", "WARN")

    async def close_position_live(self, symbol: str, entry_side: str, qty: float, reason: str = "Manual Close") -> bool:
        """Close posisi via market reduceOnly."""
        ok = False
        try:
            opposite = 'sell' if entry_side.lower() == 'buy' else 'buy'
            params = {'reduceOnly': True}
            position_side = self._position_side(entry_side)
            if position_side:
                params['positionSide'] = position_side
            await self.loader.exchange.create_order(
                symbol,
                'market',
                opposite,
                float(qty),
                None,
                params,
            )
            ok = True
            self._log(f"LIVE EXIT {symbol} {entry_side} reason={reason} qty={float(qty):.6f}", "WARN")
        except Exception as e:
            self._log(f"Close position failed {symbol} ({reason}): {e}", "ERROR")
        await self._cancel_tracked_orders(symbol)
        return ok

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
