import asyncio
import logging
import sys
import time
from typing import Dict, List
import pandas as pd
from config import Config
from loader import ExchangeLoader
from smc import SMCAnalyzer
from brain import NeuroBrain
from manager import RiskManager
from execution import ExecutionHandler
from dashboard import Dashboard

class NeuroBot:
    def __init__(self):
        self.dashboard = Dashboard()
        self.dashboard.log("Initializing Modules...", "INFO")

        self.logger = logging.getLogger("neurobot")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if Config.LOG_FILE:
            try:
                fh = logging.FileHandler(Config.LOG_FILE)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                self.logger.addHandler(fh)
            except Exception:
                pass

        if not getattr(Config, "AI_ENABLED", True):
            self._log("AI Disabled: validation skipped", "WARN")

        self.loader = ExchangeLoader()
        self.smc = SMCAnalyzer()
        self.brain = NeuroBrain()
        self.manager = RiskManager(self.loader)
        self.executor = ExecutionHandler(self.loader)

        self.active_positions_display: List[dict] = []
        self.positions_by_symbol: Dict[str, dict] = {}
        self.open_orders_by_symbol: Dict[str, list] = {}
        self.active_pairs_correlation: Dict[str, float] = {}
        self.btc_df: pd.DataFrame = pd.DataFrame()

        self.last_processed_ts: Dict[str, pd.Timestamp] = {}
        self.pair_sem = asyncio.Semaphore(int(Config.MAX_CONCURRENT_PAIRS))
        self.ai_sem = asyncio.Semaphore(1)
        self._last_btc_update = 0.0
        self._last_corr_update = 0.0
        self._last_balance: float = 0.0

    def _log(self, msg: str, level: str = "INFO"):
        self.dashboard.log(msg, level)
        try:
            if level == "ERROR": self.logger.error(msg)
            elif level == "WARN": self.logger.warning(msg)
            else: self.logger.info(msg)
        except Exception: pass

    def _normalize_side(self, side: str) -> str:
        s = (side or '').lower()
        if s in ('long', 'buy'):
            return 'buy'
        if s in ('short', 'sell'):
            return 'sell'
        return ''

    def _position_key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{side.lower()}"

    def _order_position_side(self, order: dict) -> str:
        info = order.get('info', {}) or {}
        ps = (order.get('positionSide') or info.get('positionSide') or '').upper()
        if ps:
            return ps
        o_side = (order.get('side') or info.get('side') or '').lower()
        if o_side == 'sell':
            return 'LONG'
        if o_side == 'buy':
            return 'SHORT'
        return ''

    def _find_stop_loss_order(self, symbol: str, position_side: str):
        orders = self.open_orders_by_symbol.get(symbol, [])
        for o in orders:
            o_type = (o.get('type') or '').upper()
            if o_type not in ['STOP', 'STOP_MARKET']:
                continue
            info = o.get('info', {}) or {}
            if not (o.get('reduceOnly') or info.get('reduceOnly')):
                continue
            if self._order_position_side(o) != position_side:
                continue
            return o
        return None

    def _estimate_total_risk(self, balance: float) -> float:
        total_risk = 0.0
        default_risk = float(balance) * float(Config.RISK_PER_TRADE)
        for pos in self.positions_by_symbol.values():
            entry = float(pos.get('entry_price') or 0.0)
            qty = float(pos.get('qty') or 0.0)
            if entry <= 0 or qty <= 0:
                total_risk += default_risk
                continue
            position_side = (pos.get('positionSide') or ('LONG' if pos.get('side') == 'buy' else 'SHORT')).upper()
            sl_order = self._find_stop_loss_order(pos.get('symbol'), position_side)
            sl_price = None
            if sl_order:
                sl_price = sl_order.get('stopPrice') or sl_order.get('stop')
            try:
                sl_price = float(sl_price) if sl_price is not None else None
            except Exception:
                sl_price = None

            if sl_price is None or sl_price <= 0:
                total_risk += default_risk
                continue

            total_risk += abs(entry - sl_price) * qty
        return float(total_risk)

    def _has_open_entry_order(self, symbol: str, side: str) -> bool:
        orders = self.open_orders_by_symbol.get(symbol, [])
        for o in orders:
            info = o.get('info', {}) or {}
            if o.get('reduceOnly') or info.get('reduceOnly') or o.get('closePosition') or info.get('closePosition'):
                continue
            ps = (o.get('positionSide') or info.get('positionSide') or '').upper()
            if ps:
                if ps == 'LONG' and side == 'buy':
                    return True
                if ps == 'SHORT' and side == 'sell':
                    return True
            o_side = (o.get('side') or info.get('side') or '').lower()
            if o_side and o_side == side:
                return True
        return False

    def _extract_raw_position_qty(self, pos: dict) -> float:
        """Extract signed position size from ccxt position payload.

        Note: contractSize is not a position size and must not be used as qty.
        """
        candidates = []
        for key in ("contracts", "amount"):
            val = pos.get(key)
            if val is not None:
                candidates.append(val)
        info = pos.get('info', {}) or {}
        for key in ("positionAmt", "positionAmt".lower()):
            if key in info:
                candidates.append(info.get(key))

        fallback = 0.0
        for val in candidates:
            try:
                fval = float(val)
            except Exception:
                continue
            fallback = fval
            if fval != 0.0:
                return fval
        return float(fallback)

    async def _update_market_data(self):
        now = time.time()
        if now - self._last_btc_update < 60: return
        self._last_btc_update = now
        try:
            self.btc_df = await self.loader.fetch_candles(Config.BTC_SYMBOL, Config.TF_ENTRY, limit=300)
        except Exception: pass

    async def _update_active_pairs_correlation(self):
        now = time.time()
        if now - self._last_corr_update < float(getattr(Config, 'CORR_UPDATE_SEC', 60)):
            return
        self._last_corr_update = now

        if self.btc_df is None or self.btc_df.empty:
            self.active_pairs_correlation = {}
            return

        active_positions = [p for p in self.positions_by_symbol.values() if p.get('symbol') and p.get('side')]
        if not active_positions:
            self.active_pairs_correlation = {}
            return

        corrs_by_side: Dict[str, List[tuple[str, float]]] = {'buy': [], 'sell': []}
        for pos in active_positions:
            sym = pos.get('symbol')
            side = pos.get('side')
            if not sym or side not in ('buy', 'sell'):
                continue
            if sym == Config.BTC_SYMBOL:
                corr = 1.0
            else:
                df_sym = await self.loader.fetch_candles(sym, Config.TF_ENTRY, limit=200)
                corr = self.manager._calculate_correlation(self.btc_df, df_sym)
            key = self._position_key(sym, side)
            corrs_by_side[side].append((key, float(corr)))

        active_corrs: Dict[str, float] = {}
        for side, items in corrs_by_side.items():
            items.sort(key=lambda item: item[1], reverse=True)
            for key, corr in items[:2]:
                active_corrs[key] = corr
        self.active_pairs_correlation = active_corrs

    async def _sync_positions(self):
        if Config.TRADING_MODE == "PAPER":
            await self._sync_positions_paper()
        else:
            await self._sync_positions_live()

    async def _sync_positions_paper(self):
        import json
        try:
            with open(self.loader.paper_state_file, 'r') as f: data = json.load(f)
            display_pos = []
            pos_map: Dict[str, dict] = {}
            positions = data.get('positions', {}) or {}
            for pid, p in list(positions.items()):
                curr = await self.loader.get_current_price(p['symbol'])
                if curr is None:
                    continue
                entry, qty = float(p['entry_price']), float(p['qty'])
                side_raw = (p.get('side') or 'buy').upper()
                tp, sl = float(p.get('tp')), float(p.get('sl'))
                side_norm = 'buy' if side_raw in ('BUY', 'LONG') else 'sell'
                position_side = 'LONG' if side_norm == 'buy' else 'SHORT'

                if side_raw in ['BUY', 'LONG']:
                    u_pnl = (curr - entry) * qty
                    if curr >= tp:
                        await self.executor.close_position(pid, p['symbol'], curr, "TP Hit")
                        continue
                    if curr <= sl:
                        await self.executor.close_position(pid, p['symbol'], curr, "SL Hit")
                        continue
                else:
                    u_pnl = (entry - curr) * qty
                    if curr <= tp:
                        await self.executor.close_position(pid, p['symbol'], curr, "TP Hit")
                        continue
                    if curr >= sl:
                        await self.executor.close_position(pid, p['symbol'], curr, "SL Hit")
                        continue

                display_pos.append({'symbol': p['symbol'], 'side': side_raw, 'entry': entry, 'current': curr, 'u_pnl': u_pnl, 'tp': tp, 'sl': sl})
                key = self._position_key(p['symbol'], side_norm)
                pos_map[key] = {
                    'symbol': p['symbol'],
                    'side': side_norm,
                    'positionSide': position_side,
                    'qty': float(qty),
                    'entry_price': float(entry),
                    'current_price': float(curr),
                }
            
            self.active_positions_display = display_pos
            self.positions_by_symbol = pos_map
            self.open_orders_by_symbol = {}
        except Exception:
            self.active_positions_display = []

    async def _sync_positions_live(self):
        # 1. Fetch Positions
        positions = await self.loader.fetch_positions()
        pos_by_symbol: Dict[str, dict] = {}
        display = []
        positions_fetch_failed = positions is None

        for p in positions or []:
            try:
                sym = p.get('symbol') or p.get('info', {}).get('symbol')
                if not sym: continue
                
                raw_qty = self._extract_raw_position_qty(p)
                qty = abs(raw_qty)
                if qty <= 0:
                    continue

                info = p.get('info', {}) or {}
                side = self._normalize_side(p.get('side') or '')
                pos_side = (info.get('positionSide') or '').upper()
                if not side and pos_side:
                    side = 'buy' if pos_side == 'LONG' else 'sell'
                if not side:
                    side = 'buy' if raw_qty > 0 else 'sell'
                position_side = pos_side or ('LONG' if side == 'buy' else 'SHORT')
                
                entry = float(p.get('entryPrice') or p.get('average') or 0.0)
                mark = float(p.get('markPrice') or p.get('last') or p.get('info', {}).get('markPrice') or 0.0)
                
                u_pnl = float(p.get('unrealizedPnl') or 0.0)
                if u_pnl == 0.0 and entry and mark:
                    signed_qty = raw_qty
                    if signed_qty == 0.0:
                        signed_qty = qty if side == 'buy' else -qty
                    u_pnl = (mark - entry) * signed_qty

                key = self._position_key(sym, side)
                pos_by_symbol[key] = {'symbol': sym, 'side': side, 'positionSide': position_side, 'qty': qty, 'entry_price': entry, 'current_price': mark}
                display.append({'symbol': sym, 'side': 'BUY' if side == 'buy' else 'SELL', 'entry': entry, 'current': mark, 'u_pnl': u_pnl, 'tp': 0.0, 'sl': 0.0})
            except Exception: continue

        # 2. Bulk Fetch Open Orders (HEMAT API)
        all_open_orders: Dict[str, list] = {}
        open_orders_fetch_failed = False
        try:
            all_orders = await self.loader.fetch_open_orders(symbol=None)
            if all_orders is None:
                open_orders_fetch_failed = True
            else:
                for o in all_orders:
                    s = o.get('symbol')
                    if s:
                        if s not in all_open_orders:
                            all_open_orders[s] = []
                        all_open_orders[s].append(o)
        except Exception as e:
            open_orders_fetch_failed = True
            self._log(f"Bulk fetch error: {e}", "WARN")

        # 3. Cleanup orphan protective orders only (no open position)
        if not positions_fetch_failed and not open_orders_fetch_failed:
            try:
                def _is_reduce_only(order):
                    info = order.get('info', {}) or {}
                    return bool(
                        order.get('reduceOnly')
                        or info.get('reduceOnly')
                        or order.get('closePosition')
                        or info.get('closePosition')
                    )

                symbols_with_positions = {p.get('symbol') for p in pos_by_symbol.values() if p.get('symbol')}
                for sym, orders in list(all_open_orders.items()):
                    if sym not in symbols_with_positions and orders:
                        protectives = [o for o in orders if _is_reduce_only(o)]
                        if not protectives:
                            continue
                        canceled = 0
                        for o in protectives:
                            oid = o.get('id')
                            if not oid:
                                continue
                            try:
                                await self.loader.exchange.cancel_order(oid, sym)
                                canceled += 1
                            except Exception as e:
                                self._log(f"Cancel orphan order failed {sym}:{oid}: {e}", "WARN")
                        remaining = [o for o in orders if not _is_reduce_only(o)]
                        if remaining:
                            all_open_orders[sym] = remaining
                        else:
                            all_open_orders.pop(sym, None)
                        if canceled:
                            self._log(f"Canceled orphan protective orders for {sym} ({canceled})", "WARN")
            except Exception as e:
                self._log(f"Orphan cleanup error: {e}", "WARN")
        else:
            if positions_fetch_failed:
                self._log("Positions fetch failed; skipping orphan cleanup to avoid canceling protections.", "WARN")
            if open_orders_fetch_failed:
                self._log("Open orders fetch failed; skipping orphan cleanup to avoid canceling protections.", "WARN")

        if not positions_fetch_failed:
            self.positions_by_symbol = pos_by_symbol
            if not open_orders_fetch_failed:
                self.open_orders_by_symbol = all_open_orders
            self.active_positions_display = display

    def _has_exposure(self, symbol: str) -> bool:
        return any(p.get('symbol') == symbol for p in self.positions_by_symbol.values())

    def _has_exposure_side(self, symbol: str, side: str) -> bool:
        side_norm = self._normalize_side(side)
        if side_norm and self._has_open_entry_order(symbol, side_norm):
            return True
        return any(
            p.get('symbol') == symbol and p.get('side') == side_norm
            for p in self.positions_by_symbol.values()
        )

    async def _process_pair(self, symbol: str):
        # Fix Syntax: Tambahkan kurung tutup
        await asyncio.sleep(0.5)
        
        async with self.pair_sem:
            try:
                # Scan limit
                scan_limit = min(350, int(Config.TRAINING_LOOKBACK_CANDLES))
                df = await self.loader.fetch_candles(symbol, Config.TF_ENTRY, limit=scan_limit)
                if df is None or df.empty: return

                last_ts = df['timestamp'].iloc[-1]
                prev_ts = self.last_processed_ts.get(symbol)
                if prev_ts is not None and last_ts <= prev_ts: return
                self.last_processed_ts[symbol] = last_ts

                signal, setup, _ = self.smc.analyze(df)
                if not signal or not setup: return

                side = 'buy' if signal == 'LONG' else 'sell'
                if self._has_exposure_side(symbol, side):
                    return

                if getattr(Config, 'SMC_USE_HTF_FILTER', False):
                    df_macro = await self.loader.fetch_candles(symbol, Config.TF_MACRO, limit=300)
                    if not self.smc.passes_htf_filter(signal, df_macro):
                        self._log(f"SMC HTF Filter Rejected {symbol} {signal}", "INFO")
                        return

                self._log(f"SMC Signal {symbol}: {signal}", "INFO")

                ai_prob = None
                ai_p_value = None
                ai_real_score = None
                if getattr(Config, "AI_ENABLED", True):
                    # AI Validation
                    df_ai = df
                    if scan_limit < int(Config.TRAINING_LOOKBACK_CANDLES):
                        # --- PERBAIKAN: Cek eksplisit .empty ---
                        fetched_ai = await self.loader.fetch_candles(symbol, Config.TF_ENTRY, limit=int(Config.TRAINING_LOOKBACK_CANDLES))
                        if fetched_ai is not None and not fetched_ai.empty:
                            df_ai = fetched_ai
                        else:
                            df_ai = df
    
    
                    async with self.ai_sem:
                        mcpt_ok = await asyncio.to_thread(self.brain.mcpt_validation, symbol, df_ai)
                        if not mcpt_ok:
                            st = self.brain.states.get(symbol)
                            if st:
                                self._log(
                                    f"AI Rejected {symbol} (MCPT) p={st.last_p_value:.4f} real={st.last_real_score:.3f}",
                                    "INFO",
                                )
                            else:
                                self._log(f"AI Rejected {symbol} (MCPT) p=n/a real=n/a", "INFO")
                            return
    
                        prob = await asyncio.to_thread(self.brain.predict, symbol, df_ai, signal)
                    if prob < float(Config.AI_CONFIDENCE_THRESHOLD):
                        self._log(f"AI Reject {symbol} after SMC {signal} (Low Conf {prob:.2%})", "INFO")
                        return
                    st = self.brain.states.get(symbol)
                    ai_prob = float(prob)
                    if st:
                        ai_p_value = float(st.last_p_value)
                        ai_real_score = float(st.last_real_score)
                        self._log(
                            f"AI Pass {symbol} {signal} prob={ai_prob:.2%} p={ai_p_value:.4f} real={ai_real_score:.3f}",
                            "INFO",
                        )
                    else:
                        self._log(f"AI Pass {symbol} {signal} prob={ai_prob:.2%} p=n/a real=n/a", "INFO")
    
                can_trade = await self.manager.check_rules(symbol, side, len(self.positions_by_symbol), self.active_pairs_correlation, self.btc_df, df)
                if not can_trade: return

                balance = self._last_balance if self._last_balance > 0 else await self.loader.get_balance()
                total_risk = self._estimate_total_risk(balance)
                max_risk = float(balance) * float(getattr(Config, 'MAX_TOTAL_RISK', 0.07))
                remaining_risk = max_risk - total_risk
                if remaining_risk <= 0:
                    self._log(f"Risk cap reached {symbol}: {total_risk:.2f}/{max_risk:.2f}", "INFO")
                    return

                opp_side = 'sell' if side == 'buy' else 'buy'
                opp_pos = next(
                    (p for p in self.positions_by_symbol.values() if p.get('symbol') == symbol and p.get('side') == opp_side),
                    None,
                )
                qty_cap = None
                if opp_pos:
                    reduction = float(getattr(Config, 'HEDGE_SECOND_LEG_REDUCTION', 0.30))
                    qty_cap = max(float(opp_pos.get('qty') or 0.0) * (1.0 - reduction), 0.0)

                qty = await self.manager.calculate_size(
                    symbol,
                    setup['entry'],
                    setup['sl'],
                    max_risk_amount=remaining_risk,
                    qty_cap=qty_cap,
                    balance_override=balance,
                )
                if qty <= 0: return

                res = await self.executor.place_entry(symbol, side, qty, setup['entry'], setup['sl'], setup['tp'])
                if res:
                    base_msg = (
                        f"ORDER SENT: {symbol} {side} {qty} "
                        f"entry={setup['entry']:.6f} sl={setup['sl']:.6f} tp={setup['tp']:.6f}"
                    )
                    if ai_prob is not None:
                        base_msg += f" ai_prob={ai_prob:.2%}"
                    self._log(base_msg, "WARN")

            except Exception as e:
                self._log(f"Loop error {symbol}: {e}", "ERROR")

    async def _manage_active_positions(self):
        """Logic Trailing Stop / Break Even untuk Live Trading (Fast Loop)"""
        if Config.TRADING_MODE != "LIVE": return

        # Gunakan cache positions yang sudah di-update oleh _sync_positions_live
        # untuk menghindari request berlebih
        for pos in self.positions_by_symbol.values():
            try:
                sym = pos['symbol']
                entry_price = float(pos['entry_price'])
                curr_price = float(pos['current_price'])
                side = pos['side'].lower()
                qty = pos['qty']
                position_side = (pos.get('positionSide') or ('LONG' if side == 'buy' else 'SHORT')).upper()

                # Cari order SL aktif di cache memory
                orders = self.open_orders_by_symbol.get(sym, [])
                def _is_reduce_only(order):
                    return bool(order.get('reduceOnly') or (order.get('info', {}) or {}).get('reduceOnly'))

                sl_order = next(
                    (
                        o for o in orders
                        if o.get('type', '').upper() in ['STOP', 'STOP_MARKET']
                        and _is_reduce_only(o)
                        and self._order_position_side(o) == position_side
                    ),
                    None,
                )

                if sl_order:
                    current_sl = float(sl_order.get('stopPrice') or sl_order.get('stop') or 0.0)
                    
                    # --- LOGIKA BREAK EVEN SIMPEL ---
                    # Jika profit sudah > 1% (misal), geser SL ke Entry
                    if side == 'buy':
                        pnl_pct = (curr_price - entry_price) / entry_price
                        if pnl_pct > 0.01 and current_sl < entry_price:
                            self._log(f"Moving SL to BE for {sym}", "WARN")
                            ok = await self.executor.update_sl_to_breakeven(sym, 'BUY', qty, entry_price, sl_order=sl_order)
                            if not ok:
                                self._log(f"Failed to move SL to BE for {sym}", "WARN")

                    elif side == 'sell':
                        pnl_pct = (entry_price - curr_price) / entry_price
                        if pnl_pct > 0.01 and current_sl > entry_price:
                            self._log(f"Moving SL to BE for {sym}", "WARN")
                            ok = await self.executor.update_sl_to_breakeven(sym, 'SELL', qty, entry_price, sl_order=sl_order)
                            if not ok:
                                self._log(f"Failed to move SL to BE for {sym}", "WARN")

            except Exception as e:
                self._log(f"Manager error {sym}: {e}", "ERROR")

    async def task_market_scanner(self):
        """Loop Lambat: Scanner Entry (Hemat API)"""
        while True:
            try:
                bal = await self.loader.get_balance()
                self._last_balance = float(bal)
                await self._update_market_data()
                await self._update_active_pairs_correlation()
                if bal and float(bal) > 0:
                    risk_used = self._estimate_total_risk(float(bal))
                    max_risk = float(bal) * float(getattr(Config, 'MAX_TOTAL_RISK', 0.07))
                    remaining = max(max_risk - risk_used, 0.0)
                    self._log(
                        f"Risk used: {risk_used:.2f}/{max_risk:.2f} (rem {remaining:.2f})",
                        "INFO",
                    )
                
                # Render Dashboard di loop scanner
                self.dashboard.render(bal, self.active_positions_display)
                
                tasks = [self._process_pair(pair) for pair in Config.PAIRS]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Sleep lama sesuai config baru (misal 45s)
                await asyncio.sleep(float(Config.LOOP_SLEEP_SEC))
            except Exception as e:
                self._log(f"Scanner crash: {e}", "ERROR")
                await asyncio.sleep(5)

    async def task_position_manager(self):
        """Loop Cepat: Monitor Posisi & Trailing (Responsif)"""
        while True:
            try:
                # Update posisi (Bulk Fetch)
                await self._sync_positions()
                
                # Cek Trailing Stop / BE
                await self._manage_active_positions()
                
                # Sleep sebentar (misal 5s)
                await asyncio.sleep(getattr(Config, 'MANAGER_SLEEP_SEC', 5))
            except Exception as e:
                self._log(f"Manager crash: {e}", "ERROR")
                await asyncio.sleep(5)

    async def run(self):
        self._log("Bot Started. Running Hybrid Loops...", "INFO")
        # Jalankan 2 task parallel
        await asyncio.gather(
            self.task_market_scanner(),
            self.task_position_manager()
        )

if __name__ == "__main__":
    bot = NeuroBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        sys.exit(0)
