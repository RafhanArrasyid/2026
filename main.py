import asyncio
import atexit
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from alerts import AlertManager
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

        self.alerts = AlertManager()

        if not getattr(Config, "AI_ENABLED", True):
            self._log("AI Disabled: validation skipped", "WARN")

        self.loader = ExchangeLoader()
        self.smc = SMCAnalyzer()
        self.brain = NeuroBrain()
        self.manager = RiskManager(self.loader)
        self.executor = ExecutionHandler(self.loader, on_pending_entry=self._set_pending_entry, on_pending_clear=self._clear_pending_entry)

        self.active_positions_display: List[dict] = []
        self.positions_by_symbol: Dict[str, dict] = {}
        self.open_orders_by_symbol: Dict[str, list] = {}
        self.active_pairs_correlation: Dict[str, float] = {}
        self.btc_df: pd.DataFrame = pd.DataFrame()
        self.trading_paused = False
        self._drawdown_triggered = False
        self._equity_high = 0.0
        self._drawdown_pause_until = 0.0
        self._lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(getattr(Config, 'LOCK_FILE', 'bot.lock')))
        self._lock_acquired = False
        self.pairs: List[str] = list(getattr(Config, "PAIRS", []))

        self.last_processed_ts: Dict[str, pd.Timestamp] = {}
        self.pair_sem = asyncio.Semaphore(int(Config.MAX_CONCURRENT_PAIRS))
        self.ai_sem = asyncio.Semaphore(1)
        self._last_btc_update = 0.0
        self._last_corr_update = 0.0
        self._last_balance: float = 0.0
        self._positions_fresh = False
        self._open_orders_fresh = False
        self._initial_r_by_key: Dict[str, float] = {}
        self._paper_initial_r_by_id: Dict[str, float] = {}
        self._intent_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_intents.json")
        self._pending_entries: Dict[str, dict] = {}
        self._intent_by_key: Dict[str, dict] = self._load_intents()
        self._last_protective_restore: Dict[str, float] = {}
        self._close_all_requested = False
        self._close_all_inflight = False
        self._stdin_thread = None

    def _start_stdin_listener(self) -> bool:
        try:
            if not sys.stdin or not sys.stdin.isatty():
                return False
        except Exception:
            return False
        if self._stdin_thread:
            return True

        def _stdin_loop():
            while True:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    break
                if line == "":
                    break
                cmd = line.strip().lower()
                if not cmd:
                    continue
                if cmd in ("close", "closeall", "close-all", "close all"):
                    self._close_all_requested = True
        t = threading.Thread(target=_stdin_loop, daemon=True)
        t.start()
        self._stdin_thread = t
        return True

    def _log(self, msg: str, level: str = "INFO"):
        self.dashboard.log(msg, level)
        try:
            if level == "ERROR": self.logger.error(msg)
            elif level == "WARN": self.logger.warning(msg)
            else: self.logger.info(msg)
        except Exception: pass
        try:
            if self.alerts:
                self.alerts.notify(level, msg)
        except Exception:
            pass

    def _normalize_side(self, side: str) -> str:
        s = (side or '').lower()
        if s in ('long', 'buy'):
            return 'buy'
        if s in ('short', 'sell'):
            return 'sell'
        return ''

    def _position_key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{side.lower()}"

    async def _validate_pairs(self):
        pairs = [p.strip() for p in getattr(Config, "PAIRS", []) if isinstance(p, str)]
        if not pairs:
            self._log("Config.PAIRS kosong atau tidak valid.", "ERROR")
            self.pairs = []
            return
        self._log("Validating pairs against exchange markets...", "INFO")
        try:
            await self.loader.ensure_markets()
            markets = getattr(self.loader.exchange, "markets", None) if self.loader.exchange else None
            if not markets:
                self._log("Pair validation skipped: markets unavailable.", "WARN")
                self.pairs = pairs
                return
            invalid = [p for p in pairs if p not in markets]
            invalid_set = set(invalid)
            valid = [p for p in pairs if p not in invalid_set]
            if invalid:
                self._log(f"Invalid pairs in Config.PAIRS: {', '.join(invalid)}", "ERROR")
                self._log("Invalid pairs will be skipped.", "WARN")
            self._log(f"Pair validation: {len(valid)} valid, {len(invalid)} invalid.", "INFO")
            self.pairs = valid
        except Exception as e:
            self._log(f"Pair validation failed: {e}", "WARN")
            self.pairs = pairs

    def _build_btc_corr_display(self) -> str:
        try:
            threshold = float(getattr(Config, "MAX_CORRELATION_BTC", 0.0))
        except Exception:
            threshold = 0.0
        items = []
        for key, corr in sorted(self.active_pairs_correlation.items(), key=lambda x: x[1], reverse=True):
            try:
                corr_val = float(corr)
            except Exception:
                continue
            if corr_val != corr_val:
                continue
            if corr_val <= threshold:
                continue
            if isinstance(key, str) and ':' in key:
                sym, side = key.rsplit(':', 1)
            else:
                sym, side = key, ''
            if side == 'buy':
                side_disp = 'BUY'
            elif side == 'sell':
                side_disp = 'SELL'
            else:
                side_disp = side.upper() if isinstance(side, str) else ''
            if side_disp:
                items.append(f"{sym} {side_disp} {corr_val:.2f}")
            else:
                items.append(f"{sym} {corr_val:.2f}")
        return ' | '.join(items) if items else 'none'

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
            if not (o.get('reduceOnly') or info.get('reduceOnly') or o.get('closePosition') or info.get('closePosition')):
                continue
            if self._order_position_side(o) != position_side:
                continue
            return o
        return None

    def _find_take_profit_order(self, symbol: str, position_side: str):
        orders = self.open_orders_by_symbol.get(symbol, [])
        for o in orders:
            o_type = (o.get('type') or '').upper()
            if 'TAKE_PROFIT' not in o_type:
                continue
            info = o.get('info', {}) or {}
            if not (o.get('reduceOnly') or info.get('reduceOnly') or o.get('closePosition') or info.get('closePosition')):
                continue
            if self._order_position_side(o) != position_side:
                continue
            return o
        return None

    def _load_intents(self) -> Dict[str, dict]:
        self._pending_entries = {}
        try:
            if not os.path.exists(self._intent_file):
                return {}
            with open(self._intent_file, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {}
            if 'positions' in data or 'pending_entries' in data:
                positions = data.get('positions') or {}
                pending = data.get('pending_entries') or {}
                out = {}
                for key, val in positions.items():
                    if isinstance(val, dict):
                        out[key] = val
                pend_out = {}
                for key, val in pending.items():
                    if isinstance(val, dict):
                        pend_out[key] = val
                self._pending_entries = pend_out
                return out
            out = {}
            for key, val in data.items():
                if isinstance(val, dict):
                    out[key] = val
            return out
        except Exception:
            return {}

    def _save_intents(self):
        try:
            tmp_path = self._intent_file + ".tmp"
            payload = {"positions": self._intent_by_key, "pending_entries": self._pending_entries}
            with open(tmp_path, 'w') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, self._intent_file)
        except Exception:
            pass

    def _pending_key(self, symbol: str, side: str, entry_order_id: str) -> str:
        return f"{symbol}:{side}:{entry_order_id}"

    def _set_pending_entry(self, symbol: str, side: str, entry_order_id: str, entry: float, sl: float, tp: float, qty: float):
        if not entry_order_id:
            return
        try:
            key = self._pending_key(symbol, side, str(entry_order_id))
            self._pending_entries[key] = {
                'symbol': symbol,
                'side': side,
                'entry_order_id': str(entry_order_id),
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'qty': float(qty),
                'ts': float(time.time()),
            }
            self._save_intents()
        except Exception:
            return

    def _clear_pending_entry(self, symbol: str, side: str, entry_order_id: str):
        if not entry_order_id:
            return
        try:
            key = self._pending_key(symbol, side, str(entry_order_id))
            if key in self._pending_entries:
                self._pending_entries.pop(key, None)
                self._save_intents()
        except Exception:
            return

    def _find_pending_entry_for_position(self, symbol: str, side: str, entry: float, qty: float):
        if not self._pending_entries:
            return None
        try:
            entry_tol = float(getattr(Config, 'PROTECTIVE_INTENT_ENTRY_TOL_PCT', 0.01))
        except Exception:
            entry_tol = 0.01
        try:
            qty_tol = float(getattr(Config, 'PROTECTIVE_INTENT_QTY_TOL_PCT', 0.10))
        except Exception:
            qty_tol = 0.10
        best_key = None
        best_item = None
        best_ts = -1.0
        for key, item in self._pending_entries.items():
            if not isinstance(item, dict):
                continue
            if item.get('symbol') != symbol or item.get('side') != side:
                continue
            try:
                p_entry = float(item.get('entry') or 0.0)
                p_qty = float(item.get('qty') or 0.0)
            except Exception:
                continue
            if p_entry <= 0 or p_qty <= 0:
                continue
            entry_diff = abs(float(entry) - p_entry) / p_entry if p_entry else 0.0
            if entry_diff > entry_tol:
                continue
            if float(qty) > (p_qty * (1.0 + qty_tol)):
                continue
            try:
                ts = float(item.get('ts') or 0.0)
            except Exception:
                ts = 0.0
            if ts >= best_ts:
                best_ts = ts
                best_key = key
                best_item = item
        if best_key and best_item:
            return best_key, best_item
        return None

    def _set_intent(self, key: str, entry: float, sl: float, tp: float, qty: float):
        try:
            self._intent_by_key[key] = {
                'entry': float(entry),
                'sl': float(sl),
                'tp': float(tp),
                'qty': float(qty),
                'ts': float(time.time()),
            }
            self._save_intents()
        except Exception:
            pass

    def _update_intent_sl(self, key: str, sl: float):
        intent = self._intent_by_key.get(key)
        if not intent:
            return
        try:
            intent['sl'] = float(sl)
            intent['ts'] = float(time.time())
            self._save_intents()
        except Exception:
            pass

    def _clear_intent(self, key: str):
        if key in self._intent_by_key:
            self._intent_by_key.pop(key, None)
            self._save_intents()

    def _get_initial_r(self, key: str, entry: float, current_sl: float, store: Dict[str, float]) -> Optional[float]:
        r = store.get(key)
        if r and r > 0:
            return r
        try:
            r = abs(float(entry) - float(current_sl))
        except Exception:
            return None
        if r > 0:
            store[key] = r
            return r
        return None

    def _calc_trailing_sl(
        self,
        side: str,
        entry: float,
        current: float,
        current_sl: float,
        risk_r: float,
    ) -> Optional[tuple[float, float, float]]:
        if risk_r <= 0:
            return None
        try:
            entry_f = float(entry)
            curr_f = float(current)
            sl_f = float(current_sl)
        except Exception:
            return None
        if sl_f <= 0:
            return None
        profit = curr_f - entry_f if side == 'buy' else entry_f - curr_f
        if profit <= 0:
            return None
        profit_r = profit / float(risk_r)
        base_gap = float(getattr(Config, 'TRAILING_GAP_R', 0.5))
        gap_r = base_gap
        if profit_r >= 2.5:
            gap_r = max(0.2, base_gap * (2.5 / profit_r))
        start_r = max(float(getattr(Config, 'BREAK_EVEN_TRIGGER_R', 1.0)), 1.0) + gap_r
        if profit_r < start_r:
            return None
        if side == 'buy':
            new_sl = curr_f - (gap_r * risk_r)
            if new_sl <= sl_f:
                return None
        else:
            new_sl = curr_f + (gap_r * risk_r)
            if new_sl >= sl_f:
                return None
        min_step = 0.1 * float(risk_r)
        if abs(new_sl - sl_f) < min_step:
            return None
        return float(new_sl), float(profit_r), float(gap_r)

    def _order_qty_mismatch(self, order: dict, position_qty: float) -> bool:
        try:
            pos_qty = float(position_qty)
        except Exception:
            return False
        if pos_qty <= 0:
            return False
        info = order.get('info', {}) or {}
        raw_qty = order.get('amount') or info.get('origQty') or info.get('origQty'.lower()) or info.get('executedQty')
        try:
            order_qty = float(raw_qty)
        except Exception:
            return False
        if order_qty <= 0:
            return False
        try:
            tol = float(getattr(Config, 'PROTECTIVE_ORDER_QTY_TOL_PCT', 0.10))
        except Exception:
            tol = 0.10
        if tol < 0:
            tol = 0.0
        return abs(order_qty - pos_qty) / pos_qty > tol

    def _calc_equity(self, balance: float) -> float:
        total_pnl = 0.0
        for pos in self.active_positions_display:
            try:
                total_pnl += float(pos.get('u_pnl', 0.0))
            except Exception:
                continue
        try:
            bal = float(balance)
        except Exception:
            bal = 0.0
        return bal + total_pnl

    def _update_drawdown(self, equity: float):
        try:
            eq = float(equity)
        except Exception:
            return
        if eq <= 0:
            return
        now_ts = time.time()
        if self.trading_paused:
            try:
                pause_until = float(self._drawdown_pause_until or 0.0)
            except Exception:
                pause_until = 0.0
            if pause_until > 0 and now_ts >= pause_until:
                self.trading_paused = False
                self._drawdown_triggered = False
                self._equity_high = eq
                self._drawdown_pause_until = 0.0
                self._log("Drawdown cooldown elapsed; trading resumed.", "WARN")
            return
        if self._equity_high <= 0 or eq > self._equity_high:
            self._equity_high = eq
            return
        try:
            max_dd = float(getattr(Config, 'MAX_DRAWDOWN_PCT', 0.0))
        except Exception:
            max_dd = 0.0
        if max_dd <= 0:
            return
        drawdown = (self._equity_high - eq) / self._equity_high
        if not self._drawdown_triggered and drawdown >= max_dd:
            self._drawdown_triggered = True
            self.trading_paused = True
            try:
                cooldown = float(getattr(Config, 'DRAWDOWN_COOLDOWN_SEC', 0.0))
            except Exception:
                cooldown = 0.0
            if cooldown < 0:
                cooldown = 0.0
            self._drawdown_pause_until = now_ts + cooldown if cooldown > 0 else 0.0
            if cooldown > 0:
                self._log(f"Drawdown limit hit ({drawdown:.2%}); trading paused for {cooldown:.0f}s.", "ERROR")
            else:
                self._log(f"Drawdown limit hit ({drawdown:.2%}); trading paused.", "ERROR")

    def _pid_alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    def _acquire_instance_lock(self) -> bool:
        if not getattr(Config, 'SINGLE_INSTANCE_LOCK', True):
            return True
        lock_path = self._lock_file
        try:
            if os.path.exists(lock_path):
                old_pid = None
                try:
                    with open(lock_path, 'r') as f:
                        old_pid = int((f.read() or '').strip() or 0)
                except Exception:
                    old_pid = None
                if old_pid and self._pid_alive(old_pid):
                    self._log(f"Another instance is running (pid={old_pid}).", "ERROR")
                    return False
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
            with open(lock_path, 'w') as f:
                f.write(str(os.getpid()))
            self._lock_acquired = True
            try:
                atexit.register(self._release_instance_lock)
            except Exception:
                pass
            return True
        except Exception as e:
            self._log(f"Lock file error: {e}", "WARN")
            return True

    def _release_instance_lock(self):
        if not self._lock_acquired:
            return
        try:
            if os.path.exists(self._lock_file):
                os.remove(self._lock_file)
        except Exception:
            pass
        self._lock_acquired = False

    async def _preflight_checks(self) -> bool:
        if Config.TRADING_MODE == "LIVE":
            if not Config.API_KEY or not Config.SECRET_KEY:
                self._log("Missing API key/secret for LIVE mode.", "ERROR")
                return False
        if not self.pairs:
            self._log("No valid pairs to trade.", "ERROR")
            return False
        try:
            await self.loader.ensure_markets()
            markets = getattr(self.loader.exchange, 'markets', None) if self.loader.exchange else None
            if markets:
                self._log(f"Markets loaded: {len(markets)} symbols.", "INFO")
            else:
                self._log("Markets not available; continuing.", "WARN")
        except Exception as e:
            self._log(f"Preflight markets check failed: {e}", "WARN")
        if Config.TRADING_MODE == "LIVE":
            try:
                bal = await self.loader.get_balance()
                self._log(f"Preflight balance: {float(bal):.2f}", "INFO")
            except Exception:
                self._log("Preflight balance check failed.", "WARN")
            try:
                pos = await self.loader.fetch_positions()
                if pos is None:
                    self._log("Preflight fetch_positions failed.", "WARN")
            except Exception:
                self._log("Preflight fetch_positions failed.", "WARN")
            try:
                orders = await self.loader.fetch_open_orders(symbol=None)
                if orders is None:
                    self._log("Preflight fetch_open_orders failed.", "WARN")
            except Exception:
                self._log("Preflight fetch_open_orders failed.", "WARN")
            await self._validate_account_mode_and_leverage()
        return True

    async def _validate_account_mode_and_leverage(self):
        if Config.TRADING_MODE != "LIVE":
            return
        if getattr(Config, 'PREFLIGHT_VALIDATE_MODE', True):
            try:
                if hasattr(self.loader.exchange, 'set_position_mode'):
                    await self.loader.exchange.set_position_mode(bool(Config.HEDGE_MODE))
                    mode = 'HEDGE' if Config.HEDGE_MODE else 'ONE-WAY'
                    self._log(f"Position mode ensured: {mode}", "INFO")
            except Exception as e:
                self._log(f"Position mode check failed: {e}", "WARN")
        if getattr(Config, 'PREFLIGHT_VALIDATE_LEVERAGE', True):
            ok = 0
            fail = 0
            for sym in self.pairs:
                try:
                    await self.loader.exchange.set_leverage(int(Config.LEVERAGE), sym)
                    ok += 1
                except Exception:
                    fail += 1
                await asyncio.sleep(0.05)
            if ok or fail:
                level = 'INFO' if fail == 0 else 'WARN'
                self._log(f"Leverage preflight: ok={ok} fail={fail}", level)

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
            changed = False

            def _close_paper_position(pid: str, price: float, reason: str):
                nonlocal changed
                pos = positions.get(pid)
                if not pos:
                    return
                q = float(pos['qty'])
                entry = float(pos['entry_price'])
                side_p = pos.get('side', 'buy')
                symbol = pos.get('symbol')

                if side_p in ('buy', 'BUY', 'LONG'):
                    pnl = (float(price) - entry) * q
                else:
                    pnl = (entry - float(price)) * q

                self._log(
                    f"PAPER EXIT {symbol} {side_p} reason={reason} entry={entry:.6f} exit={float(price):.6f} pnl={pnl:.4f}",
                    "WARN",
                )

                data['balance'] = float(data.get('balance', 0.0)) + float(pnl)
                pos['exit_price'] = float(price)
                pos['exit_reason'] = reason
                pos['pnl'] = float(pnl)
                pos['close_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pos['status'] = "CLOSED"
                data.setdefault('history', []).append(pos)
                positions.pop(pid, None)
                self._paper_initial_r_by_id.pop(str(pid), None)
                changed = True

            for pid, p in list(positions.items()):
                curr = await self.loader.get_current_price(p['symbol'])
                if curr is None:
                    continue
                entry, qty = float(p['entry_price']), float(p['qty'])
                side_raw = (p.get('side') or 'buy').upper()
                tp = float(p.get('tp'))
                sl = float(p.get('sl'))
                side_norm = 'buy' if side_raw in ('BUY', 'LONG') else 'sell'
                position_side = 'LONG' if side_norm == 'buy' else 'SHORT'

                risk_r = self._get_initial_r(str(pid), entry, sl, self._paper_initial_r_by_id)
                sl_updated = False

                if sl > 0:
                    if risk_r:
                        be_trigger_r = float(getattr(Config, 'BREAK_EVEN_TRIGGER_R', 1.5))
                        be_offset_r = max(0.0, float(getattr(Config, 'BREAK_EVEN_OFFSET_R', 0.0)))
                        profit = (curr - entry) if side_norm == 'buy' else (entry - curr)
                        profit_r = profit / float(risk_r) if risk_r else 0.0
                        if profit_r >= be_trigger_r:
                            if side_norm == 'buy':
                                target_sl = entry + (be_offset_r * risk_r)
                                if sl < target_sl:
                                    sl = target_sl
                                    sl_updated = True
                                    self._log(
                                        f"Paper BE+ {p['symbol']} {side_norm} -> {sl:.6f} ({profit_r:.2f}R)",
                                        "INFO",
                                    )
                            else:
                                target_sl = entry - (be_offset_r * risk_r)
                                if sl > target_sl:
                                    sl = target_sl
                                    sl_updated = True
                                    self._log(
                                        f"Paper BE+ {p['symbol']} {side_norm} -> {sl:.6f} ({profit_r:.2f}R)",
                                        "INFO",
                                    )

                    if getattr(Config, 'TRAILING_ENABLED', True) and risk_r:
                        trailing = self._calc_trailing_sl(side_norm, entry, curr, sl, risk_r)
                        if trailing:
                            new_sl, profit_r, gap_r = trailing
                            sl = new_sl
                            sl_updated = True
                            self._log(
                                f"Paper Trailing SL {p['symbol']} {side_norm} -> {sl:.6f} (gap {gap_r:.2f}R, pnl {profit_r:.2f}R)",
                                "INFO",
                            )

                if sl_updated:
                    p['sl'] = float(sl)
                    changed = True

                if side_raw in ['BUY', 'LONG']:
                    u_pnl = (curr - entry) * qty
                    if curr >= tp:
                        _close_paper_position(pid, curr, "TP Hit")
                        continue
                    if curr <= sl:
                        reason = "BE+ Hit" if sl > entry else "SL Hit"
                        _close_paper_position(pid, curr, reason)
                        continue
                else:
                    u_pnl = (entry - curr) * qty
                    if curr <= tp:
                        _close_paper_position(pid, curr, "TP Hit")
                        continue
                    if curr >= sl:
                        reason = "BE+ Hit" if sl < entry else "SL Hit"
                        _close_paper_position(pid, curr, reason)
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

            if changed:
                with open(self.loader.paper_state_file, 'w') as f:
                    json.dump(data, f, indent=4)

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
        self._positions_fresh = not positions_fetch_failed
        prev_keys = set(self.positions_by_symbol.keys())

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

        self._open_orders_fresh = (not open_orders_fetch_failed) and (not positions_fetch_failed)

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
            closed_keys = prev_keys - set(pos_by_symbol.keys())
            if closed_keys:
                for key in closed_keys:
                    self._clear_intent(key)
                    self._initial_r_by_key.pop(key, None)

            if not open_orders_fetch_failed and self._pending_entries:
                try:
                    open_entry_ids = set()
                    for orders in all_open_orders.values():
                        for o in orders:
                            info = o.get('info', {}) or {}
                            if o.get('reduceOnly') or info.get('reduceOnly') or o.get('closePosition') or info.get('closePosition'):
                                continue
                            oid = o.get('id')
                            if oid:
                                open_entry_ids.add(str(oid))
                    active_pairs = {(p.get('symbol'), p.get('side')) for p in pos_by_symbol.values() if p.get('symbol') and p.get('side')}
                    removed = []
                    for key, item in list(self._pending_entries.items()):
                        if not isinstance(item, dict):
                            continue
                        sym = item.get('symbol')
                        side = item.get('side')
                        entry_id = str(item.get('entry_order_id') or '')
                        if (sym, side) in active_pairs:
                            continue
                        if entry_id and entry_id in open_entry_ids:
                            continue
                        removed.append(key)
                        self._pending_entries.pop(key, None)
                    if removed:
                        self._save_intents()
                except Exception:
                    pass

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
                if self.trading_paused:
                    return
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
                    if Config.TRADING_MODE == "LIVE":
                        key = self._position_key(symbol, side)
                        self._set_intent(key, res.get('entry_price', setup['entry']), res.get('sl', setup['sl']), res.get('tp', setup['tp']), res.get('qty', qty))

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
                    info = order.get('info', {}) or {}
                    return bool(order.get('reduceOnly') or info.get('reduceOnly') or order.get('closePosition') or info.get('closePosition'))

                sl_order = next(
                    (
                        o for o in orders
                        if o.get('type', '').upper() in ['STOP', 'STOP_MARKET']
                        and _is_reduce_only(o)
                        and self._order_position_side(o) == position_side
                    ),
                    None,
                )
                tp_order = self._find_take_profit_order(sym, position_side)

                if sl_order and self._order_qty_mismatch(sl_order, qty):
                    try:
                        self._log(f"Protective SL qty mismatch {sym}; refreshing.", "WARN")
                        oid = sl_order.get('id') if isinstance(sl_order, dict) else None
                        if oid:
                            await self.executor.cancel_order(sym, oid)
                    except Exception:
                        pass
                    sl_order = None

                if tp_order and self._order_qty_mismatch(tp_order, qty):
                    try:
                        self._log(f"Protective TP qty mismatch {sym}; refreshing.", "WARN")
                        oid = tp_order.get('id') if isinstance(tp_order, dict) else None
                        if oid:
                            await self.executor.cancel_order(sym, oid)
                    except Exception:
                        pass
                    tp_order = None

                if getattr(Config, 'AUTO_RESTORE_PROTECTIVE', True) and self._open_orders_fresh and self._positions_fresh:
                    missing_sl = sl_order is None
                    missing_tp = tp_order is None
                    if missing_sl or missing_tp:
                        pos_key = self._position_key(sym, side)
                        intent = self._intent_by_key.get(pos_key)
                        if not intent and getattr(Config, 'RESTORE_FROM_PENDING', True):
                            pending_match = self._find_pending_entry_for_position(sym, side, entry_price, qty)
                            if pending_match:
                                p_key, p_item = pending_match
                                try:
                                    self._set_intent(pos_key, p_item.get('entry'), p_item.get('sl'), p_item.get('tp'), p_item.get('qty'))
                                    self._clear_pending_entry(sym, side, p_item.get('entry_order_id'))
                                    intent = self._intent_by_key.get(pos_key)
                                    self._log(f"Pending entry matched {sym}; restoring protections.", "WARN")
                                except Exception:
                                    intent = self._intent_by_key.get(pos_key)
                        now_ts = time.time()
                        cooldown = float(getattr(Config, 'PROTECTIVE_RESTORE_COOLDOWN_SEC', 30))
                        last_try = self._last_protective_restore.get(pos_key, 0.0)
                        if now_ts - last_try >= cooldown:
                            self._last_protective_restore[pos_key] = now_ts
                            if intent:
                                intent_ok = True
                                try:
                                    intent_entry = float(intent.get('entry') or 0.0)
                                except Exception:
                                    intent_entry = 0.0
                                try:
                                    intent_qty = float(intent.get('qty') or 0.0)
                                except Exception:
                                    intent_qty = 0.0
                                try:
                                    entry_tol = float(getattr(Config, 'PROTECTIVE_INTENT_ENTRY_TOL_PCT', 0.01))
                                except Exception:
                                    entry_tol = 0.01
                                try:
                                    qty_tol = float(getattr(Config, 'PROTECTIVE_INTENT_QTY_TOL_PCT', 0.10))
                                except Exception:
                                    qty_tol = 0.10

                                if intent_entry > 0:
                                    entry_diff = abs(entry_price - intent_entry) / intent_entry
                                    if entry_diff > entry_tol:
                                        intent_ok = False
                                if intent_qty > 0:
                                    if qty > (intent_qty * (1.0 + qty_tol)):
                                        intent_ok = False

                                if not intent_ok:
                                    self._log(
                                        f"Protective restore skipped {sym} intent mismatch entry={intent_entry:.6f} qty={intent_qty:.6f}",
                                        "WARN",
                                    )
                                else:
                                    sl_price = intent.get('sl')
                                    tp_price = intent.get('tp')
                                    sl_id, tp_id = await self.executor.ensure_protective_orders(
                                        sym,
                                        'BUY' if side == 'buy' else 'SELL',
                                        qty,
                                        sl_price,
                                        tp_price,
                                        sl_order=sl_order,
                                        tp_order=tp_order,
                                    )
                                    if sl_id or tp_id:
                                        self._log(
                                            f"Restored protective orders {sym} missing_sl={missing_sl} missing_tp={missing_tp}",
                                            "WARN",
                                        )
                                    else:
                                        self._log(
                                            f"Protective restore failed {sym} missing_sl={missing_sl} missing_tp={missing_tp}",
                                            "WARN",
                                        )
                            else:
                                self._log(
                                    f"Missing protective orders {sym} but no intent found; skip restore",
                                    "WARN",
                                )

                if sl_order:
                    sl_info = sl_order.get('info', {}) or {}
                    current_sl = float(
                        sl_order.get('stopPrice')
                        or sl_order.get('stop')
                        or sl_info.get('stopPrice')
                        or sl_info.get('stop')
                        or 0.0
                    )
                    pos_key = self._position_key(sym, side)
                    risk_r = self._get_initial_r(pos_key, entry_price, current_sl, self._initial_r_by_key)
                    sl_updated = False

                    # --- LOGIKA BREAK EVEN ADAPTIF (R-based) ---
                    if risk_r:
                        be_trigger_r = float(getattr(Config, 'BREAK_EVEN_TRIGGER_R', 1.5))
                        be_offset_r = max(0.0, float(getattr(Config, 'BREAK_EVEN_OFFSET_R', 0.0)))
                        profit = (curr_price - entry_price) if side == 'buy' else (entry_price - curr_price)
                        profit_r = profit / float(risk_r) if risk_r else 0.0
                        if profit_r >= be_trigger_r:
                            if side == 'buy':
                                target_sl = entry_price + (be_offset_r * risk_r)
                                if current_sl < target_sl:
                                    self._log(f"Moving SL to BE+ for {sym} ({profit_r:.2f}R)", "WARN")
                                    ok = await self.executor.update_sl_to_breakeven(sym, 'BUY', qty, target_sl, sl_order=sl_order)
                                    if ok:
                                        current_sl = target_sl
                                        sl_updated = True
                                        self._update_intent_sl(pos_key, current_sl)
                                    else:
                                        self._log(f"Failed to move SL to BE for {sym}", "WARN")
                            else:
                                target_sl = entry_price - (be_offset_r * risk_r)
                                if current_sl > target_sl:
                                    self._log(f"Moving SL to BE+ for {sym} ({profit_r:.2f}R)", "WARN")
                                    ok = await self.executor.update_sl_to_breakeven(sym, 'SELL', qty, target_sl, sl_order=sl_order)
                                    if ok:
                                        current_sl = target_sl
                                        sl_updated = True
                                        self._update_intent_sl(pos_key, current_sl)
                                    else:
                                        self._log(f"Failed to move SL to BE for {sym}", "WARN")

                    if not sl_updated and getattr(Config, 'TRAILING_ENABLED', True) and risk_r:
                        trailing = self._calc_trailing_sl(side, entry_price, curr_price, current_sl, risk_r)
                        if trailing:
                            new_sl, profit_r, gap_r = trailing
                            ok = await self.executor.update_sl_to_breakeven(
                                sym,
                                'BUY' if side == 'buy' else 'SELL',
                                qty,
                                new_sl,
                                sl_order=sl_order,
                            )
                            if ok:
                                self._update_intent_sl(pos_key, new_sl)
                                self._log(
                                    f"Trailing SL {sym} {side} -> {new_sl:.6f} (gap {gap_r:.2f}R, pnl {profit_r:.2f}R)",
                                    "WARN",
                                )
                            else:
                                self._log(f"Failed to trail SL for {sym}", "WARN")

            except Exception as e:
                self._log(f"Manager error {sym}: {e}", "ERROR")

    async def _close_all_positions(self):
        prev_paused = self.trading_paused
        self.trading_paused = True
        self._log("Manual CLOSE-ALL triggered; closing positions and canceling orders.", "WARN")

        canceled = 0
        cancel_failed = 0
        closed = 0
        close_failed = 0

        if Config.TRADING_MODE == "LIVE":
            orders = await self.loader.fetch_open_orders(symbol=None)
            if orders is None:
                orders = []
                for items in self.open_orders_by_symbol.values():
                    orders.extend(items)
            for o in orders:
                sym = o.get('symbol') or (o.get('info', {}) or {}).get('symbol')
                oid = o.get('id') or (o.get('info', {}) or {}).get('orderId')
                if not sym or not oid:
                    continue
                try:
                    await self.executor.cancel_order(sym, oid)
                    canceled += 1
                except Exception:
                    cancel_failed += 1

            positions = await self.loader.fetch_positions()
            if positions is None:
                positions = []
                for p in self.positions_by_symbol.values():
                    if p.get('symbol'):
                        positions.append(p)

            for p in positions:
                try:
                    if 'qty' in p and p.get('symbol') and p.get('side'):
                        sym = p.get('symbol')
                        qty = float(p.get('qty') or 0.0)
                        side = p.get('side')
                    else:
                        sym = p.get('symbol') or p.get('info', {}).get('symbol')
                        if not sym:
                            continue
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
                    if qty <= 0:
                        continue
                    ok = await self.executor.close_position_live(sym, side, qty, reason="Manual Close All")
                    if ok:
                        closed += 1
                    else:
                        close_failed += 1
                except Exception:
                    close_failed += 1
        else:
            try:
                with open(self.loader.paper_state_file, 'r') as f:
                    data = json.load(f)
                positions = data.get('positions', {}) or {}
            except Exception:
                positions = {}
            for pid, pos in list(positions.items()):
                sym = pos.get('symbol')
                if not sym:
                    continue
                price = await self.loader.get_current_price(sym)
                if price is None:
                    try:
                        price = float(pos.get('entry_price') or 0.0)
                    except Exception:
                        price = 0.0
                try:
                    await self.executor.close_position(pid, sym, float(price), reason="Manual Close All")
                    closed += 1
                except Exception:
                    close_failed += 1

        self._intent_by_key = {}
        self._pending_entries = {}
        self._initial_r_by_key = {}
        self._paper_initial_r_by_id = {}
        self._save_intents()

        self._log(
            f"Close-all done: closed={closed} canceled_orders={canceled} "
            f"cancel_failed={cancel_failed} close_failed={close_failed}",
            "WARN",
        )
        self.trading_paused = prev_paused

    async def task_market_scanner(self):
        """Loop Lambat: Scanner Entry (Hemat API)"""
        while True:
            try:
                bal = await self.loader.get_balance()
                self._last_balance = float(bal)
                await self._update_market_data()
                await self._update_active_pairs_correlation()
                risk_display = ""
                if bal and float(bal) > 0:
                    risk_used = self._estimate_total_risk(float(bal))
                    max_risk = float(bal) * float(getattr(Config, 'MAX_TOTAL_RISK', 0.07))
                    remaining = max(max_risk - risk_used, 0.0)
                    risk_display = f"Risk Used: {risk_used:.2f}/{max_risk:.2f} (rem {remaining:.2f})"
                
                equity = self._calc_equity(bal)
                self._update_drawdown(equity)

                # Render Dashboard di loop scanner
                corr_display = self._build_btc_corr_display()
                self.dashboard.render(bal, self.active_positions_display, btc_corr_display=corr_display, risk_display=risk_display)

                if self.trading_paused:
                    await asyncio.sleep(float(Config.LOOP_SLEEP_SEC))
                    continue
                
                tasks = [self._process_pair(pair) for pair in self.pairs]
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
                if self._close_all_requested and not self._close_all_inflight:
                    self._close_all_inflight = True
                    try:
                        await self._close_all_positions()
                    finally:
                        self._close_all_requested = False
                        self._close_all_inflight = False

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
        if not self._acquire_instance_lock():
            return
        await self._validate_pairs()
        ok = await self._preflight_checks()
        if not ok:
            self._release_instance_lock()
            return
        self.alerts.start()
        stdin_ok = self._start_stdin_listener()
        if stdin_ok:
            self._log("Console command ready: type 'close' + Enter to close all positions/orders.", "INFO")
        self._log("Bot Started. Running Hybrid Loops...", "INFO")
        try:
            # Jalankan 2 task parallel
            await asyncio.gather(
                self.task_market_scanner(),
                self.task_position_manager()
            )
        finally:
            self._release_instance_lock()

if __name__ == "__main__":
    bot = NeuroBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        sys.exit(0)
