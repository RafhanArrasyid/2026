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

        self.loader = ExchangeLoader()
        self.smc = SMCAnalyzer()
        self.brain = NeuroBrain()
        self.manager = RiskManager(self.loader)
        self.executor = ExecutionHandler(self.loader)

        self.active_positions_display: List[dict] = []
        self.positions_by_symbol: Dict[str, dict] = {}
        self.open_orders_by_symbol: Dict[str, list] = {}
        self.btc_df: pd.DataFrame = pd.DataFrame()

        self.last_processed_ts: Dict[str, pd.Timestamp] = {}
        self.pair_sem = asyncio.Semaphore(int(Config.MAX_CONCURRENT_PAIRS))
        self._last_btc_update = 0.0

    def _log(self, msg: str, level: str = "INFO"):
        self.dashboard.log(msg, level)
        try:
            if level == "ERROR": self.logger.error(msg)
            elif level == "WARN": self.logger.warning(msg)
            else: self.logger.info(msg)
        except Exception: pass

    async def _update_market_data(self):
        now = time.time()
        if now - self._last_btc_update < 60: return
        self._last_btc_update = now
        try:
            self.btc_df = await self.loader.fetch_candles(Config.BTC_SYMBOL, Config.TF_ENTRY, limit=300)
        except Exception: pass

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
            positions = data.get('positions', {}) or {}
            for pid, p in list(positions.items()):
                curr = await self.loader.get_current_price(p['symbol'])
                if curr is None: continue
                entry, qty = float(p['entry_price']), float(p['qty'])
                side = (p.get('side') or 'buy').upper()
                tp, sl = float(p.get('tp')), float(p.get('sl'))

                if side in ['BUY', 'LONG']:
                    u_pnl = (curr - entry) * qty
                    if curr >= tp: await self.executor.close_position(pid, p['symbol'], curr, "TP Hit"); continue
                    if curr <= sl: await self.executor.close_position(pid, p['symbol'], curr, "SL Hit"); continue
                else:
                    u_pnl = (entry - curr) * qty
                    if curr <= tp: await self.executor.close_position(pid, p['symbol'], curr, "TP Hit"); continue
                    if curr >= sl: await self.executor.close_position(pid, p['symbol'], curr, "SL Hit"); continue

                display_pos.append({'symbol': p['symbol'], 'side': side, 'entry': entry, 'current': curr, 'u_pnl': u_pnl, 'tp': tp, 'sl': sl})
            
            self.active_positions_display = display_pos
            self.positions_by_symbol = {p['symbol']: p for p in display_pos}
            self.open_orders_by_symbol = {}
        except Exception:
            self.active_positions_display = []

    async def _sync_positions_live(self):
        # 1. Fetch Positions
        positions = await self.loader.fetch_positions()
        pos_by_symbol: Dict[str, dict] = {}
        display = []

        for p in positions or []:
            try:
                sym = p.get('symbol') or p.get('info', {}).get('symbol')
                if not sym: continue
                
                contracts = float(p.get('contracts') or p.get('contractSize') or p.get('amount') or 0.0)
                if abs(contracts) <= 0: continue

                side = (p.get('side') or '').lower()
                if not side: side = 'long' if contracts > 0 else 'short'
                
                entry = float(p.get('entryPrice') or p.get('average') or 0.0)
                mark = float(p.get('markPrice') or p.get('last') or p.get('info', {}).get('markPrice') or 0.0)
                
                u_pnl = float(p.get('unrealizedPnl') or 0.0)
                if u_pnl == 0.0 and entry and mark:
                    u_pnl = (mark - entry) * contracts if side in ('long', 'buy') else (entry - mark) * abs(contracts)

                pos_by_symbol[sym] = {'symbol': sym, 'side': 'buy' if side in ('long','buy') else 'sell', 'qty': abs(contracts), 'entry_price': entry, 'current_price': mark}
                display.append({'symbol': sym, 'side': 'BUY' if side in ('long','buy') else 'SELL', 'entry': entry, 'current': mark, 'u_pnl': u_pnl, 'tp': 0.0, 'sl': 0.0})
            except Exception: continue

        # 2. Bulk Fetch Open Orders (HEMAT API)
        all_open_orders: Dict[str, list] = {}
        try:
            all_orders = await self.loader.fetch_open_orders(symbol=None)
            for o in all_orders:
                s = o.get('symbol')
                if s: 
                    if s not in all_open_orders: all_open_orders[s] = []
                    all_open_orders[s].append(o)
        except Exception as e:
            self._log(f"Bulk fetch error: {e}", "WARN")

        self.positions_by_symbol = pos_by_symbol
        self.open_orders_by_symbol = all_open_orders
        self.active_positions_display = display

    def _has_exposure(self, symbol: str) -> bool:
        return (symbol in self.positions_by_symbol) or (symbol in self.open_orders_by_symbol and self.open_orders_by_symbol[symbol])

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

                if self._has_exposure(symbol): return

                signal, setup, _ = self.smc.analyze(df)
                if not signal or not setup: return

                # AI Validation
                df_ai = df
                if scan_limit < int(Config.TRAINING_LOOKBACK_CANDLES):
                    # --- PERBAIKAN: Cek eksplisit .empty ---
                    fetched_ai = await self.loader.fetch_candles(symbol, Config.TF_ENTRY, limit=int(Config.TRAINING_LOOKBACK_CANDLES))
                    if fetched_ai is not None and not fetched_ai.empty:
                        df_ai = fetched_ai
                    else:
                        df_ai = df

                self._log(f"SMC Signal {symbol}: {signal}", "INFO")

                if not self.brain.mcpt_validation(symbol, df_ai):
                    self._log(f"AI Rejected {symbol} (MCPT)", "INFO")
                    return
                
                prob = self.brain.predict(symbol, df_ai, direction=signal)
                if prob < float(Config.AI_CONFIDENCE_THRESHOLD):
                    self._log(f"AI Low Conf {symbol}: {prob:.2%}", "INFO")
                    return

                can_trade = await self.manager.check_rules(symbol, len(self.positions_by_symbol), {}, self.btc_df, df)
                if not can_trade: return

                qty = await self.manager.calculate_size(symbol, setup['entry'], setup['sl'])
                if qty <= 0: return

                side = 'buy' if signal == 'LONG' else 'sell'
                res = await self.executor.place_entry(symbol, side, qty, setup['entry'], setup['sl'], setup['tp'])
                if res: self._log(f"ORDER SENT: {symbol} {side} {qty}", "WARN")

            except Exception as e:
                self._log(f"Loop error {symbol}: {e}", "ERROR")

    async def _manage_active_positions(self):
        """Logic Trailing Stop / Break Even untuk Live Trading (Fast Loop)"""
        if Config.TRADING_MODE != "LIVE": return

        # Gunakan cache positions yang sudah di-update oleh _sync_positions_live
        # untuk menghindari request berlebih
        for sym, pos in self.positions_by_symbol.items():
            try:
                entry_price = float(pos['entry_price'])
                curr_price = float(pos['current_price'])
                side = pos['side'].lower()
                qty = pos['qty']

                # Cari order SL aktif di cache memory
                orders = self.open_orders_by_symbol.get(sym, [])
                sl_order = next((o for o in orders if o.get('type', '').upper() in ['STOP', 'STOP_MARKET'] and o.get('reduceOnly')), None)

                if sl_order:
                    current_sl = float(sl_order.get('stopPrice') or sl_order.get('stop') or 0.0)
                    
                    # --- LOGIKA BREAK EVEN SIMPEL ---
                    # Jika profit sudah > 1% (misal), geser SL ke Entry
                    if side == 'buy':
                        pnl_pct = (curr_price - entry_price) / entry_price
                        if pnl_pct > 0.01 and current_sl < entry_price:
                            self._log(f"Moving SL to BE for {sym}", "WARN")
                            await self.executor.cancel_order(sym, sl_order['id'])
                            # Pasang SL baru di Entry Price
                            await self.executor._place_protective_orders(sym, 'BUY', qty, entry_price, 0) # TP 0 artinya biarkan TP lama (kurang sempurna tapi cukup aman)

                    elif side == 'sell':
                        pnl_pct = (entry_price - curr_price) / entry_price
                        if pnl_pct > 0.01 and current_sl > entry_price:
                            self._log(f"Moving SL to BE for {sym}", "WARN")
                            await self.executor.cancel_order(sym, sl_order['id'])
                            await self.executor._place_protective_orders(sym, 'SELL', qty, entry_price, 0)

            except Exception as e:
                self._log(f"Manager error {sym}: {e}", "ERROR")

    async def task_market_scanner(self):
        """Loop Lambat: Scanner Entry (Hemat API)"""
        while True:
            try:
                bal = await self.loader.get_balance()
                await self._update_market_data()
                
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