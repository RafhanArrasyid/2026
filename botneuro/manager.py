import pandas as pd
from config import Config

class RiskManager:
    """
    Bertanggung jawab atas Money Management & Risk Guard.
    Menghitung Position Sizing berdasarkan Risk % dan menjaga eksposur korelasi.
    """

    def __init__(self, loader):
        self.loader = loader

    async def check_rules(self, symbol, open_positions_count, active_pairs_correlation, df_btc, df_symbol):
        """
        Cek apakah trade diperbolehkan berdasarkan aturan ketat.
        """
        # 1. Cek Slot Maksimal
        if open_positions_count >= Config.MAX_OPEN_POSITIONS:
            print(f"[RISK] Reject {symbol}: Max positions reached ({Config.MAX_OPEN_POSITIONS}).")
            return False

        # 2. Cek Korelasi BTC (Correlation Guard)
        # Jika korelasi pair ini dengan BTC > 0.8, dan kita sudah punya posisi lain
        # yang juga berkorelasi tinggi dengan BTC, maka HATI-HATI.
        corr = self._calculate_correlation(df_btc, df_symbol)
        
        # Hitung berapa banyak posisi aktif yang juga berkorelasi tinggi dgn BTC
        high_corr_count = sum(1 for c in active_pairs_correlation.values() if c > Config.MAX_CORRELATION_BTC)
        
        # Batasan: Maksimal 2 posisi yang "mengekor" BTC secara ketat
        if corr > Config.MAX_CORRELATION_BTC and high_corr_count >= 2:
            print(f"[RISK] Reject {symbol}: Too much BTC exposure (Corr: {corr:.2f}).")
            return False

        return True

    async def calculate_size(self, symbol, entry_price, sl_price):
        """
        THE GOLDEN RULE: Position Sizing Calculator.
        Risk Amount = Balance * Risk % (misal 2%)
        Qty = Risk Amount / Jarak SL
        """
        # Pastikan markets loaded supaya precision/limits valid
        try:
            await self.loader.ensure_markets()
        except Exception:
            pass

        balance = await self.loader.get_balance()
        
        if balance <= 0:
            return 0.0

        # 1. Tentukan Uang yang Siap Hilang (Risk Amount)
        risk_amount = balance * Config.RISK_PER_TRADE  # Contoh: $1000 * 2% = $20

        # 2. Hitung Jarak SL per koin
        # Contoh: Entry $0.50, SL $0.48 -> Jarak $0.02
        sl_distance = abs(entry_price - sl_price)
        
        if sl_distance == 0:
            return 0.0

        # 3. Hitung Quantity (Jumlah Koin)
        # Qty = $20 / $0.02 = 1000 Koin
        quantity = risk_amount / sl_distance
        
        # 4. Validasi Nominal Value (Notional)
        notional_value = quantity * entry_price
        
        # Cek apakah leverage cukup?
        # Max Notional = Balance * Leverage
        max_notional = balance * Config.LEVERAGE
        
        if notional_value > max_notional:
            # Jika size terlalu besar untuk leverage kita, kecilkan size-nya
            quantity = max_notional / entry_price
            print(f"[RISK] Size adjusted by Leverage limit for {symbol}")

        # Precision + limits (minQty / minNotional)
        quantity = float(self.loader.amount_to_precision(symbol, quantity))
        limits = self.loader.get_market_limits(symbol)
        amount_min = limits.get('amount_min')
        if amount_min is not None and quantity < float(amount_min):
            return 0.0
        cost_min = limits.get('cost_min')
        if cost_min is not None and (quantity * entry_price) < float(cost_min):
            return 0.0
        return float(quantity)

    def _calculate_correlation(self, df_btc, df_symbol):
        """Korelasi Pearson berbasis return agar lebih stabil."""
        if df_btc is None or df_symbol is None or df_btc.empty or df_symbol.empty:
            return 0.0
        len_data = min(len(df_btc), len(df_symbol), 200)
        if len_data < 30:
            return 0.0
        btc_ret = df_btc['close'].pct_change().iloc[-len_data:].dropna().reset_index(drop=True)
        sym_ret = df_symbol['close'].pct_change().iloc[-len_data:].dropna().reset_index(drop=True)
        n = min(len(btc_ret), len(sym_ret))
        if n < 30:
            return 0.0
        return float(btc_ret.iloc[-n:].corr(sym_ret.iloc[-n:]))