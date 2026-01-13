import pandas as pd
import numpy as np

from config import Config

class SMCAnalyzer:
    """
    Modul Analisa Smart Money Concepts (SMC) & ICT.
    Fokus: Mendeteksi Market Structure Shift (MSS), Fair Value Gaps (FVG), 
    dan Order Blocks (OB).
    """

    def __init__(self):
        pass

    def identify_swings(self, df: pd.DataFrame, pivot_len: int = 3):
        """
        Mendeteksi Swing High dan Swing Low (Fractals).
        pivot_len = jumlah candle kiri/kanan yang harus lebih rendah/tinggi.
        """
        df = df.copy()
        
        # Logika Swing High: High candle ini > High pivot_len candle kiri & kanan
        df['swing_high'] = False
        df['swing_low'] = False

        # Menggunakan window rolling untuk cek local max/min
        # (Implementasi manual loop dioptimalkan dengan numpy untuk kecepatan)
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        # Array untu menyimpan harga swing
        swing_high_prices = np.full(n, np.nan)
        swing_low_prices = np.full(n, np.nan)

        for i in range(pivot_len, n - pivot_len):
            # Cek Swing High
            if all(highs[i] > highs[i-k] for k in range(1, pivot_len+1)) and \
               all(highs[i] > highs[i+k] for k in range(1, pivot_len+1)):
                df.at[df.index[i], 'swing_high'] = True
                swing_high_prices[i] = highs[i]

            # Cek Swing Low
            if all(lows[i] < lows[i-k] for k in range(1, pivot_len+1)) and \
               all(lows[i] < lows[i+k] for k in range(1, pivot_len+1)):
                df.at[df.index[i], 'swing_low'] = True
                swing_low_prices[i] = lows[i]

        df['sh_price'] = pd.Series(swing_high_prices, index=df.index).ffill()
        df['sl_price'] = pd.Series(swing_low_prices, index=df.index).ffill()
        
        return df

    def detect_fvg(self, df: pd.DataFrame):
        """
        Mendeteksi Fair Value Gap (Imbalance).
        Bullish FVG: Low candle[0] > High candle[2]
        Bearish FVG: High candle[0] < Low candle[2]
        """
        df['fvg_bull'] = False
        df['fvg_bear'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        # Shift data untuk perbandingan 3 candle (ICT concept)
        # Candle saat ini (0), Kemarin (-1), Dua hari lalu (-2)
        # FVG terbentuk di candle -1 (tengah)
        
        high = df['high']
        low = df['low']
        
        # Bullish FVG: Low candle sekarang > High 2 candle lalu
        # Kita cek pada candle yang sudah close (i-1 sebagai konfirmasi)
        bull_cond = (low > high.shift(2))
        
        # Bearish FVG: High candle sekarang < Low 2 candle lalu
        bear_cond = (high < low.shift(2))

        df.loc[bull_cond, 'fvg_bull'] = True
        df.loc[bull_cond, 'fvg_top'] = low            # Area atas gap
        df.loc[bull_cond, 'fvg_bottom'] = high.shift(2) # Area bawah gap

        df.loc[bear_cond, 'fvg_bear'] = True
        df.loc[bear_cond, 'fvg_top'] = low.shift(2)     # Area atas gap
        df.loc[bear_cond, 'fvg_bottom'] = high          # Area bawah gap

        return df

    def check_market_structure(self, df: pd.DataFrame):
        """
        Menentukan arah tren berdasarkan Break of Structure (BOS) / MSS.
        """
        # Ambil swing terakhir
        last_sh = df['sh_price'].iloc[-2] # Swing High terakhir (bukan candle curr)
        last_sl = df['sl_price'].iloc[-2] # Swing Low terakhir
        
        curr_close = df['close'].iloc[-1]
        
        trend = "NEUTRAL"
        
        # Logic MSS Simple:
        # Jika close menembus Swing High terakhir -> Bullish MSS
        if pd.notna(last_sh) and curr_close > last_sh:
            trend = "BULLISH"
            
        # Jika close menembus Swing Low terakhir -> Bearish MSS
        elif pd.notna(last_sl) and curr_close < last_sl:
            trend = "BEARISH"
            
        return trend, last_sh, last_sl

    def analyze(self, df: pd.DataFrame):
        """
        FUNGSI UTAMA yang dipanggil oleh bot.
        Menggabungkan Swing, FVG, dan MSS untuk mencari setup.
        """
        # Selalu return tuple (signal, setup, df_debug) agar caller tidak crash.
        if df is None or df.empty or len(df) < 50:
            return None, None, df

        # 1. Identifikasi Swing Points
        df = self.identify_swings(df, pivot_len=3)
        
        # 2. Identifikasi FVG
        df = self.detect_fvg(df)
        
        # 3. Cek Struktur Market (Trend)
        trend, last_sh, last_sl = self.check_market_structure(df)
        
        # 4. Cari Setup Entry
        # Kita melihat 3 candle terakhir untuk mendeteksi FVG yang baru terbentuk
        # atau FVG yang sedang di-retest.
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        signal = None
        setup_details = None

        # --- LOGIKA BULLISH SETUP (LONG) ---
        # Syarat: Trend Bullish (Break SH) + Ada Bullish FVG baru-baru ini
        if trend == "BULLISH":
            # Cek apakah ada FVG Bullish di 5 candle terakhir
            recent_fvg = df['fvg_bull'].iloc[-5:].any()
            
            if recent_fvg:
                # Entry di area FVG atau sedikit di bawah close
                entry_price = last_candle['close']
                # SL di bawah Swing Low terakhir (Struktur)
                sl_price = last_sl if pd.notna(last_sl) else last_candle['low'] * 0.99
                
                # Validasi jarak SL (jangan terlalu dekat/jauh)
                dist_pct = abs(entry_price - sl_price) / entry_price
                if 0.002 < dist_pct < 0.05: # Min 0.2%, Max 5% jarak SL
                    r = abs(entry_price - sl_price)
                    tp_price = entry_price + (Config.MIN_RR_RATIO * r)
                    signal = "LONG"
                    setup_details = {
                        "entry": float(entry_price),
                        "sl": float(sl_price),
                        "tp": float(tp_price),
                        "r": float(r),
                        "rr": float(Config.MIN_RR_RATIO),
                        "sh": float(last_sh) if pd.notna(last_sh) else None,
                        "sl_structure": float(last_sl) if pd.notna(last_sl) else None,
                        "reason": "Bullish MSS + FVG"
                    }

        # --- LOGIKA BEARISH SETUP (SHORT) ---
        # Syarat: Trend Bearish (Break SL) + Ada Bearish FVG
        elif trend == "BEARISH":
            # Cek apakah ada FVG Bearish di 5 candle terakhir
            recent_fvg = df['fvg_bear'].iloc[-5:].any()
            
            if recent_fvg:
                entry_price = last_candle['close']
                # SL di atas Swing High terakhir
                sl_price = last_sh if pd.notna(last_sh) else last_candle['high'] * 1.01
                
                dist_pct = abs(sl_price - entry_price) / entry_price
                if 0.002 < dist_pct < 0.05:
                    r = abs(sl_price - entry_price)
                    tp_price = entry_price - (Config.MIN_RR_RATIO * r)
                    signal = "SHORT"
                    setup_details = {
                        "entry": float(entry_price),
                        "sl": float(sl_price),
                        "tp": float(tp_price),
                        "r": float(r),
                        "rr": float(Config.MIN_RR_RATIO),
                        "sh": float(last_sh) if pd.notna(last_sh) else None,
                        "sl_structure": float(last_sl) if pd.notna(last_sl) else None,
                        "reason": "Bearish MSS + FVG"
                    }

        # Bisa jadi signal ada tetapi setup tidak lolos validasi jarak SL.
        if signal is None or setup_details is None:
            return None, None, df

        return signal, setup_details, df

# Unit Test (Bisa dijalankan langsung untuk cek logika)
if __name__ == "__main__":
    # Buat data dummy untuk test
    data = {
        'open': [100, 102, 101, 105, 108, 107, 110],
        'high': [103, 104, 103, 109, 110, 108, 112],
        'low':  [99, 101, 100, 104, 106, 105, 108],
        'close': [102, 101, 102, 108, 107, 106, 111],
        'volume': [1000]*7
    }
    df_test = pd.DataFrame(data)
    
    smc = SMCAnalyzer()
    sig, details, df_res = smc.analyze(df_test)
    
    print("DataFrame Hasil Analisa:")
    print(df_res[['close', 'swing_high', 'swing_low', 'fvg_bull', 'fvg_bear']].tail())
    print("\nSignal:", sig)
    print("Details:", details)