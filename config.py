import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ==========================================
    # 1. CORE SETTINGS & RUNTIME
    # ==========================================
    TRADING_MODE = "PAPER" 
    PAPER_INITIAL_BALANCE = 50.0  
    
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_SECRET")

    # --- SETTING RATE LIMIT (YANG BARU) ---
    # Scanner berjalan tiap 45-60 detik (Aman untuk TF 15m)
    LOOP_SLEEP_SEC = 45
    # Manager (Trailing) berjalan cepat (tiap 5 detik)
    MANAGER_SLEEP_SEC = 5 
    
    # Batasi concurrency agar tidak kena ban
    MAX_CONCURRENT_PAIRS = 2

    ENTRY_ORDER_TIMEOUT_SEC = 90
    ENTRY_ORDER_POLL_SEC = 2
    CORR_UPDATE_SEC = 60
    AI_RETRAIN_COOLDOWN_SEC = 3600
    LOG_FILE = os.getenv("BOT_LOG_FILE", "bot.log")

    # ==========================================
    # 2. RISK MANAGEMENT
    # ==========================================
    RISK_PER_TRADE = 0.02
    LEVERAGE = 10
    HEDGE_MODE = True
    MAX_OPEN_POSITIONS = 4
    MAX_TOTAL_RISK = 0.07
    HEDGE_SECOND_LEG_REDUCTION = 0.30
    MIN_RR_RATIO = 2.0          
    
    BREAK_EVEN_TRIGGER_R = 1.5
    BREAK_EVEN_OFFSET_R = 0.1
    TRAILING_ENABLED = False
    TRAILING_GAP_R = 0.5
    AUTO_RESTORE_PROTECTIVE = True
    PROTECTIVE_RESTORE_COOLDOWN_SEC = 30
    PROTECTIVE_INTENT_ENTRY_TOL_PCT = 0.01
    PROTECTIVE_INTENT_QTY_TOL_PCT = 0.10

    # ==========================================
    # 3. PAIRS & AI
    # ==========================================
    PAIRS = [
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "FTM/USDT",
        "SAND/USDT", "GALA/USDT", "VET/USDT", "GRT/USDT", "CHZ/USDT",
        "ONE/USDT", "HBAR/USDT", "ALGO/USDT", "EOS/USDT", "MANA/USDT",
        "ZIL/USDT", "KLAY/USDT", "BAT/USDT", "REN/USDT", "AUDIO/USDT"
    ]
    BTC_SYMBOL = "BTC/USDT"
    MAX_CORRELATION_BTC = 0.8

    TF_MACRO = '4h'
    TF_ENTRY = '15m'
    TRAINING_LOOKBACK_CANDLES = 1000
    MCPT_ITERATIONS = 500
    AI_ENABLED = True
    AI_CONFIDENCE_THRESHOLD = 0.52
    AI_LABEL_THRESHOLD = 0.005
    AI_P_VALUE_THRESHOLD = 0.2

    # ==========================================
    # 4. SMC FILTERS
    # ==========================================
    SMC_USE_HTF_FILTER = True
    SMC_HTF_PIVOT_LEN = 5
    SMC_USE_VOLUME_FILTER = True
    SMC_VOLUME_WINDOW = 20
    SMC_VOLUME_MIN_MULT = 1.0
    SMC_ALLOWED_UTC_HOURS = []
