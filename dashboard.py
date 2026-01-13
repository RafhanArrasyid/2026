import os
import sys
from datetime import datetime
from config import Config

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"  # <--- INI YANG SEBELUMNYA KURANG

class Dashboard:
    def __init__(self):
        self.logs = []
        self.max_logs = 10

    def log(self, message, level="INFO"):
        """Menyimpan log untuk ditampilkan di dashboard"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Colors.GREEN if level == "INFO" else (Colors.YELLOW if level == "WARN" else Colors.RED)
        # Sekarang Colors.DIM sudah ada, jadi baris ini tidak akan error lagi
        self.logs.append(f"{Colors.DIM}{timestamp}{Colors.RESET} | {color}{level:<5}{Colors.RESET} | {message}")
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self, balance, active_positions, btc_status="STABLE"):
        self.clear_screen()
        
        # Header
        mode_color = Colors.GREEN if Config.TRADING_MODE == "LIVE" else Colors.CYAN
        header_width = 70
        title = "NEUROBOT V1 Trade"
        border = "+" + ("-" * (header_width - 2)) + "+"
        print(f"{Colors.HEADER}{Colors.BOLD}{border}{Colors.RESET}")
        print(f"{Colors.HEADER}| {title:^{header_width - 4}} |{Colors.RESET}")
        print(f"{Colors.HEADER}{border}{Colors.RESET}")

        try:
            bal = float(balance)
        except Exception:
            bal = 0.0
        total_pnl = 0.0
        for pos in active_positions:
            try:
                total_pnl += float(pos.get('u_pnl', 0.0))
            except Exception:
                continue
        equity = bal + total_pnl

        # Info Bar
        print(f" Mode    : {mode_color}{Config.TRADING_MODE}{Colors.RESET}")
        print(f" Balance : {Colors.GREEN}${bal:,.2f}{Colors.RESET}")
        print(f" Equity  : {Colors.GREEN}${equity:,.2f}{Colors.RESET}")
        print(f" BTC Stat: {Colors.YELLOW if btc_status != 'STABLE' else Colors.GREEN}{btc_status}{Colors.RESET}")
        print(f" Risk    : {Config.RISK_PER_TRADE*100}% | Pairs: {len(Config.PAIRS)}")
        print("-" * 70)

        # Active Positions Table
        print(f"{Colors.BOLD}ACTIVE POSITIONS ({len(active_positions)}/{Config.MAX_OPEN_POSITIONS}){Colors.RESET}")
        print(f"{'SYMBOL':<10} {'SIDE':<6} {'ENTRY':<10} {'PRICE':<10} {'PNL (Est)':<10} {'TP':<10} {'SL':<10}")
        print("-" * 70)

        if not active_positions:
            print(f"{Colors.DIM}No active positions running...{Colors.RESET}")
        else:
            for pos in active_positions:
                pnl_color = Colors.GREEN if pos['u_pnl'] >= 0 else Colors.RED
                side_color = Colors.GREEN if pos['side'].upper() == 'BUY' else Colors.RED
                print(f"{pos['symbol']:<10} {side_color}{pos['side'].upper():<6}{Colors.RESET} "
                      f"{pos['entry']:<10.4f} {pos['current']:<10.4f} "
                      f"{pnl_color}${pos['u_pnl']:<9.2f}{Colors.RESET} "
                      f"{pos['tp']:<10.4f} {pos['sl']:<10.4f}")

        print("-" * 70)
        
        # Recent Logs
        print(f"{Colors.BOLD}RECENT ACTIVITY LOGS{Colors.RESET}")
        for log in self.logs:
            print(log)
        print("-" * 70)
        print(f"{Colors.DIM}Press Ctrl+C to stop the bot safely.{Colors.RESET}")
