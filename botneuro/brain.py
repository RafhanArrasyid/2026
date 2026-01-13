import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

from config import Config


@dataclass
class ModelState:
    model: RandomForestClassifier
    scaler: StandardScaler
    last_fit_ts: float
    last_mcpt_ts: float
    last_real_score: float
    last_p_value: float


class NeuroBrain:
    """AI validator + predictor.

    Perubahan penting dibanding versi awal:
    - Label jadi 3 kelas: DOWN(-1), FLAT(0), UP(1) supaya SHORT tidak "dites" oleh model long-bias.
    - MCPT menghitung skor pada validation set yang sama (bukan training score) dengan metrik yang sama.
    - Ada cooldown retrain supaya CPU stabil saat bot 24/7.
    """

    FEATURE_COLS = ['log_ret', 'atr_pct', 'rsi', 'dist_ema50', 'vol_chg']

    def __init__(self):
        self.states: dict[str, ModelState] = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / df['close']
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['dist_ema50'] = (df['close'] - df['ema50']) / df['ema50']
        df['vol_chg'] = df['volume'].pct_change()
        df.dropna(inplace=True)
        return df

    def _make_labels(self, df_feat: pd.DataFrame, horizon: int = 3) -> pd.Series:
        """3-class labeling.

        UP   : future_return >= +threshold
        DOWN : future_return <= -threshold
        FLAT : otherwise
        """
        future_close = df_feat['close'].shift(-horizon)
        current_close = df_feat['close']
        future_ret = (future_close / current_close) - 1.0
        thr = float(Config.AI_LABEL_THRESHOLD)
        y = pd.Series(np.zeros(len(df_feat), dtype=int), index=df_feat.index)
        y[future_ret >= thr] = 1
        y[future_ret <= -thr] = -1
        return y

    def _time_split(self, X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
        split = int(len(X) * train_frac)
        if split <= 50 or len(X) - split <= 25:
            return None
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def _fit_model(self, X_train: pd.DataFrame, y_train: pd.Series, *, light: bool = False):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # RandomForest multi-class
        model = RandomForestClassifier(
            n_estimators=120 if light else 250,
            max_depth=4 if light else 6,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
        )
        model.fit(X_train_scaled, y_train)
        return model, scaler

    def _score_model(self, model, scaler, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        X_test_scaled = scaler.transform(X_test)
        preds = model.predict(X_test_scaled)
        # Macro precision menilai kualitas prediksi untuk UP dan DOWN (dan FLAT) secara seimbang.
        return float(precision_score(y_test, preds, average='macro', zero_division=0))

    def ensure_trained(self, symbol: str, df: pd.DataFrame) -> bool:
        """Train model jika belum ada atau cooldown sudah lewat."""
        now = time.time()
        st = self.states.get(symbol)
        if st and Config.AI_RETRAIN_COOLDOWN_SEC > 0:
            if (now - st.last_fit_ts) < Config.AI_RETRAIN_COOLDOWN_SEC:
                return True

        df_feat = self.prepare_features(df)
        if len(df_feat) < 250:
            return False

        y = self._make_labels(df_feat)
        # drop bar yang tidak punya future
        valid = y.notna()
        df_feat = df_feat.loc[valid]
        y = y.loc[valid]
        y = y.iloc[:-3]
        df_feat = df_feat.iloc[:-3]

        X = df_feat[self.FEATURE_COLS]
        split = self._time_split(X, y)
        if split is None:
            return False
        X_train, X_test, y_train, y_test = split

        model, scaler = self._fit_model(X_train, y_train)
        real_score = self._score_model(model, scaler, X_test, y_test)

        self.states[symbol] = ModelState(
            model=model,
            scaler=scaler,
            last_fit_ts=now,
            last_mcpt_ts=0.0,
            last_real_score=real_score,
            last_p_value=1.0,
        )
        return True

    def mcpt_validation(self, symbol: str, df: pd.DataFrame, n_iterations: int | None = None) -> bool:
        """Monte Carlo Permutation Test (MCPT) yang konsisten.

        - Train model asli + hitung skor pada X_test/y_test.
        - Ulangi: permutasi y_train, train ulang, ukur skor pada X_test/y_test.
        - p-value = proporsi skor fake >= skor real.

        Untuk menjaga stabilitas 24/7:
        - MCPT di-skip jika sudah validasi baru-baru ini (cooldown mengikuti retrain).
        - Early stop jika p-value sudah jelas buruk.
        """
        if n_iterations is None:
            n_iterations = int(Config.MCPT_ITERATIONS)

        now = time.time()
        st = self.states.get(symbol)

        # Pastikan trained dulu
        ok = self.ensure_trained(symbol, df)
        if not ok:
            return False

        st = self.states[symbol]
        # jika MCPT masih "fresh", pakai hasil sebelumnya
        if st.last_mcpt_ts > 0 and Config.AI_RETRAIN_COOLDOWN_SEC > 0:
            if (now - st.last_mcpt_ts) < Config.AI_RETRAIN_COOLDOWN_SEC:
                return st.last_p_value < float(Config.AI_P_VALUE_THRESHOLD)

        df_feat = self.prepare_features(df)
        if len(df_feat) < 250:
            return False

        y = self._make_labels(df_feat)
        y = y.iloc[:-3]
        df_feat = df_feat.iloc[:-3]
        X = df_feat[self.FEATURE_COLS]
        split = self._time_split(X, y)
        if split is None:
            return False
        X_train, X_test, y_train, y_test = split

        # model real (fit ulang untuk memastikan konsistensi dataset)
        model_real, scaler_real = self._fit_model(X_train, y_train, light=False)
        real_score = self._score_model(model_real, scaler_real, X_test, y_test)

        # Reject cepat jika skor real sangat rendah
        if real_score < 0.35:
            self.states[symbol] = ModelState(
                model=model_real,
                scaler=scaler_real,
                last_fit_ts=st.last_fit_ts,
                last_mcpt_ts=now,
                last_real_score=real_score,
                last_p_value=1.0,
            )
            return False

        fake_better = 0

        # Base config untuk fake models (lebih ringan, tapi evaluasi tetap pada X_test yang sama)
        for i in range(int(n_iterations)):
            y_perm = np.random.permutation(y_train)
            model_fake, scaler_fake = self._fit_model(X_train, y_perm, light=True)
            fake_score = self._score_model(model_fake, scaler_fake, X_test, y_test)
            if fake_score >= real_score:
                fake_better += 1

            # Early stop: kalau udah jelas pvalue jelek
            if i >= 50:
                p_so_far = fake_better / (i + 1)
                if p_so_far > 0.3:
                    break

        p_value = fake_better / (i + 1)

        # simpan state: model real terbaru + pvalue
        self.states[symbol] = ModelState(
            model=model_real,
            scaler=scaler_real,
            last_fit_ts=st.last_fit_ts,
            last_mcpt_ts=now,
            last_real_score=real_score,
            last_p_value=float(p_value),
        )

        return p_value < float(Config.AI_P_VALUE_THRESHOLD)

    def predict(self, symbol: str, df: pd.DataFrame, direction: str = "LONG") -> float:
        """Return probability untuk arah yang diminta.

        direction: LONG => P(UP)
                   SHORT => P(DOWN)
        """
        st = self.states.get(symbol)
        if st is None:
            return 0.5

        try:
            df_feat = self.prepare_features(df)
            if df_feat.empty:
                return 0.5
            last_row = df_feat.iloc[[-1]][self.FEATURE_COLS]
            last_scaled = st.scaler.transform(last_row)

            probs = st.model.predict_proba(last_scaled)[0]
            classes = list(st.model.classes_)

            # cari index kelas
            def p_of(cls):
                if cls in classes:
                    return float(probs[classes.index(cls)])
                return 0.0

            if direction.upper() == "SHORT":
                return p_of(-1)
            return p_of(1)
        except Exception:
            return 0.5
