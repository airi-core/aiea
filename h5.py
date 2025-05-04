"""
# Model Pelatihan XAUUSD dengan Deep Learning (Ditingkatkan & Debugged untuk Colab)
# Implementasi Arsitektur LSTM untuk Prediksi OHLC
# Menangani data CSV tanpa header, meningkatkan robustness, dan instruksi Colab
""" # Menutup docstring awal skrip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
# Pastikan layer yang digunakan diimpor dengan benar
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
import os
import time
import json
import sys # Import sys untuk sys.exit

# --- Pengaturan Reproducibility ---
# Mengatur seed untuk hasil yang dapat direproduksi
np.random.seed(42)
tf.random.set_seed(42)

# --- Konfigurasi tampilan grafik ---
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 8)

# --- Kelas untuk pelatihan model ---
# Pastikan tidak ada indentasi sebelum 'class'
class XAUUSDModelTrainer:
    """
    Kelas untuk pelatihan model prediksi XAUUSD menggunakan arsitektur deep learning (LSTM)
    """ # Pastikan docstring kelas ini memiliki indentasi yang benar (4 spasi dari 'class')

    def __init__(self, data_path, output_path="model_output", config=None):
        """
        Inisialisasi trainer model XAUUSD

        Parameters:
        -----------
        data_path : str
            Path ke file CSV data OHLCV XAUUSD (tanpa header)
        output_path : str
            Path untuk menyimpan model dan hasil pelatihan
        config : dict, optional
            Dictionary konfigurasi pelatihan. Jika None, gunakan default.
        """ # Pastikan docstring method ini memiliki indentasi yang benar (4 spasi dari 'def')
        # Pastikan semua baris kode di dalam method ini memiliki indentasi yang konsisten (4 spasi dari 'def')
        self.data_path = data_path
        self.output_path = output_path

        # Membuat direktori output jika belum ada
        # Menggunakan try-except untuk penanganan error pembuatan direktori
        try:
            # Pastikan indentasi di sini benar
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                print(f"[INFO] Direktori output dibuat: {output_path}")
        except OSError as e:
            print(f"[ERROR] Gagal membuat direktori output {output_path}: {e}", file=sys.stderr)
            # Lanjutkan eksekusi, tapi perlu diingat penyimpanan file mungkin gagal

        # Parameter pelatihan dari konfigurasi
        self.config = {
            'batch_size': 423600,
            'epochs': 150,
            'warmup_epochs': 15,
            'learning_rate': 1.618,
            'momentum': 0.618,
            'model_architecture': {
                'lstm_layers': [128, 128, 64],
                'dense_layers': [64, 32],
                'dropout_rate': 0.3,
                'leaky_relu_alpha': 0.1
            },
            'callbacks': {
                'early_stopping_patience': 20,
                'reduce_lr_patience': 10,
                'reduce_lr_factor': 0.5,
                'reduce_lr_min_lr': 0.00001,
                'model_checkpoint_monitor': 'val_loss'
            },
            'data_split': {
                'train_size': 0.7,
                'val_size': 0.15,
                'test_size': 0.15
            },
            'sequence': {
                'window_size': 90,
                'horizon': 15
            }
        }
        if config:
            # Update konfigurasi default dengan yang diberikan
            # Menggunakan loop untuk update rekursif jika ada nested dict
            self._recursive_config_update(self.config, config)


        # Menyiapkan scaler untuk normalisasi data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        # Hasil evaluasi
        self.evaluation_results = {}
        self.history = None
        self.model = None
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.target_shape = None # Akan diisi setelah prepare_sequences

    def _recursive_config_update(self, base_dict, update_dict):
        """Recursively updates a dictionary."""
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                self._recursive_config_update(base_dict[k], v)
            else:
                base_dict[k] = v


    def load_data(self):
        """
        Membaca dan memproses data XAUUSD dari file CSV (tanpa header)
        Kolom diharapkan: time, open, high, low, close, volume
        """
        print("[INFO] Memuat data XAUUSD dari:", self.data_path)

        try:
            # Membaca file CSV tanpa header
            # Menggunakan parameter on_bad_lines='skip' atau 'warn' untuk melewati baris yang error
            # Namun, header=None dan nama kolom eksplisit lebih baik untuk data tanpa header
            df = pd.read_csv(self.data_path, header=None)

            # Memberi nama kolom sesuai urutan yang diharapkan: time, o, h, l, c, v
            # Asumsi: Kolom 0=Time, 1=Open, 2=High, 3=Low, 4=Close, 5=Volume
            # Jika ada kolom lain, mereka akan diberi nama 6, 7, ...
            column_names = ['time', 'open', 'high', 'low', 'close', 'volume']
            # Pastikan DataFrame memiliki minimal jumlah kolom yang diharapkan (OHLCV)
            min_required_cols = len(column_names)

            if df.shape[1] < min_required_cols:
                 raise ValueError(f"File CSV diharapkan memiliki minimal {min_required_cols} kolom (time, o, h, l, c, v). Ditemukan {df.shape[1]} kolom.")

            # Beri nama hanya kolom yang ada
            df.columns = column_names[:df.shape[1]]

            # Pastikan kolom 'time' ada dan bertipe datetime
            if 'time' not in df.columns:
                 # Ini seharusnya tidak terjadi jika shape[1] >= min_required_cols
                 raise ValueError("Kolom 'time' tidak ditemukan setelah memberi nama kolom.")

            # --- Konversi dan Validasi Kolom Waktu ---
            try:
                # Menggunakan errors='coerce' untuk mengubah nilai yang tidak valid menjadi NaT (Not a Time)
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                if df['time'].isnull().any():
                    initial_rows = len(df)
                    df.dropna(subset=['time'], inplace=True)
                    if len(df) < initial_rows:
                        print(f"[WARNING] Menghapus {initial_rows - len(df)} baris dengan nilai waktu yang tidak valid.")

                # Set index setelah membersihkan waktu
                df.set_index('time', inplace=True)

                # Sort index untuk memastikan urutan waktu
                df.sort_index(inplace=True)

            except Exception as e:
                 raise ValueError(f"Gagal mengkonversi atau memproses kolom waktu: {e}")


            # --- Konversi dan Validasi Kolom Numerik ---
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        # Cek apakah kolom bisa dikonversi ke numerik, paksa konversi
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                         print(f"[WARNING] Gagal mengkonversi kolom '{col}' ke numerik: {e}. Kolom mungkin akan berisi NaN.", file=sys.stderr)


            # Hapus baris dengan nilai NaN di kolom OHLCV setelah konversi paksa
            initial_rows = len(df)
            df.dropna(subset=numeric_cols, inplace=True)
            if len(df) < initial_rows:
                 print(f"[INFO] Menghapus {initial_rows - len(df)} baris dengan nilai NaN di kolom OHLCV.")


            # Pastikan ada data yang tersisa setelah dropna
            if df.empty:
                 raise ValueError("Dataset kosong setelah memuat dan membersihkan nilai non-numerik/NaN di kolom OHLCV.")


            # Tambahkan fitur teknikal
            df = self.add_technical_features(df)

            # --- Validasi Data Setelah Fitur Teknikal ---
            # Hapus baris dengan nilai NaN yang dihasilkan dari fitur teknikal
            initial_rows = len(df)
            df.dropna(inplace=True)
            if len(df) < initial_rows:
                 print(f"[INFO] Menghapus {initial_rows - len(df)} baris dengan NaN setelah penambahan fitur teknikal.")

            # Cek nilai tak terhingga (inf) yang mungkin muncul dari perhitungan fitur
            if np.isinf(df.values).any():
                print("[WARNING] Nilai tak terhingga (inf) ditemukan setelah penambahan fitur teknikal. Mengganti dengan NaN dan menghapus baris.")
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                initial_rows_inf = len(df)
                df.dropna(inplace=True)
                if len(df) < initial_rows_inf:
                    print(f"[INFO] Menghapus {initial_rows_inf - len(df)} baris dengan inf.")

            # Pastikan ada data yang tersisa setelah semua pembersihan
            if df.empty:
                 raise ValueError("Dataset kosong setelah semua langkah pembersihan data.")


            # Simpan data yang telah diproses
            self.data = df
            print(f"[INFO] Data dimuat dan diproses dengan sukses. Total baris: {len(df)}")
            print(f"[INFO] Kolom yang tersedia: {df.columns.tolist()}")

            return df

        except FileNotFoundError:
            print(f"[ERROR] File tidak ditemukan di: {self.data_path}", file=sys.stderr)
            # Di Colab, ini sering berarti file belum diunggah atau path salah.
            print("[HINT] Di Google Colab, pastikan file CSV Anda sudah diunggah atau di-mount dari Google Drive.")
            print("[HINT] Gunakan ikon folder di sidebar kiri untuk mengunggah file atau mount Drive.")
            print("[HINT] Pastikan path di `data_file_path` sesuai dengan lokasi file di Colab.")
            sys.exit(1) # Keluar jika file tidak ditemukan
        except ValueError as ve:
            print(f"[ERROR] Kesalahan format data: {ve}", file=sys.stderr)
            print("[HINT] Periksa kembali format file CSV Anda. Pastikan tidak ada header dan kolom pertama adalah waktu, diikuti OHLCV.")
            sys.exit(1) # Keluar jika format data salah
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat memuat atau memproses data: {e}", file=sys.stderr)
            sys.exit(1) # Keluar untuk kesalahan lain


    def add_technical_features(self, df):
        """
        Menambahkan fitur teknikal untuk meningkatkan kinerja model

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame dengan data OHLCV

        Returns:
        --------
        pandas.DataFrame
            DataFrame dengan fitur teknikal tambahan
        """
        # Pastikan ada kolom OHLCV sebelum menambahkan fitur
        required_cols_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols_ohlcv):
             raise ValueError("DataFrame harus berisi kolom OHLCV (open, high, low, close, volume) untuk menambahkan fitur teknikal.")

        # Menggunakan salinan untuk menghindari SettingWithCopyWarning
        df_features = df.copy()

        # Hitung perubahan harga
        df_features['return'] = df_features['close'].pct_change()

        # Tambahkan SMA (Simple Moving Average)
        for window in [5, 10, 20, 50]:
            df_features[f'sma_{window}'] = df_features['close'].rolling(window=window).mean()
            # Hindari pembagian dengan nol jika SMA adalah nol
            df_features[f'sma_ratio_{window}'] = df_features['close'] / df_features[f'sma_{window}'].replace(0, np.nan)


        # Tambahkan EMA (Exponential Moving Average)
        for window in [5, 10, 20, 50]:
            df_features[f'ema_{window}'] = df_features['close'].ewm(span=window, adjust=False).mean()

        # Hitung MACD (Moving Average Convergence Divergence)
        df_features['ema_12'] = df_features['close'].ewm(span=12, adjust=False).mean()
        df_features['ema_26'] = df_features['close'].ewm(span=26, adjust=False).mean()
        df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
        df_features['macd_signal'] = df_features['macd'].ewm(span=9, adjust=False).mean()
        df_features['macd_hist'] = df_features['macd'] - df_features['macd_signal']

        # Hitung RSI (Relative Strength Index)
        # Menggunakan implementasi yang lebih robust
        def calculate_rsi(data, window=14):
            diff = data.diff()
            gain = diff.where(diff > 0, 0)
            loss = -diff.where(diff < 0, 0)

            # Menggunakan ewm untuk smoothing rata-rata (lebih umum)
            avg_gain = gain.ewm(com=window-1, adjust=False).mean()
            avg_loss = loss.ewm(com=window-1, adjust=False).mean()

            rs = avg_gain / avg_loss
            # Menangani kasus avg_loss = 0
            rs = rs.replace([np.inf, -np.inf], np.nan) # Ganti inf dengan NaN
            rs.fillna(0, inplace=True) # Ganti NaN (dari inf) dengan 0 where avg_loss was 0

            rsi = 100 - (100 / (1 + rs))
            return rsi

        df_features['rsi_14'] = calculate_rsi(df_features['close'], 14)


        # Hitung Bollinger Bands
        window_bb = 20
        df_features['bb_middle'] = df_features['close'].rolling(window=window_bb).mean()
        df_features['bb_std'] = df_features['close'].rolling(window=window_bb).std()
        df_features['bb_upper'] = df_features['bb_middle'] + 2 * df_features['bb_std']
        df_features['bb_lower'] = df_features['bb_middle'] - 2 * df_features['bb_std']
        # Hindari pembagian dengan nol jika bb_middle adalah nol
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle'].replace(0, np.nan)


        # Hitung candlestick features
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        # Menggunakan .loc untuk menghindari SettingWithCopyWarning
        df_features.loc[:, 'shadow_upper'] = df_features['high'] - df_features[['open', 'close']].max(axis=1)
        df_features.loc[:, 'shadow_lower'] = df_features[['open', 'close']].min(axis=1) - df_features['low']
        df_features['range'] = df_features['high'] - df_features['low']

        # Hitung volatilitas
        df_features['atr_14'] = self._calculate_atr(df_features, 14)

        # Hitung tren pergerakan harga menggunakan ADX
        df_adx = self._calculate_adx(df_features, 14)
        # Gabungkan kembali kolom ADX ke df_features
        df_features[f'plus_di_{14}'] = df_adx[f'plus_di_{14}']
        df_features[f'minus_di_{14}'] = df_adx[f'minus_di_{14}']
        df_features[f'adx_{14}'] = df_adx[f'adx_{14}']


        # Tambahkan volume
        df_features['volume'] = df_features['volume'] # Kolom volume sudah ada, pastikan termasuk

        # Hapus kolom intermediate jika tidak diperlukan
        df_features.drop(columns=['ema_12', 'ema_26'], errors='ignore', inplace=True) # Kolom ADX intermediate dihapus di _calculate_adx

        return df_features

    def _calculate_atr(self, df, period=14):
        """
        Menghitung Average True Range (ATR)
        """
        # Menggunakan salinan untuk menghindari SettingWithCopyWarning
        df_atr = df.copy()
        df_atr.loc[:, 'high_low'] = df_atr['high'] - df_atr['low']
        df_atr.loc[:, 'high_close'] = np.abs(df_atr['high'] - df_atr['close'].shift())
        df_atr.loc[:, 'low_close'] = np.abs(df_atr['low'] - df_atr['close'].shift())

        ranges = df_atr[['high_low', 'high_close', 'low_close']].max(axis=1)

        # Menggunakan ewm untuk smoothing ATR (lebih umum)
        atr = ranges.ewm(com=period-1, adjust=False).mean()
        return atr

    def _calculate_adx(self, df, period=14):
        """
        Menghitung Average Directional Index (ADX)
        """
        df_adx = df.copy()

        # +DM dan -DM
        df_adx.loc[:, 'up_move'] = df_adx['high'].diff()
        df_adx.loc[:, 'down_move'] = df_adx['low'].diff().multiply(-1)

        df_adx.loc[:, 'plus_dm'] = np.where((df_adx['up_move'] > df_adx['down_move']) & (df_adx['up_move'] > 0), df_adx['up_move'], 0)
        df_adx.loc[:, 'minus_dm'] = np.where((df_adx['down_move'] > df_adx['up_move']) & (df_adx['down_move'] > 0), df_adx['down_move'], 0)

        # Smoothing dengan EWM (lebih umum untuk DI dan ADX)
        alpha = 1 / period
        plus_di_ewm = df_adx['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
        minus_di_ewm = df_adx['minus_dm'].ewm(alpha=alpha, adjust=False).mean()

        # Hitung True Range untuk normalisasi DI
        high_low = df_adx['high'] - df_adx['low']
        high_close = np.abs(df_adx['high'] - df_adx['close'].shift())
        low_close = np.abs(df_adx['low'] - df_adx['close'].shift())
        true_range_ewm = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).ewm(alpha=alpha, adjust=False).mean()

        # +DI dan -DI
        # Hindari pembagian dengan nol
        divisor = true_range_ewm.replace(0, np.nan)
        df_adx[f'plus_di_{period}'] = 100 * (plus_di_ewm / divisor)
        df_adx[f'minus_di_{period}'] = 100 * (minus_di_ewm / divisor)

        # Hitung directional index (DX)
        # Hindari pembagian dengan nol pada sum DI
        sum_di = df_adx[f'plus_di_{period}'] + df_adx[f'minus_di_{period}']
        divisor_dx = sum_di.replace(0, np.nan)
        df_adx[f'dx_{period}'] = 100 * np.abs((df_adx[f'plus_di_{period}'] - df_adx[f'minus_di_{period}']) / divisor_dx)

        # Hitung ADX (average of DX)
        df_adx[f'adx_{period}'] = df_adx[f'dx_{period}'].ewm(alpha=alpha, adjust=False).mean()

        # Hapus kolom intermediate ADX
        df_adx.drop(columns=['up_move', 'down_move', 'plus_dm', 'minus_dm', f'dx_{period}'], errors='ignore', inplace=True)


        return df_adx

    def prepare_sequences(self):
        """
        Menyiapkan data time series dalam format sequence untuk model

        Returns:
        --------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        window_size = self.config['sequence']['window_size']
        horizon = self.config['sequence']['horizon']
        train_size = self.config['data_split']['train_size']
        val_size = self.config['data_split']['val_size']
        # test_size = self.config['data_split']['test_size'] # Tidak digunakan langsung

        print(f"[INFO] Menyiapkan data sequence dengan window size: {window_size}, horizon: {horizon}")

        # Pilih fitur yang akan digunakan (semua kolom kecuali index)
        feature_columns = self.data.columns.tolist()

        # Pilih target yang akan diprediksi (OHLC)
        target_columns = ['open', 'high', 'low', 'close']
        # Pastikan kolom target ada
        if not all(col in feature_columns for col in target_columns):
             raise ValueError(f"Kolom target {target_columns} tidak ditemukan dalam data setelah pemrosesan fitur.")


        # Menyiapkan fitur dan target
        data_features = self.data[feature_columns].values
        data_targets = self.data[target_columns].values

        # Normalisasi data
        # Fit scaler hanya pada data training untuk menghindari data leakage
        # Tentukan indeks split sebelum scaling
        total_samples_data = len(data_features)
        # Hitung jumlah sampel yang akan digunakan untuk membuat sequence
        # Ini adalah jumlah baris data_features_scaled dan data_targets_scaled
        n_sequence_samples = total_samples_data - window_size - horizon + 1
        if n_sequence_samples <= 0:
             raise ValueError(f"Tidak cukup data ({total_samples_data} baris) untuk membuat sequence dengan window_size={window_size} dan horizon={horizon}. Minimal dibutuhkan {window_size + horizon} baris.")


        # Bagi data features dan targets *sebelum* scaling
        # Gunakan proporsi split pada total data yang *bisa* di-sequence
        train_end_idx_data = int(n_sequence_samples * train_size)
        val_end_idx_data = train_end_idx_data + int(n_sequence_samples * val_size)

        # Ambil data untuk scaling (hanya bagian training)
        # Data training untuk scaler adalah data features/targets yang akan membentuk X_train dan y_train
        # X_train terbentuk dari data_features[i:i+window_size]
        # y_train terbentuk dari data_targets[i+window_size:i+window_size+horizon]
        # Indeks i berjalan dari 0 hingga train_end_idx_data - 1
        # Jadi data yang dibutuhkan untuk scaling X_train adalah data_features[0 : train_end_idx_data + window_size]
        # Data yang dibutuhkan untuk scaling y_train adalah data_targets[window_size : train_end_idx_data + window_size + horizon]

        # Perbaiki logika scaling: Fit scaler hanya pada data yang akan menjadi bagian dari set training
        # Data training untuk scaler adalah data_features[0 : train_end_idx_data + window_size]
        # Data training untuk scaler target adalah data_targets[window_size : train_end_idx_data + window_size + horizon]
        # Pastikan indeks tidak melebihi batas array
        features_for_scaler_fit = data_features[:min(total_samples_data, train_end_idx_data + window_size)]
        targets_for_scaler_fit = data_targets[min(total_samples_data, window_size) : min(total_samples_data, train_end_idx_data + window_size + horizon)]

        # --- Validasi Data untuk Scaling ---
        if features_for_scaler_fit.shape[0] == 0 or targets_for_scaler_fit.shape[0] == 0:
             raise ValueError(f"Tidak cukup data untuk fitting scaler. Features shape: {features_for_scaler_fit.shape}, Targets shape: {targets_for_scaler_fit.shape}")


        # Fit scaler hanya pada data training
        features_for_scaler_fit_scaled = self.feature_scaler.fit_transform(features_for_scaler_fit)
        targets_for_scaler_fit_scaled = self.target_scaler.fit_transform(targets_for_scaler_fit)

        # Transform seluruh data menggunakan scaler yang sudah di-fit pada data training
        data_features_scaled_all = self.feature_scaler.transform(data_features)
        data_targets_scaled_all = self.target_scaler.transform(data_targets)


        X, y = [], []

        # Membuat sequences dari data yang sudah di-scale menggunakan scaler dari data training
        for i in range(n_sequence_samples):
            X.append(data_features_scaled_all[i:i+window_size])
            y.append(data_targets_scaled_all[i+window_size:i+window_size+horizon])

        X = np.array(X)
        y = np.array(y)

        # --- Validasi Bentuk Data Sequence ---
        if X.shape[0] == 0 or y.shape[0] == 0:
             # Ini seharusnya sudah ditangani oleh cek n_sequence_samples > 0 di atas
             raise ValueError(f"Tidak cukup data untuk membuat sequence setelah scaling. Bentuk X: {X.shape}, bentuk y: {y.shape}")


        # Reshaping y untuk pelatihan multi-timestep
        # Reshape dari (samples, timesteps, features) menjadi (samples, timesteps*features)
        y = y.reshape(y.shape[0], -1)

        # Membagi data X dan y yang sudah dalam format sequence
        X_train, y_train = X[:train_end_idx_data], y[:train_end_idx_data]
        X_val, y_val = X[train_end_idx_data:val_end_idx_data], y[train_end_idx_data:val_end_idx_data]
        X_test, y_test = X[val_end_idx_data:], y[val_end_idx_data:]

        # --- Validasi Ukuran Split ---
        if len(X_train) == 0:
             print(f"[WARNING] Data training kosong setelah split. Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}. Sesuaikan ukuran split atau periksa jumlah data.")
        if len(X_val) == 0:
             print(f"[WARNING] Data validasi kosong setelah split. Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}. Sesuaikan ukuran split atau periksa jumlah data.")
        if len(X_test) == 0:
             print(f"[WARNING] Data testing kosong setelah split. Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}. Sesuaikan ukuran split atau periksa jumlah data.")


        print(f"[INFO] Bentuk data training: X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"[INFO] Bentuk data validasi: X_val = {X_val.shape}, y_val = {y_val.shape}")
        print(f"[INFO] Bentuk data testing: X_test = {X_test.shape}, y_test = {y_test.shape}")

        # Simpan data untuk digunakan kembali
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        self.target_shape = (horizon, len(target_columns)) # Simpan bentuk target asli

        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self, window_size, n_features):
        """
        Membangun model deep learning (LSTM)
        """
        lstm_layers_config = self.config['model_architecture']['lstm_layers']
        dense_layers_config = self.config['model_architecture']['dense_layers']
        dropout_rate = self.config['model_architecture']['dropout_rate']
        leaky_relu_alpha = self.config['model_architecture']['leaky_relu_alpha']
        horizon = self.config['sequence']['horizon']
        n_targets = 4 # OHLC

        print("[INFO] Membangun model LSTM...")

        model = Sequential()

        # Input layer - LSTM
        # Return sequences true untuk semua layer LSTM kecuali yang terakhir
        if lstm_layers_config:
            model.add(LSTM(
                lstm_layers_config[0],
                return_sequences=len(lstm_layers_config) > 1,
                input_shape=(window_size, n_features)
            ))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=leaky_relu_alpha))
            model.add(Dropout(dropout_rate))

            # Hidden LSTM layers
            for units in lstm_layers_config[1:]:
                model.add(LSTM(
                    units,
                    return_sequences=lstm_layers_config.index(units) < len(lstm_layers_config) - 1 # True kecuali layer terakhir
                ))
                model.add(BatchNormalization())
                model.add(LeakyReLU(alpha=leaky_relu_alpha))
                model.add(Dropout(dropout_rate))
        else:
            # Jika tidak ada layer LSTM, gunakan Dense sebagai input layer
            print("[WARNING] Tidak ada layer LSTM yang dikonfigurasi. Menggunakan Dense sebagai input layer.", file=sys.stderr)
            model.add(Dense(64, input_shape=(window_size, n_features)))
            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=leaky_relu_alpha))
            model.add(Dropout(dropout_rate))
            model.add(tf.keras.layers.Flatten()) # Flatten jika input layer bukan LSTM terakhir


        # Dense layers setelah LSTM (atau setelah Flatten jika tanpa LSTM)
        # Jika tidak ada layer LSTM, layer Dense pertama akan menerima input dari Flatten
        # Jika ada layer LSTM, layer Dense pertama akan menerima input dari LSTM terakhir
        input_from_lstm = bool(lstm_layers_config) # True jika ada layer LSTM
        # Perbaiki input shape untuk Dense pertama jika tanpa LSTM
        first_dense_input_shape = (lstm_layers_config[-1],) if input_from_lstm and lstm_layers_config else (window_size * n_features,) # Bentuk input untuk Dense pertama

        for i, units in enumerate(dense_layers_config):
            if i == 0 and not input_from_lstm:
                 # Layer Dense pertama jika tidak ada LSTM
                 # Gunakan input_shape hanya untuk layer pertama model Sequential
                 model.add(Dense(units)) # Input shape sudah ditangani di layer sebelumnya (Flatten)
            else:
                 model.add(Dense(units))

            model.add(BatchNormalization())
            model.add(LeakyReLU(alpha=leaky_relu_alpha))
            model.add(Dropout(dropout_rate))

        # Output layer - memprediksi flattened output (horizon * n_targets)
        model.add(Dense(horizon * n_targets))


        # Konfigurasi optimizer
        # Menggunakan try-except untuk menangani potensi error konfigurasi optimizer
        try:
            optimizer = SGD(
                learning_rate=self.config['learning_rate'],
                momentum=self.config['momentum'],
                nesterov=True
            )
        except Exception as e:
             print(f"[ERROR] Gagal mengkonfigurasi optimizer SGD: {e}. Menggunakan optimizer default (Adam).", file=sys.stderr)
             optimizer = 'adam' # Fallback ke optimizer default Keras


        # Compile model
        # Menggunakan try-except untuk menangani potensi error kompilasi
        try:
            model.compile(
                optimizer=optimizer,
                loss='mean_squared_error',
                metrics=['mae']
            )
        except Exception as e:
             print(f"[ERROR] Gagal mengkompilasi model: {e}", file=sys.stderr)
             self.model = None # Set model menjadi None jika kompilasi gagal
             return None # Kembalikan None jika kompilasi gagal


        # Tampilkan ringkasan model
        model.summary()

        self.model = model
        return model

    def create_lr_schedule(self):
        """
        Membuat jadwal learning rate dengan warmup dan linear decay
        """
        initial_lr = self.config['learning_rate']
        warmup_epochs = self.config['warmup_epochs']
        total_epochs = self.config['epochs']

        def lr_schedule(epoch, lr):
            # Warmup phase
            if epoch < warmup_epochs:
                return initial_lr * ((epoch + 1) / max(1, warmup_epochs)) # Hindari pembagian dengan nol
            # Linear decay phase
            else:
                decay_epochs = total_epochs - warmup_epochs
                decay_progress = (epoch - warmup_epochs) / max(1, decay_epochs) if decay_epochs > 0 else 0 # Hindari pembagian dengan nol
                return initial_lr * (1.0 - decay_progress * 0.9)  # Decay to 10% of initial LR

        return LearningRateScheduler(lr_schedule)

    def train_model(self):
        """
        Melatih model dengan data yang telah disiapkan
        """
        if self.model is None:
             print("[ERROR] Model belum berhasil dibangun atau dikompilasi. Melewati pelatihan.", file=sys.stderr)
             self.history = None
             return None

        print("[INFO] Memulai pelatihan model...")

        # Menyiapkan callbacks
        lr_scheduler = self.create_lr_schedule()

        # Pastikan monitor metric ada di history model
        monitor_metric = self.config['callbacks']['model_checkpoint_monitor']
        # Keras secara otomatis menambahkan 'loss' dan metrik lain (seperti 'mae')
        # Jika monitor adalah 'val_loss' atau 'val_mae', pastikan ada data validasi
        if monitor_metric.startswith('val_') and (self.val_data is None or self.val_data[0].shape[0] == 0):
             print(f"[WARNING] Monitor metric '{monitor_metric}' memerlukan data validasi, tetapi data validasi kosong. Mengubah monitor menjadi 'loss'.", file=sys.stderr)
             monitor_metric = 'loss'

        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=self.config['callbacks']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=self.config['callbacks']['reduce_lr_factor'],
            patience=self.config['callbacks']['reduce_lr_patience'],
            min_lr=self.config['callbacks']['reduce_lr_min_lr'],
            verbose=1
        )

        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.output_path, 'best_model.h5'),
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1
        )

        # Mendapatkan data train dan validation
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data

        # --- Validasi Data Training/Validation Sebelum Fit ---
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            print("[ERROR] Data training kosong. Tidak dapat melatih model.", file=sys.stderr)
            self.history = None
            return None

        # Data validasi bisa kosong jika split size 0
        validation_data = (X_val, y_val) if X_val.shape[0] > 0 else None

        # Melatih model
        start_time = time.time()

        # Menggunakan try-except untuk menangani potensi error saat pelatihan
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=[lr_scheduler, early_stopping, reduce_lr, model_checkpoint],
                verbose=1
            )

            training_time = time.time() - start_time
            print(f"[INFO] Pelatihan selesai dalam {training_time:.2f} detik")

            # Simpan model terakhir
            # Menggunakan try-except untuk penanganan error penyimpanan model
            try:
                self.model.save(os.path.join(self.output_path, 'final_model.h5'))
                print(f"[INFO] Model terakhir tersimpan di: {self.output_path}")
            except Exception as e:
                print(f"[WARNING] Gagal menyimpan model terakhir: {e}", file=sys.stderr)


            # Simpan history untuk analisis
            self.history = history.history

            return history

        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat pelatihan model: {e}", file=sys.stderr)
            self.history = None # Set history menjadi None jika pelatihan gagal
            return None # Kembalikan None jika pelatihan gagal


    def evaluate_model(self):
        """
        Mengevaluasi model pada data test
        """
        if self.model is None:
             print("[ERROR] Model belum dilatih atau dimuat. Tidak dapat melakukan evaluasi.", file=sys.stderr)
             self.evaluation_results = {"error": "Model tidak tersedia untuk evaluasi"}
             return self.evaluation_results

        # Mendapatkan data test
        X_test, y_test = self.test_data

        # --- Validasi Data Test Sebelum Evaluasi ---
        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
             print("[INFO] Data test kosong. Melewati evaluasi model.")
             self.evaluation_results = {"message": "Data test kosong, evaluasi dilewati."}
             return self.evaluation_results

        print("[INFO] Mengevaluasi kinerja model...")

        # Prediksi
        # Menggunakan try-except untuk menangani potensi error saat prediksi
        try:
            y_pred_scaled = self.model.predict(X_test, verbose=0) # verbose=0 agar tidak print progress bar
        except Exception as e:
             print(f"[ERROR] Gagal melakukan prediksi pada data test: {e}", file=sys.stderr)
             self.evaluation_results = {"error": f"Prediksi pada data test gagal: {e}"}
             return self.evaluation_results


        # Reshape predictions untuk inversi scaling
        horizon, n_targets = self.target_shape
        # Menggunakan try-except untuk menangani potensi error reshape
        try:
            y_test_reshaped = y_test.reshape(y_test.shape[0], horizon, n_targets)
            y_pred_reshaped = y_pred_scaled.reshape(y_pred_scaled.shape[0], horizon, n_targets)
        except Exception as e:
             print(f"[ERROR] Gagal melakukan reshape untuk inverse transform: {e}. Bentuk y_test: {y_test.shape}, bentuk y_pred_scaled: {y_pred_scaled.shape}, target_shape: {self.target_shape}", file=sys.stderr)
             self.evaluation_results = {"error": f"Reshape untuk inverse transform gagal: {e}"}
             return self.evaluation_results


        # Denormalisasi
        # Menggunakan try-except untuk menangani potensi error denormalisasi jika scaler tidak fit dengan benar
        try:
            y_test_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_test_reshaped])
            y_pred_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_pred_reshaped])
        except Exception as e:
            print(f"[ERROR] Gagal melakukan inverse transform pada hasil prediksi: {e}", file=sys.stderr)
            self.evaluation_results = {"error": f"Inverse transform gagal: {e}"}
            # Melewati metrik dan visualisasi jika inverse transform gagal
            return self.evaluation_results


        # Evaluasi untuk setiap timestep
        results = {}

        for t in range(horizon):
            results[f'timestep_{t+1}'] = {
                'mse': {},
                'r2': {}
            }

            # Untuk setiap komponen OHLC
            for i, comp in enumerate(['open', 'high', 'low', 'close']):
                # Hindari error jika ada NaN atau Inf setelah inverse transform
                actual_values = y_test_inverted[:, t, i]
                pred_values = y_pred_inverted[:, t, i]

                # Hapus pasangan NaN/Inf sebelum menghitung metrik
                valid_indices = np.isfinite(actual_values) & np.isfinite(pred_values)
                actual_values_valid = actual_values[valid_indices]
                pred_values_valid = pred_values[valid_indices]

                if len(actual_values_valid) > 0:
                    mse = mean_squared_error(actual_values_valid, pred_values_valid)
                    r2 = r2_score(actual_values_valid, pred_values_valid)
                else:
                    mse = np.nan
                    r2 = np.nan


                results[f'timestep_{t+1}']['mse'][comp] = mse
                results[f'timestep_{t+1}']['r2'][comp] = r2

        # Hitung signal accuracy (mendeteksi arah pergerakan)
        # Menggunakan harga close untuk timestep pertama (prediksi 1 hari ke depan)
        # Sinyal: Naik jika close[t+1] > close[t], Turun jika close[t+1] < close[t], Tetap jika close[t+1] == close[t]
        # Membutuhkan harga close saat ini (timestep terakhir dari window input)
        # Kita perlu data X_test yang belum di-sequence untuk mendapatkan harga close saat ini
        # Atau ambil harga close terakhir dari window input dari X_test
        # Ambil harga close terakhir dari window input (index terakhir dari window_size)
        # Kolom close adalah index 3 di OHLCV (pastikan index kolom 'close' di self.data)
        window_size = self.config['sequence']['window_size']
        try:
            # Pastikan self.data tidak None dan kolom 'close' ada
            if self.data is None or 'close' not in self.data.columns:
                 raise ValueError("Data atau kolom 'close' tidak tersedia untuk menghitung metrik signal.")

            close_col_idx = self.data.columns.get_loc('close')
            # Ambil harga close terakhir dari window input (index window_size-1) dari X_test scaled
            close_current_scaled = X_test[:, window_size-1, close_col_idx]
            # Inverse transform hanya kolom close dari data scaled
            # Perlu membuat array dummy dengan shape (n_samples, n_features) untuk inverse_transform feature_scaler
            # Pastikan feature_scaler sudah di-fit
            if self.feature_scaler is None or not hasattr(self.feature_scaler, 'n_features_in_'):
                 raise ValueError("Feature scaler belum di-fit dengan benar.")

            dummy_arr_scaled = np.zeros((X_test.shape[0], self.feature_scaler.n_features_in_))
            dummy_arr_scaled[:, close_col_idx] = close_current_scaled
            close_current_inverted = self.feature_scaler.inverse_transform(dummy_arr_scaled)[:, close_col_idx]

            # Harga close aktual untuk timestep pertama yang diprediksi (t=0)
            actual_close_t1 = y_test_inverted[:, 0, 3] # Index 3 untuk 'close' di target OHLC

            # Harga close prediksi untuk timestep pertama yang diprediksi (t=0)
            pred_close_t1 = y_pred_inverted[:, 0, 3] # Index 3 untuk 'close' di target OHLC

            # Hitung arah pergerakan aktual
            # Naik: actual_close_t1 > close_current_inverted
            # Turun: actual_close_t1 < close_current_inverted
            # Tetap: actual_close_t1 == close_current_inverted
            actual_direction = np.sign(actual_close_t1 - close_current_inverted)

            # Hitung arah pergerakan prediksi
            pred_direction = np.sign(pred_close_t1 - close_current_inverted)

            # Hapus pasangan NaN/Inf sebelum menghitung metrik signal
            valid_signal_indices = np.isfinite(actual_direction) & np.isfinite(pred_direction)
            actual_direction_valid = actual_direction[valid_signal_indices]
            pred_direction_valid = pred_direction[valid_signal_indices]

            if len(actual_direction_valid) > 0:
                # Accuracy: (TP + TN) / Total
                signals_accuracy = np.mean(actual_direction_valid == pred_direction_valid)

                # Precision dan Recall untuk sinyal "Naik" (nilai = 1)
                tp_up = np.sum((actual_direction_valid == 1) & (pred_direction_valid == 1))
                fp_up = np.sum(((actual_direction_valid == 0) | (actual_direction_valid == -1)) & (pred_direction_valid == 1))
                fn_up = np.sum((actual_direction_valid == 1) & ((pred_direction_valid == 0) | (pred_direction_valid == -1)))

                precision_up = tp_up / (tp_up + fp_up) if (tp_up + fp_up) > 0 else 0
                recall_up = tp_up / (tp_up + fn_up) if (tp_up + fn_up) > 0 else 0

                # Precision dan Recall untuk sinyal "Turun" (nilai = -1)
                tp_down = np.sum((actual_direction_valid == -1) & (pred_direction_valid == -1))
                fp_down = np.sum(((actual_direction_valid == 0) | (actual_direction_valid == 1)) & (pred_direction_valid == -1))
                fn_down = np.sum((actual_direction_valid == -1) & ((pred_direction_valid == 0) | (pred_direction_valid == 1)))

                precision_down = tp_down / (tp_down + fp_down) if (tp_down + fp_down) > 0 else 0
                recall_down = tp_down / (tp_down + fn_down) if (tp_down + fn_down) > 0 else 0

            else:
                signals_accuracy = np.nan
                precision_up = np.nan
                recall_up = np.nan
                precision_down = np.nan
                recall_down = np.nan

            results['signals'] = {
                'accuracy': signals_accuracy,
                'precision_up': precision_up,
                'recall_up': recall_up,
                'precision_down': precision_down,
                'recall_down': recall_down
            }
        except Exception as e:
             print(f"[WARNING] Gagal menghitung metrik signal: {e}. Melewati metrik signal.", file=sys.stderr)
             results['signals'] = {"error": f"Gagal menghitung metrik signal: {e}"}


        # Tampilkan hasil
        print("\n=== HASIL EVALUASI ===")
        if "error" in self.evaluation_results:
             print(f"Error Evaluasi: {self.evaluation_results['error']}")
        elif "message" in self.evaluation_results:
             print(self.evaluation_results["message"])
        else:
            print(f"Metrik Regresi per Timestep (Denormalisasi):")
            for t in range(horizon):
                print(f"  Timestep {t+1}:")
                for comp in ['open', 'high', 'low', 'close']:
                    mse_val = results[f'timestep_{t+1}']['mse'][comp]
                    r2_val = results[f'timestep_{t+1}']['r2'][comp]
                    print(f"    {comp.upper()}: MSE={mse_val:.6f}, R²={r2_val:.6f}")

            if 'signals' in results and "error" not in results['signals']:
                print(f"\nMetrik Prediksi Arah (Timestep 1):")
                print(f"  Accuracy: {results['signals']['accuracy']:.4f}")
                print(f"  Sinyal Naik: Precision={results['signals']['precision_up']:.4f}, Recall={results['signals']['recall_up']:.4f}\n")
                print(f"  Sinyal Turun: Precision={results['signals']['precision_down']:.4f}, Recall={results['signals']['recall_down']:.4f}")
            elif 'signals' in results:
                 print(f"\nMetrik Prediksi Arah (Timestep 1): {results['signals']['error']}")


        # Simpan hasil evaluasi
        self.evaluation_results = results

        # Visualisasi hasil
        # Hanya visualisasi jika inverse transform berhasil dan data test tidak kosong
        if "error" not in self.evaluation_results and "message" not in self.evaluation_results:
             self.visualize_predictions(y_test_inverted, y_pred_inverted)

        return results


    def visualize_predictions(self, y_test_inverted, y_pred_inverted, samples=100):
        """
        Membuat visualisasi prediksi vs aktual pada data test
        """
        print("[INFO] Membuat visualisasi prediksi pada data test...")

        # Mengambil subset data test untuk visualisasi
        # Pastikan jumlah sampel tidak melebihi jumlah data yang tersedia
        n_available_samples = y_test_inverted.shape[0]
        samples_to_plot = min(samples, n_available_samples)

        if samples_to_plot <= 0:
             print("[INFO] Tidak cukup sampel data test untuk visualisasi prediksi vs aktual.")
             return

        y_test_subset = y_test_inverted[:samples_to_plot]
        y_pred_subset = y_pred_inverted[:samples_to_plot]

        # Plot untuk harga close
        plt.figure(figsize=(16, 10))

        component_idx = 3  # Close price
        timestep = 0  # First timestep (prediksi 1 hari ke depan)

        # Pastikan data subset tidak kosong
        if y_test_subset.shape[0] > 0:
            plt.plot(y_test_subset[:, timestep, component_idx], label='Actual', linewidth=2)
            plt.plot(y_pred_subset[:, timestep, component_idx], label='Predicted', linewidth=2, alpha=0.7)
            plt.title(f'Prediksi vs Aktual: Harga Close XAUUSD (Timestep {timestep+1}) pada Data Test', fontsize=16)
            plt.xlabel('Sample', fontsize=14)
            plt.ylabel('Harga', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            # Simpan plot
            plot_path = os.path.join(self.output_path, 'predictions_vs_actual_test.png')
            try:
                plt.savefig(plot_path)
                plt.close()
                print(f"[INFO] Visualisasi prediksi vs aktual pada data test tersimpan di: {plot_path}")
            except Exception as e:
                print(f"[WARNING] Gagal menyimpan visualisasi prediksi vs aktual: {e}", file=sys.stderr)

        else:
             print("[INFO] Tidak cukup sampel data test untuk visualisasi prediksi vs aktual.")


        # Plot training history
        if self.history:
            plt.figure(figsize=(16, 8))

            plt.subplot(1, 2, 1)
            plt.plot(self.history['loss'], label='Training Loss')
            if 'val_loss' in self.history:
                plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Loss History', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(self.history['mae'], label='Training MAE')
            if 'val_mae' in self.history:
                plt.plot(self.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('MAE', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)

            plt.tight_layout()

            # Simpan plot history
            history_path = os.path.join(self.output_path, 'training_history.png')
            try:
                plt.savefig(history_path)
                plt.close()
                print(f"[INFO] Visualisasi history pelatihan tersimpan di: {history_path}")
            except Exception as e:
                print(f"[WARNING] Gagal menyimpan visualisasi history pelatihan: {e}", file=sys.stderr)

        else:
             print("[INFO] History pelatihan tidak tersedia untuk visualisasi.")


    def run_pipeline(self):
        """
        Menjalankan seluruh pipeline pelatihan dan evaluasi
        """
        print("[INFO] Menjalankan pipeline pelatihan model XAUUSD...")
        print(f"[INFO] Menggunakan konfigurasi: {json.dumps(self.config, indent=2)}")

        # 1. Load data
        # load_data sudah menangani sys.exit jika gagal
        self.load_data() # Jika load_data gagal, sys.exit akan dipanggil

        # 2. Prepare sequences
        # prepare_sequences sudah menangani ValueError jika data tidak cukup
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_sequences()
        except ValueError as ve:
            print(f"[ERROR] Gagal menyiapkan sequence data: {ve}", file=sys.stderr)
            return {"error": f"Gagal menyiapkan sequence data: {ve}"}


        # 3. Build model
        # Pastikan data training tidak kosong sebelum membangun model
        if self.train_data[0].shape[0] == 0:
             print("[ERROR] Data training kosong setelah menyiapkan sequence. Tidak dapat membangun atau melatih model.", file=sys.stderr)
             return {"error": "Data training kosong setelah menyiapkan sequence"}

        # build_model sudah menangani error kompilasi
        self.build_model(
            window_size=self.train_data[0].shape[1], # Ambil window_size dari bentuk data
            n_features=self.train_data[0].shape[2]  # Ambil n_features dari bentuk data
        )

        if self.model is None:
             print("[ERROR] Model gagal dibangun atau dikompilasi. Melewati pelatihan dan evaluasi.", file=sys.stderr)
             return {"error": "Model gagal dibangun atau dikompilasi"}


        # 4. Train model
        # train_model sudah menangani error pelatihan
        self.train_model() # Jika train_model gagal, self.history akan None

        # Muat kembali model terbaik untuk evaluasi dan prediksi
        best_model_path = os.path.join(self.output_path, 'best_model.h5')
        if os.path.exists(best_model_path):
            print(f"[INFO] Memuat model terbaik dari: {best_model_path}")
            try:
                # Pastikan model dimuat dengan custom_objects jika ada layer non-standar
                # Dalam kasus ini LeakyReLU, BatchNormalization, LSTM, Dense adalah standar Keras
                # Tapi jika ada layer kustom, perlu custom_objects={...}
                self.model = tf.keras.models.load_model(best_model_path)
            except Exception as e:
                 print(f"[ERROR] Gagal memuat model terbaik dari {best_model_path}: {e}. Menggunakan model terakhir jika tersedia.", file=sys.stderr)
                 # Jika gagal memuat model terbaik, model instance tetap model terakhir dari training


        # 5. Evaluate model
        # evaluate_model sudah menangani data test kosong dan error evaluasi
        results = self.evaluate_model()


        print("[INFO] Pipeline selesai dijalankan!")
        return results

    def save_results_summary(self):
        """
        Menyimpan ringkasan hasil pelatihan dan evaluasi dalam format CSV dan TXT
        """
        if not self.evaluation_results or ("error" in self.evaluation_results and "message" not in self.evaluation_results):
             print("[WARNING] Tidak ada hasil evaluasi yang valid untuk disimpan.", file=sys.stderr)
             return

        # Simpan hasil evaluasi dalam CSV
        results_list = []

        # Proses data hasil evaluasi regresi
        # Pastikan keys ada sebelum iterasi
        if self.evaluation_results and not ("error" in self.evaluation_results or "message" in self.evaluation_results):
            for timestep in [key for key in self.evaluation_results.keys() if key.startswith('timestep')]:
                t_num = int(timestep.split('_')[1])
                for metric in ['mse', 'r2']:
                    if metric in self.evaluation_results[timestep]: # Pastikan metrik ada
                        for comp, value in self.evaluation_results[timestep][metric].items():
                            results_list.append({
                                'Timestep': t_num,
                                'Komponen': comp.upper(),
                                'Metrik': metric.upper(),
                                'Nilai': value
                            })

            # Tambahkan metrik signal
            if 'signals' in self.evaluation_results and "error" not in self.evaluation_results['signals']:
                 for metric, value in self.evaluation_results['signals'].items():
                     results_list.append({
                         'Timestep': 'Timestep 1', # Metrik signal hanya untuk timestep pertama
                         'Komponen': 'Signal',
                         'Metrik': metric,
                         'Nilai': value
                     })


        results_df = pd.DataFrame(results_list)

        # Simpan ke CSV
        csv_path = os.path.join(self.output_path, 'evaluation_results.csv')
        try:
            results_df.to_csv(csv_path, index=False)
            print(f"[INFO] Hasil evaluasi tersimpan dalam CSV di: {csv_path}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan hasil evaluasi CSV: {e}", file=sys.stderr)


        # Buat ringkasan teks
        summary_path = os.path.join(self.output_path, 'model_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write("===== RINGKASAN MODEL XAUUSD =====\n\n")
                f.write(f"Tanggal pelatihan: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {self.data_path}\n")
                f.write(f"Jumlah data setelah pembersihan: {len(self.data) if self.data is not None else 'N/A'}\n\n")

                f.write("Konfigurasi Pelatihan:\n")
                f.write(json.dumps(self.config, indent=2) + "\n\n")

                f.write("Hasil Evaluasi:\n")
                if 'message' in self.evaluation_results:
                    f.write(f"  {self.evaluation_results['message']}\n")
                elif 'error' in self.evaluation_results:
                     f.write(f"  Error Evaluasi: {self.evaluation_results['error']}\n")
                else:
                    f.write("Metrik Regresi per Timestep (Denormalisasi):\n")
                    for timestep in [key for key in self.evaluation_results.keys() if key.startswith('timestep')]:
                        t_num = timestep.split('_')[1]
                        f.write(f"\nTimestep {t_num}:\n")

                        for comp in ['open', 'high', 'low', 'close']:
                             mse_val = self.evaluation_results[timestep]['mse'][comp]
                             r2_val = self.evaluation_results[timestep]['r2'][comp]
                             f.write(f"  {comp.upper()}: MSE={mse_val:.6f}, R²={r2_val:.6f}\n")

                    if 'signals' in self.evaluation_results and "error" not in self.evaluation_results['signals']:
                        f.write("\nMetrik Prediksi Arah (Timestep 1):\n")
                        f.write(f"  Accuracy: {self.evaluation_results['signals']['accuracy']:.4f}\n")
                        f.write(f"  Sinyal Naik: Precision={self.evaluation_results['signals']['precision_up']:.4f}, Recall={self.evaluation_results['signals']['recall_up']:.4f}\n")
                        f.write(f"  Sinyal Turun: Precision={self.evaluation_results['signals']['precision_down']:.4f}, Recall={self.evaluation_results['signals']['recall_down']:.4f}\n")
                    elif 'signals' in self.evaluation_results:
                         f.write(f"\nMetrik Prediksi Arah (Timestep 1): {self.evaluation_results['signals']['error']}\n")


                f.write("\n===== AKHIR RINGKASAN =====\n")

            print(f"[INFO] Ringkasan model tersimpan dalam TXT di: {summary_path}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan ringkasan model TXT: {e}", file=sys.stderr)


    def predict_future(self, days=30):
        """
        Memprediksi pergerakan harga untuk beberapa hari ke depan secara iteratif
        """
        if self.model is None:
             print("[ERROR] Model belum dilatih atau dimuat. Tidak dapat melakukan prediksi masa depan.", file=sys.stderr)
             return None
        if self.data is None or self.feature_scaler is None or self.target_scaler is None or self.target_shape is None:
             print("[ERROR] Data atau scaler belum disiapkan. Jalankan run_pipeline terlebih dahulu.", file=sys.stderr)
             return None


        window_size = self.config['sequence']['window_size']
        horizon, n_targets = self.target_shape
        feature_columns = self.data.columns.tolist() # Gunakan semua fitur yang dilatih

        # Ambil data terakhir sesuai window size
        if len(self.data) < window_size:
             print(f"[ERROR] Tidak cukup data historis ({len(self.data)} baris) untuk membuat window input ({window_size} baris) untuk prediksi masa depan.", file=sys.stderr)
             return None

        latest_data = self.data.iloc[-window_size:].copy()

        # Hasil prediksi
        predictions = []

        # Jumlah iterasi untuk mendapatkan jumlah hari yang diinginkan
        iterations = (days + horizon - 1) // horizon

        print(f"[INFO] Memulai prediksi masa depan untuk sekitar {days} hari ke depan ({iterations} iterasi)...")

        # Prediksi iteratif
        for i in range(iterations):
            # Persiapkan data untuk prediksi
            input_data = latest_data[feature_columns].values

            # --- Validasi Input Prediksi ---
            if np.isnan(input_data).any() or np.isinf(input_data).any():
                 print(f"[WARNING] Input data untuk prediksi iterasi {i+1} mengandung NaN atau Inf. Menghentikan prediksi.", file=sys.stderr)
                 break # Hentikan prediksi jika input tidak valid

            # Menggunakan try-except untuk menangani error scaling input
            try:
                input_scaled = self.feature_scaler.transform(input_data)
                input_reshaped = input_scaled.reshape(1, window_size, len(feature_columns))
            except Exception as e:
                 print(f"[ERROR] Gagal melakukan scaling atau reshape input pada iterasi {i+1}: {e}. Menghentikan prediksi.", file=sys.stderr)
                 break # Hentikan prediksi jika scaling/reshape gagal


            # Prediksi
            try:
                pred_scaled = self.model.predict(input_reshaped, verbose=0)[0] # verbose=0 agar tidak print progress bar setiap iterasi
                pred_reshaped = pred_scaled.reshape(1, horizon, n_targets)
                pred_values = self.target_scaler.inverse_transform(pred_reshaped[0])
            except Exception as e:
                print(f"[ERROR] Gagal melakukan prediksi atau inverse transform pada iterasi {i+1}: {e}. Menghentikan prediksi.", file=sys.stderr)
                break # Hentikan prediksi jika model gagal memprediksi


            # Membuat DataFrame dari hasil prediksi
            # Asumsi prediksi harian, tambahkan 1 hari dari tanggal terakhir data
            # Jika ini iterasi pertama, mulai dari hari setelah data terakhir
            # Jika ini iterasi berikutnya, mulai dari hari setelah prediksi terakhir
            last_known_date = self.data.index[-1] # Tanggal terakhir dari data historis asli
            start_date_this_iter = predictions[-1].index[-1] + pd.Timedelta(days=1) if predictions else last_known_date + pd.Timedelta(days=1)


            pred_dates = pd.date_range(
                start=start_date_this_iter,
                periods=horizon,
                freq='D' # Asumsi data harian
            )

            # Membuat DataFrame untuk prediksi OHLC
            pred_df = pd.DataFrame(
                pred_values,
                columns=['open', 'high', 'low', 'close'], # Hanya OHLC yang diprediksi langsung
                index=pred_dates
            )

            # Tambahkan ke hasil
            predictions.append(pred_df)

            # Update data terbaru untuk prediksi berikutnya
            # Gabungkan data historis terbaru dengan prediksi OHLC baru
            # Ambil kolom OHLC dari latest_data untuk digabung dengan pred_df
            latest_ohlc = latest_data[['open', 'high', 'low', 'close']].copy()
            combined_latest = pd.concat([latest_ohlc, pred_df])

            # Hitung fitur-fitur teknikal untuk data gabungan terbaru
            # Ini akan menghasilkan NaN di awal combined_latest, tapi itu normal
            # Menggunakan try-except untuk menangani error penambahan fitur
            try:
                combined_latest_with_features = self.add_technical_features(combined_latest)
            except Exception as e:
                 print(f"[ERROR] Gagal menghitung fitur teknikal untuk data prediksi pada iterasi {i+1}: {e}. Menghentikan prediksi.", file=sys.stderr)
                 break # Hentikan prediksi jika penambahan fitur gagal


            # Ambil data terbaru (window_size terakhir) termasuk fitur teknikal yang baru dihitung
            # Pastikan kolom fitur yang digunakan konsisten
            if not all(col in combined_latest_with_features.columns for col in feature_columns):
                 print(f"[ERROR] DataFrame gabungan setelah penambahan fitur tidak memiliki semua kolom fitur yang diharapkan pada iterasi {i+1}. Menghentikan prediksi.", file=sys.stderr)
                 break # Hentikan prediksi jika kolom fitur hilang

            latest_data = combined_latest_with_features.iloc[-window_size:][feature_columns].copy()


        # Gabungkan semua prediksi OHLC
        if not predictions:
             print("[WARNING] Tidak ada prediksi masa depan yang dihasilkan.", file=sys.stderr)
             return None

        all_predictions = pd.concat(predictions)

        # Batasi jumlah hari sesuai permintaan
        all_predictions = all_predictions.iloc[:days]

        # Simpan prediksi
        predictions_path = os.path.join(self.output_path, 'future_predictions.csv')
        try:
            all_predictions.to_csv(predictions_path, index=True) # Simpan index tanggal
            print(f"[INFO] Prediksi masa depan tersimpan di: {predictions_path}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan prediksi masa depan CSV: {e}", file=sys.stderr)


        # Visualisasi prediksi
        # Hanya visualisasi jika ada data prediksi
        if not all_predictions.empty:
            plt.figure(figsize=(16, 10))
            plt.plot(all_predictions.index, all_predictions['close'], label='Close', linewidth=2, color='blue')
            plt.fill_between(
                all_predictions.index,
                all_predictions['low'],
                all_predictions['high'],
                alpha=0.2,
                color='blue',
                label='Range High-Low'
            )

            plt.title(f'Prediksi Harga XAUUSD untuk {len(all_predictions)} Hari ke Depan', fontsize=16)
            plt.xlabel('Tanggal', fontsize=14)
            plt.ylabel('Harga', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            # Simpan plot
            future_plot_path = os.path.join(self.output_path, 'future_predictions.png')
            try:
                plt.savefig(future_plot_path)
                plt.close()
                print(f"[INFO] Visualisasi prediksi tersimpan di: {future_plot_path}")
            except Exception as e:
                print(f"[ERROR] Gagal menyimpan visualisasi prediksi masa depan: {e}", file=sys.stderr)

        else:
             print("[INFO] Tidak ada data prediksi masa depan untuk visualisasi.")


        return all_predictions


# --- Main Execution ---
if __name__ == "__main__":
    # --- INSTRUKSI UNTUK GOOGLE COLAB ---
    # 1. Pastikan Anda sudah menginstal library yang dibutuhkan:
    #    Biasanya numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow sudah ada di Colab.
    #    Jika tidak, jalankan:
    #    !pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
    #
    # 2. Unggah file data CSV Anda ke lingkungan Colab.
    #    - Di sidebar kiri Colab, klik ikon folder.
    #    - Klik ikon "Upload to session storage".
    #    - Pilih file CSV data XAUUSD Anda.
    #    - ATAU mount Google Drive Anda:
    #      from google.colab import drive
    #      drive.mount('/content/drive')
    #      Lalu sesuaikan `data_file_path` di bawah.
    #
    # 3. Sesuaikan `data_file_path` di bawah agar sesuai dengan nama file yang Anda unggah.
    #    Jika diunggah langsung, biasanya namanya sama.
    #    Jika dari Drive, path-nya akan seperti '/content/drive/My Drive/folder_anda/nama_file.csv'
    #
    # 4. Jalankan sel kode ini di Colab.
    # --- AKHIR INSTRUKSI COLAB ---

    # Ganti dengan path data XAUUSD Anda di lingkungan Colab
    # Pastikan file ini ada dan memiliki kolom: time, open, high, low, close, volume (tanpa header)
    data_file_path = "xauusd_data.csv" # Sesuaikan ini!

    # --- Konfigurasi Pelatihan ---
    # Anda bisa mengubah nilai-nilai di sini untuk eksperimen
    training_config = {
        'batch_size': 4096,
        'epochs': 150,
        'warmup_epochs': 15,
        'learning_rate': 0.005,
        'momentum': 0.95,
        'model_architecture': {
            'lstm_layers': [128, 128, 64],
            'dense_layers': [64, 32],
            'dropout_rate': 0.3,
            'leaky_relu_alpha': 0.1
        },
        'callbacks': {
            'early_stopping_patience': 20,
            'reduce_lr_patience': 10,
            'reduce_lr_factor': 0.5,
            'reduce_lr_min_lr': 0.00001,
            'model_checkpoint_monitor': 'val_loss' # Pastikan ada data validasi jika menggunakan val_loss
        },
        'data_split': {
            'train_size': 0.7,
            'val_size': 0.15, # Atur > 0 jika monitor val_loss
            'test_size': 0.15
        },
        'sequence': {
            'window_size': 90, # Menambah window size
            'horizon': 15 # Menambah horizon
        }
    }

    # Inisialisasi trainer dengan konfigurasi
    trainer = XAUUSDModelTrainer(data_path=data_file_path, config=training_config)

    # Jalankan pipeline pelatihan dan evaluasi
    evaluation_results = trainer.run_pipeline()

    # Simpan ringkasan hasil (hanya jika pipeline berhasil sebagian atau penuh)
    if evaluation_results is not None:
        trainer.save_results_summary()

    # Prediksi pergerakan harga masa depan (jika pipeline berhasil dan model ada)
    # Cek apakah ada model dan tidak ada error fatal di pipeline
    if trainer.model is not None and evaluation_results is not None and "error" not in evaluation_results:
        future_predictions = trainer.predict_future(days=60) # Prediksi untuk 60 hari ke depan

    print("Proses eksekusi skrip selesai!")

