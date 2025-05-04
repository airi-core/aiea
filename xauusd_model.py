"""
# Model Pelatihan XAUUSD dengan Deep Learning
# Implementasi Arsitektur 24 Layer untuk Prediksi OHLC

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
import os
import time

# Konfigurasi tampilan grafik
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (14, 8)

class XAUUSDModelTrainer:
    """
    Kelas untuk pelatihan model prediksi XAUUSD menggunakan arsitektur deep learning 24 layer
    """
    
    def __init__(self, data_path, output_path="model_output"):
        """
        Inisialisasi trainer model XAUUSD
        
        Parameters:
        -----------
        data_path : str
            Path ke file CSV data OHLC XAUUSD
        output_path : str
            Path untuk menyimpan model dan hasil pelatihan
        """
        self.data_path = data_path
        self.output_path = output_path
        
        # Membuat direktori output jika belum ada
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Parameter pelatihan dari konfigurasi JSON
        self.batch_size = 8192
        self.epochs = 89
        self.warmup_epochs = 21
        self.learning_rate = 1.618
        self.momentum = 0.618
        
        # Menyiapkan scaler untuk normalisasi data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # Hasil evaluasi
        self.evaluation_results = {}
        
    def load_data(self):
        """
        Membaca dan memproses data XAUUSD dari file CSV
        """
        print("[INFO] Memuat data XAUUSD dari:", self.data_path)
        
        # Membaca file CSV
        df = pd.read_csv(self.data_path)
        
        # Cek kolom data
        required_columns = ['open', 'high', 'low', 'close']
        # Konversi nama kolom ke lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Pastikan kolom yang diperlukan ada
        for col in required_columns:
            if col not in df.columns:
                possible_cols = [c for c in df.columns if col.lower() in c.lower()]
                if possible_cols:
                    df[col] = df[possible_cols[0]]
                else:
                    raise ValueError(f"Kolom {col} tidak ditemukan dalam dataset")
        
        # Pastikan data bertipe datetime jika ada kolom tanggal
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'timestamp'])]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
        
        # Tambahkan fitur teknikal
        df = self.add_technical_features(df)
        
        # Hapus baris dengan nilai NaN
        df.dropna(inplace=True)
        
        # Simpan data yang telah diproses
        self.data = df
        print(f"[INFO] Data dimuat dengan sukses. Total baris: {len(df)}")
        print(f"[INFO] Kolom yang tersedia: {df.columns.tolist()}")
        
        return df
    
    def add_technical_features(self, df):
        """
        Menambahkan fitur teknikal untuk meningkatkan kinerja model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame dengan data OHLC
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame dengan fitur teknikal tambahan
        """
        # Pastikan ada kolom OHLC
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("Data harus berisi kolom OHLC (open, high, low, close)")
        
        # Hitung perubahan harga
        df['return'] = df['close'].pct_change()
        
        # Tambahkan SMA (Simple Moving Average)
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
        
        # Tambahkan EMA (Exponential Moving Average)
        for window in [5, 10, 20, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Hitung MACD (Moving Average Convergence Divergence)
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Hitung RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Hitung Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Hitung candlestick features
        df['body_size'] = abs(df['close'] - df['open'])
        df['shadow_upper'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['shadow_lower'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        
        # Hitung volatilitas
        df['atr_14'] = self._calculate_atr(df, 14)
        
        # Hitung tren pergerakan harga menggunakan ADX
        df = self._calculate_adx(df, 14)
        
        return df
    
    def _calculate_atr(self, df, period=14):
        """
        Menghitung Average True Range (ATR)
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        atr = true_range.rolling(period).mean()
        return atr
    
    def _calculate_adx(self, df, period=14):
        """
        Menghitung Average Directional Index (ADX)
        """
        # +DM dan -DM
        df['up_move'] = df['high'].diff()
        df['down_move'] = df['low'].diff().multiply(-1)
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Smoothing dengan ATR
        atr = self._calculate_atr(df, period)
        
        # +DI dan -DI
        df[f'plus_di_{period}'] = 100 * (df['plus_dm'].rolling(period).mean() / atr)
        df[f'minus_di_{period}'] = 100 * (df['minus_dm'].rolling(period).mean() / atr)
        
        # Hitung directional index
        df[f'dx_{period}'] = 100 * np.abs((df[f'plus_di_{period}'] - df[f'minus_di_{period}']) / 
                                         (df[f'plus_di_{period}'] + df[f'minus_di_{period}']))
        
        # Hitung ADX (average of DX)
        df[f'adx_{period}'] = df[f'dx_{period}'].rolling(period).mean()
        
        return df
    
    def prepare_sequences(self, window_size=10, horizon=5):
        """
        Menyiapkan data time series dalam format sequence untuk model
        
        Parameters:
        -----------
        window_size : int
            Jumlah timestep sebelumnya untuk prediksi
        horizon : int
            Jumlah timestep ke depan untuk diprediksi
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print(f"[INFO] Menyiapkan data sequence dengan window size: {window_size}, horizon: {horizon}")
        
        # Pilih fitur yang akan digunakan
        feature_columns = [
            'open', 'high', 'low', 'close', 'return',
            'sma_5', 'sma_10', 'sma_20', 'sma_ratio_5', 'sma_ratio_10',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14', 'bb_width', 'body_size', 'shadow_upper', 'shadow_lower',
            'range', 'atr_14', f'adx_14', f'plus_di_14', f'minus_di_14'
        ]
        
        # Pilih target yang akan diprediksi
        target_columns = ['open', 'high', 'low', 'close']
        
        # Menyiapkan fitur dan target
        data_features = self.data[feature_columns].values
        data_targets = self.data[target_columns].values
        
        # Normalisasi data
        data_features_scaled = self.feature_scaler.fit_transform(data_features)
        data_targets_scaled = self.target_scaler.fit_transform(data_targets)
        
        X, y = [], []
        
        # Membuat sequences
        for i in range(len(data_features_scaled) - window_size - horizon + 1):
            X.append(data_features_scaled[i:i+window_size])
            y.append(data_targets_scaled[i+window_size:i+window_size+horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshaping y untuk pelatihan multi-timestep
        # Reshape dari (samples, timesteps, features) menjadi (samples, timesteps*features)
        y = y.reshape(y.shape[0], -1)
        
        # Membagi data menjadi train, validation, dan test
        # Train = 70%, Val = 15%, Test = 15%
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
        
        print(f"[INFO] Bentuk data training: X_train = {X_train.shape}, y_train = {y_train.shape}")
        print(f"[INFO] Bentuk data validasi: X_val = {X_val.shape}, y_val = {y_val.shape}")
        print(f"[INFO] Bentuk data testing: X_test = {X_test.shape}, y_test = {y_test.shape}")
        
        # Simpan data untuk digunakan kembali
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        self.target_shape = (horizon, len(target_columns))
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self, window_size, n_features, horizon=5, n_targets=4):
        """
        Membangun model deep learning dengan 24 layer untuk memprediksi OHLC
        
        Parameters:
        -----------
        window_size : int
            Jumlah timestep sebelumnya sebagai input
        n_features : int
            Jumlah fitur input per timestep
        horizon : int
            Jumlah timestep yang diprediksi
        n_targets : int
            Jumlah target yang diprediksi (4 untuk OHLC)
            
        Returns:
        --------
        tf.keras.models.Sequential
            Model deep learning yang telah dibuat
        """
        print("[INFO] Membangun model dengan 24 layer...")
        
        # Inisialisasi model
        model = Sequential()
        
        # Input layer
        model.add(Dense(128, input_shape=(window_size, n_features)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
        
        # Hidden layers (total 22 layer processing)
        layer_sizes = [256, 512, 768, 1024, 1024, 768, 512, 384, 256, 192, 128]
        
        # Encoder path (meningkatkan dimensi)
        for units in layer_sizes[:5]:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.1))
            model.add(BatchNormalization())
            if units >= 512:
                model.add(Dropout(0.3))
            else:
                model.add(Dropout(0.2))
        
        # Decoder path (menurunkan dimensi)
        for units in layer_sizes[5:]:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.1))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
        
        # Flatten untuk output
        model.add(tf.keras.layers.Flatten())
        
        # Output layer - memprediksi multiple timesteps ke depan untuk OHLC
        model.add(Dense(horizon * n_targets))
        
        # Konfigurasi optimizer sesuai JSON
        optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=True
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        # Tampilkan ringkasan model
        model.summary()
        
        self.model = model
        return model
    
    def create_lr_schedule(self):
        """
        Membuat jadwal learning rate dengan warmup dan linear decay
        """
        def lr_schedule(epoch, lr):
            # Warmup phase
            if epoch < self.warmup_epochs:
                return self.learning_rate * ((epoch + 1) / self.warmup_epochs)
            # Linear decay phase
            else:
                decay_epochs = self.epochs - self.warmup_epochs
                decay_progress = (epoch - self.warmup_epochs) / decay_epochs if decay_epochs > 0 else 0
                return self.learning_rate * (1.0 - decay_progress * 0.9)  # Decay to 10% of initial LR
            
        return tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    def train_model(self):
        """
        Melatih model dengan data yang telah disiapkan
        """
        print("[INFO] Memulai pelatihan model...")
        
        # Menyiapkan callbacks
        lr_scheduler = self.create_lr_schedule()
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.output_path, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Mendapatkan data train dan validation
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Melatih model
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[lr_scheduler, early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"[INFO] Pelatihan selesai dalam {training_time:.2f} detik")
        
        # Simpan model terakhir
        self.model.save(os.path.join(self.output_path, 'final_model.h5'))
        print(f"[INFO] Model tersimpan di: {self.output_path}")
        
        # Simpan history untuk analisis
        self.history = history.history
        
        return history
    
    def evaluate_model(self):
        """
        Mengevaluasi model pada data test
        """
        print("[INFO] Mengevaluasi kinerja model...")
        
        # Mendapatkan data test
        X_test, y_test = self.test_data
        
        # Prediksi
        y_pred_scaled = self.model.predict(X_test)
        
        # Reshape predictions untuk inversi scaling
        horizon, n_targets = self.target_shape
        y_test_reshaped = y_test.reshape(y_test.shape[0], horizon, n_targets)
        y_pred_reshaped = y_pred_scaled.reshape(y_pred_scaled.shape[0], horizon, n_targets)
        
        # Denormalisasi
        y_test_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_test_reshaped])
        y_pred_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_pred_reshaped])
        
        # Evaluasi untuk setiap timestep
        results = {}
        
        for t in range(horizon):
            results[f'timestep_{t+1}'] = {
                'mse': {},
                'r2': {}
            }
            
            # Untuk setiap komponen OHLC
            for i, comp in enumerate(['open', 'high', 'low', 'close']):
                mse = mean_squared_error(y_test_inverted[:, t, i], y_pred_inverted[:, t, i])
                r2 = r2_score(y_test_inverted[:, t, i], y_pred_inverted[:, t, i])
                
                results[f'timestep_{t+1}']['mse'][comp] = mse
                results[f'timestep_{t+1}']['r2'][comp] = r2
        
        # Hitung signal accuracy (mendeteksi arah pergerakan)
        # Menggunakan harga close untuk timestep pertama
        actual_direction = np.sign(np.diff(y_test_inverted[:, 0, 3]))  # Close price
        pred_direction = np.sign(np.diff(y_pred_inverted[:, 0, 3]))  # Close price
        
        # Compute trading signals
        signals_accuracy = np.mean(actual_direction == pred_direction)
        
        # Compute confusion matrix metrics
        cm = confusion_matrix(actual_direction, pred_direction)
        if len(cm) > 1:  # Memastikan matrix berukuran minimal 2x2
            tn, fp, fn, tp = cm.ravel()
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            false_positive_rate = 0
            false_negative_rate = 0
            
        # Tambahkan ke hasil
        results['signals'] = {
            'accuracy': signals_accuracy,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
        
        # Tampilkan hasil
        print("\n=== HASIL EVALUASI ===")
        print(f"MSE (Mean Squared Error):")
        for t in range(horizon):
            print(f"  Timestep {t+1}:")
            for comp in ['open', 'high', 'low', 'close']:
                print(f"    {comp.upper()}: {results[f'timestep_{t+1}']['mse'][comp]:.6f}")
        
        print(f"\nR² Score:")
        for t in range(horizon):
            print(f"  Timestep {t+1}:")
            for comp in ['open', 'high', 'low', 'close']:
                print(f"    {comp.upper()}: {results[f'timestep_{t+1}']['r2'][comp]:.6f}")
        
        print(f"\nSignal Accuracy: {results['signals']['accuracy']:.4f}")
        print(f"False Positive Rate: {results['signals']['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {results['signals']['false_negative_rate']:.4f}")
        
        # Simpan hasil evaluasi
        self.evaluation_results = results
        
        # Visualisasi hasil
        self.visualize_predictions()
        
        return results
    
    def visualize_predictions(self, samples=100):
        """
        Membuat visualisasi prediksi vs aktual
        """
        print("[INFO] Membuat visualisasi prediksi...")
        
        # Mengambil subset data test untuk visualisasi
        X_test, y_test = self.test_data
        X_test_subset = X_test[:samples]
        
        # Prediksi
        y_pred_scaled = self.model.predict(X_test_subset)
        
        # Reshape dan inverse transform
        horizon, n_targets = self.target_shape
        y_test_reshaped = y_test[:samples].reshape(-1, horizon, n_targets)
        y_pred_reshaped = y_pred_scaled.reshape(-1, horizon, n_targets)
        
        y_test_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_test_reshaped])
        y_pred_inverted = np.array([self.target_scaler.inverse_transform(seq) for seq in y_pred_reshaped])
        
        # Plot untuk harga close
        plt.figure(figsize=(16, 10))
        
        component_idx = 3  # Close price
        timestep = 0  # First timestep
        
        plt.plot(y_test_inverted[:, timestep, component_idx], label='Actual', linewidth=2)
        plt.plot(y_pred_inverted[:, timestep, component_idx], label='Predicted', linewidth=2, alpha=0.7)
        plt.title(f'Prediksi vs Aktual: Harga Close XAUUSD (Timestep {timestep+1})', fontsize=16)
        plt.xlabel('Sample', fontsize=14)
        plt.ylabel('Harga', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Simpan plot
        plot_path = os.path.join(self.output_path, 'predictions_vs_actual.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Plot training history
        if hasattr(self, 'history'):
            plt.figure(figsize=(16, 8))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history['loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.title('Loss History', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history['mae'], label='Training MAE')
            plt.plot(self.history['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('MAE', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True)
            
            plt.tight_layout()
            
            # Simpan plot history
            history_path = os.path.join(self.output_path, 'training_history.png')
            plt.savefig(history_path)
            plt.close()
            
            print(f"[INFO] Visualisasi tersimpan di: {self.output_path}")
    
    def run_pipeline(self, window_size=10, horizon=5):
        """
        Menjalankan seluruh pipeline pelatihan dan evaluasi
        
        Parameters:
        -----------
        window_size : int
            Jumlah timestep untuk input model
        horizon : int
            Jumlah timestep untuk diprediksi
            
        Returns:
        --------
        dict
            Hasil evaluasi model
        """
        # 1. Load data
        self.load_data()
        
        # 2. Prepare sequences
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_sequences(
            window_size=window_size, 
            horizon=horizon
        )
        
        # 3. Build model
        self.build_model(
            window_size=window_size,
            n_features=X_train.shape[2],
            horizon=horizon,
            n_targets=4  # OHLC
        )
        
        # 4. Train model
        self.train_model()
        
        # 5. Evaluate model
        results = self.evaluate_model()
        
        print("[INFO] Pipeline selesai dijalankan!")
        return results
    
    def save_results_summary(self):
        """
        Menyimpan ringkasan hasil pelatihan dan evaluasi dalam format CSV dan TXT
        """
        # Simpan hasil evaluasi dalam CSV
        results_df = pd.DataFrame()
        
        # Proses data hasil evaluasi
        for timestep in [key for key in self.evaluation_results.keys() if key.startswith('timestep')]:
            for metric in ['mse', 'r2']:
                for comp, value in self.evaluation_results[timestep][metric].items():
                    results_df = results_df.append({
                        'Timestep': int(timestep.split('_')[1]),
                        'Komponen': comp.upper(),
                        'Metrik': metric.upper(),
                        'Nilai': value
                    }, ignore_index=True)
        
        # Tambahkan metrik signal
        for metric, value in self.evaluation_results['signals'].items():
            results_df = results_df.append({
                'Timestep': 'All',
                'Komponen': 'Signal',
                'Metrik': metric,
                'Nilai': value
            }, ignore_index=True)
            
        # Simpan ke CSV
        csv_path = os.path.join(self.output_path, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Buat ringkasan teks
        summary_path = os.path.join(self.output_path, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("===== RINGKASAN MODEL XAUUSD =====\n\n")
            f.write(f"Tanggal pelatihan: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Jumlah data: {len(self.data)}\n\n")
            
            f.write("Parameter Pelatihan:\n")
            f.write(f"- Batch Size: {self.batch_size}\n")
            f.write(f"- Epochs: {self.epochs}\n")
            f.write(f"- Warmup Epochs: {self.warmup_epochs}\n")
            f.write(f"- Learning Rate: {self.learning_rate}\n")
            f.write(f"- Momentum: {self.momentum}\n\n")
            
            f.write("Hasil Evaluasi:\n")
            for timestep in [key for key in self.evaluation_results.keys() if key.startswith('timestep')]:
                f.write(f"\nTimestep {timestep.split('_')[1]}:\n")
                
                for comp in ['open', 'high', 'low', 'close']:
                    f.write(f"  {comp.upper()}:\n")
                    f.write(f"    MSE: {self.evaluation_results[timestep]['mse'][comp]:.6f}\n")
                    f.write(f"    R²:  {self.evaluation_results[timestep]['r2'][comp]:.6f}\n")
            
            f.write("\nPrediksi Sinyal:\n")
            f.write(f"  Accuracy: {self.evaluation_results['signals']['accuracy']:.4f}\n")
            f.write(f"  False Positive Rate: {self.evaluation_results['signals']['false_positive_rate']:.4f}\n")
            f.write(f"  False Negative Rate: {self.evaluation_results['signals']['false_negative_rate']:.4f}\n")
            
            f.write("\n===== AKHIR RINGKASAN =====\n")
            
        print(f"[INFO] Ringkasan hasil tersimpan di: {self.output_path}")
        
    def predict_future(self, days=30):
        """
        Memprediksi pergerakan harga untuk beberapa hari ke depan
        
        Parameters:
        -----------
        days : int
            Jumlah hari untuk diprediksi
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame berisi prediksi OHLC
        """
        # Mengambil data terakhir sesuai window size
        window_size = self.train_data[0].shape[1]
        latest_data = self.data.iloc[-window_size:].copy()
        
        # Hasil prediksi
        predictions = []
        horizon, n_targets = self.target_shape
        
        # Persiapan data
        feature_columns = [
            'open', 'high', 'low', 'close', 'return',
            'sma_5', 'sma_10', 'sma_20', 'sma_ratio_5', 'sma_ratio_10',
            'ema_5', 'ema_10', 'ema_20',
            'macd', 'macd_signal', 'macd_hist',
            'rsi_14', 'bb_width', 'body_size', 'shadow_upper', 'shadow_lower',
            'range', 'atr_14', f'adx_14', f'plus_di_14', f'minus_di_14'
        ]
        
        # Jumlah iterasi untuk mendapatkan jumlah hari yang diinginkan
        iterations = (days + horizon - 1) // horizon
        
        # Prediksi iteratif
        for i in range(iterations):
            # Persiapkan data untuk prediksi
            input_data = latest_data[feature_columns].values
            input_scaled = self.feature_scaler.transform(input_data)
            input_reshaped = input_scaled.reshape(1, window_size, len(feature_columns))
            
            # Prediksi
            pred_scaled = self.model.predict(input_reshaped)[0]
            pred_reshaped = pred_scaled.reshape(1, horizon, n_targets)
            pred_values = self.target_scaler.inverse_transform(pred_reshaped[0])
            
            # Membuat DataFrame dari hasil prediksi
            pred_dates = pd.date_range(
                start=latest_data.index[-1] + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Membuat DataFrame untuk prediksi
            pred_df = pd.DataFrame(
                pred_values,
                columns=['open', 'high', 'low', 'close'],
                index=pred_dates
            )
            
            # Tambahkan ke hasil
            predictions.append(pred_df)
            
            # Update data terbaru untuk prediksi berikutnya
            # Hanya ambil data prediksi OHLC untuk ditambahkan
            new_data = pred_df.copy()
            
            # Hitung fitur-fitur teknikal
            new_data['return'] = new_data['close'].pct_change()
            
            # SMA
            for window in [5, 10, 20, 50]:
                combined_data = pd.concat([latest_data[['close']], new_data[['close']]])
                new_data[f'sma_{window}'] = combined_data['close'].rolling(window=window).mean().tail(horizon)
                new_data[f'sma_ratio_{window}'] = new_data['close'] / new_data[f'sma_{window}']
            
            # EMA
            for window in [5, 10, 20, 50]:
                combined_data = pd.concat([latest_data[['close']], new_data[['close']]])
                new_data[f'ema_{window}'] = combined_data['close'].ewm(span=window, adjust=False).mean().tail(horizon)
            
            # MACD
            combined_data = pd.concat([latest_data[['close']], new_data[['close']]])
            new_data['ema_12'] = combined_data['close'].ewm(span=12, adjust=False).mean().tail(horizon)
            new_data['ema_26'] = combined_data['close'].ewm(span=26, adjust=False).mean().tail(horizon)
            new_data['macd'] = new_data['ema_12'] - new_data['ema_26']
            new_data['macd_signal'] = new_data['macd'].ewm(span=9, adjust=False).mean()
            new_data['macd_hist'] = new_data['macd'] - new_data['macd_signal']
            
            # RSI
            combined_data = pd.concat([latest_data[['close']], new_data[['close']]])
            delta = combined_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            new_data['rsi_14'] = rsi.tail(horizon)
            
            # Bollinger Bands
            combined_data = pd.concat([latest_data[['close']], new_data[['close']]])
            new_data['bb_middle'] = combined_data['close'].rolling(window=20).mean().tail(horizon)
            new_data['bb_std'] = combined_data['close'].rolling(window=20).std().tail(horizon)
            new_data['bb_upper'] = new_data['bb_middle'] + 2 * new_data['bb_std']
            new_data['bb_lower'] = new_data['bb_middle'] - 2 * new_data['bb_std']
            new_data['bb_width'] = (new_data['bb_upper'] - new_data['bb_lower']) / new_data['bb_middle']
            
            # Candlestick features
            new_data['body_size'] = abs(new_data['close'] - new_data['open'])
            new_data['shadow_upper'] = new_data['high'] - new_data[['open', 'close']].max(axis=1)
            new_data['shadow_lower'] = new_data[['open', 'close']].min(axis=1) - new_data['low']
            new_data['range'] = new_data['high'] - new_data['low']
            
            # ATR
            combined_data = pd.concat([latest_data[['high', 'low', 'close']], new_data[['high', 'low', 'close']]])
            high_low = combined_data['high'] - combined_data['low']
            high_close = np.abs(combined_data['high'] - combined_data['close'].shift())
            low_close = np.abs(combined_data['low'] - combined_data['close'].shift())
            
            combined_ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(combined_ranges, axis=1)
            new_data['atr_14'] = true_range.rolling(14).mean().tail(horizon)
            
            # ADX
            combined_data = pd.concat([latest_data[['high', 'low']], new_data[['high', 'low']]])
            combined_data['up_move'] = combined_data['high'].diff()
            combined_data['down_move'] = combined_data['low'].diff().multiply(-1)
            
            combined_data['plus_dm'] = np.where(
                (combined_data['up_move'] > combined_data['down_move']) & 
                (combined_data['up_move'] > 0), 
                combined_data['up_move'], 0
            )
            combined_data['minus_dm'] = np.where(
                (combined_data['down_move'] > combined_data['up_move']) & 
                (combined_data['down_move'] > 0), 
                combined_data['down_move'], 0
            )
            
            # Calculate ADX
            atr_data = true_range.rolling(14).mean()
            plus_di = 100 * (combined_data['plus_dm'].rolling(14).mean() / atr_data)
            minus_di = 100 * (combined_data['minus_dm'].rolling(14).mean() / atr_data)
            dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
            adx = dx.rolling(14).mean()
            
            new_data['adx_14'] = adx.tail(horizon)
            new_data['plus_di_14'] = plus_di.tail(horizon)
            new_data['minus_di_14'] = minus_di.tail(horizon)
            
            # Update data terbaru
            latest_data = pd.concat([latest_data, new_data])
            latest_data = latest_data.iloc[-window_size:]
        
        # Gabungkan semua prediksi
        all_predictions = pd.concat(predictions)
        
        # Batasi jumlah hari sesuai permintaan
        all_predictions = all_predictions.iloc[:days]
        
        # Simpan prediksi
        predictions_path = os.path.join(self.output_path, 'future_predictions.csv')
        all_predictions.to_csv(predictions_path)
        
        # Visualisasi prediksi
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
        
        plt.title(f'Prediksi Harga XAUUSD untuk {days} Hari ke Depan', fontsize=16)
        plt.xlabel('Tanggal', fontsize=14)
        plt.ylabel('Harga', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Simpan plot
        future_plot_path = os.path.join(self.output_path, 'future_predictions.png')
        plt.savefig(future_plot_path)
        plt.close()
        
        print(f"[INFO] Prediksi masa depan tersimpan di: {predictions_path}")
        print(f"[INFO] Visualisasi prediksi tersimpan di: {future_plot_path}")
        
        return all_predictions


# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path data XAUUSD Anda
    data_path = "xauusd_data.csv"
    
    # Inisialisasi trainer
    trainer = XAUUSDModelTrainer(data_path)
    
    # Jalankan pipeline pelatihan
    results = trainer.run_pipeline(window_size=20, horizon=5)
    
    # Simpan ringkasan hasil
    trainer.save_results_summary()
    
    # Prediksi pergerakan harga masa depan
    future_predictions = trainer.predict_future(days=30)
    
    print("Proses selesai!")
"""