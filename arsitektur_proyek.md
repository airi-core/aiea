
```
Arsitektur File Proyek EA MT5 (Ramah VPS) /SanClassTradingLabs
├── SourceCode
│   ├── EAs
│   │   └── NamaEAUtama.mq5       // File utama Expert Advisor
│   ├── Includes                 // File pendukung (.mqh)
│   │   ├── SignalModule.mqh      // Modul logika sinyal trading
│   │   ├── ExecutionModule.mqh   // Modul eksekusi order (buka/tutup posisi)
│   │   ├── RiskManagement.mqh    // Modul manajemen risiko (SL/TP, lot size)
│   │   ├── LicenseClient.mqh     // Modul komunikasi klien lisensi (jika ada)
│   │   ├── Utils.mqh             // Fungsi-fungsi pembantu umum (logging efisien, dll)
│   │   └── ... (modul lain sesuai kebutuhan: UI, notifikasi, dll.)
│   └── Libraries                // Jika menggunakan library eksternal (.mqh atau .ex4/.ex5)
│       └── ...
├── Compiled                     // Hasil kompilasi (file .ex5)
│   └── Experts
│       └── NamaEAUtama.ex5       // File EA yang siap didistribusikan/dijalankan di MT5
├── Documentation
│   ├── UserGuide.md              // Panduan penggunaan untuk pengguna akhir
│   ├── TechnicalDoc.md           // Dokumentasi teknis (arsitektur, penjelasan kode)
│   ├── PerformanceReport.md      // Laporan hasil uji performa & efisiensi (termasuk di VPS)
│   └── VPS_Recommendations.md    // Rekomendasi spesifikasi & setting VPS
├── Testing
│   ├── SetFiles                 // File setting (.set) untuk backtesting/optimasi
│   │   └── OptimizedSettings.set
│   └── TestResults              // Hasil backtesting, forward testing, log performa
│       └── ... (file log, screenshot, dll.)
├── LicensingServer              // Kode backend untuk sistem lisensi (jika custom)
│   ├── ServerCode               // Script backend (misal: PHP, Python, Node.js)
│   │   └── ... (API validasi lisensi)
│   ├── Database                 // Struktur database atau file data lisensi
│   │   └── ...
│   └── API_Docs                 // Dokumentasi API server lisensi
│       └── ...
├── Tools                        // Alat bantu (misal: obfuscator, script deployment)
│   └── ...
├── .gitignore                    // File untuk Git agar mengabaikan file tertentu (misal: Compiled, TestResults)
└── README.md                     // Deskripsi singkat proyek dan cara setup
