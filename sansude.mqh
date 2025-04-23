//+------------------------------------------------------------------+
//|                                                      SanSuDe.mq5  |
//|                                      SanClass Trading Labs Team   |
//|                                                                   |
//+------------------------------------------------------------------+
#property copyright "sansude.vR963fqKw"
#property link      "stl.labs"
#property version   "1.00"
#property description "EA Akumulasi & Distribusi (Supply & Demand) untuk MT5"
#property description "Mengidentifikasi area Supply (Distribusi) dan Demand (Akumulasi)"
#property description "Entry otomatis dengan rasio Risk:Reward 1:1.5"

#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>

// Definisi enumerasi untuk tipe area
enum ENUM_ZONE_TYPE {
   ZONE_SUPPLY = 1,   // Area Supply (Distribusi)
   ZONE_DEMAND = 2    // Area Demand (Akumulasi)
};

// Definisi struktur untuk menyimpan informasi area
struct SDZone {
   ENUM_ZONE_TYPE type;    // Tipe area (Supply/Demand)
   datetime time;          // Waktu pembentukan
   double upper;           // Batas atas area
   double lower;           // Batas bawah area
   double entry;           // Level entry
   double sl;              // Level stop loss
   double tp;              // Level take profit
   bool valid;             // Status validitas area
   bool used;              // Status penggunaan area (sudah digunakan atau belum)
};

// Input Parameter
input group "==== Telegram Notification Settings ===="
input string TelegramToken = "";               // Token Bot Telegram
input string TelegramChatID = "";              // Chat ID Telegram
input bool   EnableTelegramNotifications = false; // Aktifkan notifikasi Telegram

input group "==== Risk Management Settings ===="
input double RiskPercentage = 0.423;           // Persentase risiko per transaksi (%)
input bool   EnableTrailingStop = true;        // Aktifkan Trailing Stop

input group "==== Trading Settings ===="
input int    LookbackPeriod = 100;             // Periode lookback untuk menemukan area S&D

// Global variables
CTrade trade;                 // Objek Trade untuk eksekusi order
CSymbolInfo symbolInfo;       // Informasi simbol trading
SDZone lastSupplyZone;        // Area Supply terakhir yang terdeteksi
SDZone lastDemandZone;        // Area Demand terakhir yang terdeteksi
int supplyOrderTicket = 0;    // Ticket order Supply
int demandOrderTicket = 0;    // Ticket order Demand
datetime lastDetectionTime = 0; // Waktu deteksi terakhir

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   // Inisialisasi objek trade dan symbolInfo
   trade.SetExpertMagicNumber(123456);
   symbolInfo.Name(_Symbol);
   
   // Reset area Supply dan Demand
   ResetSupplyZone();
   ResetDemandZone();
   
   // Kirim pesan Telegram bahwa EA telah dimulai
   if (EnableTelegramNotifications) {
      string message = "SanSuDe EA telah dimulai pada " + _Symbol + ", Timeframe: " + TimeframeToString(Period());
      SendTelegramMessage(message);
   }
   
   // Mulai timer untuk optimasi resource
   EventSetTimer(60); // Set timer setiap 60 detik
   
   Print("SanSuDe EA berhasil diinisialisasi pada ", _Symbol, ", Timeframe: ", TimeframeToString(Period()));
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   // Hapus timer
   EventKillTimer();
   
   // Kirim pesan Telegram bahwa EA telah berhenti
   if (EnableTelegramNotifications) {
      string message = "SanSuDe EA telah berhenti pada " + _Symbol + ", Timeframe: " + TimeframeToString(Period());
      SendTelegramMessage(message);
   }
   
   Print("SanSuDe EA telah dihentikan pada ", _Symbol);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   // Refresh data symbol
   symbolInfo.RefreshRates();
   
   // Cek trailing stop untuk posisi terbuka
   if (EnableTrailingStop) {
      CheckAndUpdateTrailingStop();
   }
   
   // Untuk optimasi resource, deteksi area baru hanya pada awal candle baru
   datetime currentTime = iTime(_Symbol, Period(), 0);
   if (currentTime == lastDetectionTime) {
      return;
   }
   
   lastDetectionTime = currentTime;
   
   // Deteksi area Supply dan Demand baru
   DetectSDZones();
   
   // Pasang order untuk area yang terdeteksi
   PlaceOrders();
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer() {
   // Timer digunakan untuk melakukan pengecekan berkala tanpa membebani fungsi OnTick
   CheckPendingOrders();
}

//+------------------------------------------------------------------+
//| Deteksi area Supply dan Demand                                   |
//+------------------------------------------------------------------+
void DetectSDZones() {
   // Deteksi Area Distribusi (Supply)
   DetectSupplyZone();
   
   // Deteksi Area Akumulasi (Demand)
   DetectDemandZone();
}

//+------------------------------------------------------------------+
//| Deteksi area Supply                                              |
//+------------------------------------------------------------------+
void DetectSupplyZone() {
   // Area Supply terbentuk ketika:
   // 1. Ada candle bullish dengan harga penutupan tertinggi dalam rentang tertentu
   // 2. Candle tepat sebelumnya juga merupakan candle bullish
   
   int highestCloseIndex = -1;
   double highestClose = 0;
   
   // Cari candle dengan close tertinggi
   for (int i = 1; i <= LookbackPeriod; i++) {
      double close = iClose(_Symbol, Period(), i);
      if (close > highestClose) {
         highestClose = close;
         highestCloseIndex = i;
      }
   }
   
   // Jika tidak ditemukan candle dengan close tertinggi, return
   if (highestCloseIndex < 1) return;
   
   // Cek apakah candle tersebut dan candle sebelumnya adalah bullish
   double highestOpen = iOpen(_Symbol, Period(), highestCloseIndex);
   double prevOpen = iOpen(_Symbol, Period(), highestCloseIndex + 1);
   double prevClose = iClose(_Symbol, Period(), highestCloseIndex + 1);
   
   // Bullish candle: Close > Open
   if (highestClose > highestOpen && prevClose > prevOpen) {
      // Area Supply ditemukan, buat objek Supply
      SDZone newSupplyZone;
      newSupplyZone.type = ZONE_SUPPLY;
      newSupplyZone.time = iTime(_Symbol, Period(), highestCloseIndex);
      
      // Titik-titik kunci untuk perhitungan Entry, SL, dan TP
      double highPoint = iHigh(_Symbol, Period(), highestCloseIndex);
      double lowPoint = iLow(_Symbol, Period(), highestCloseIndex + 1);
      double zoneHeight = highPoint - lowPoint;
      
      // Definisikan area Supply (dari level 0.618 hingga level 1)
      newSupplyZone.upper = highPoint;
      newSupplyZone.lower = highPoint - (zoneHeight * 0.618);
      
      // Entry di level 0.618
      newSupplyZone.entry = newSupplyZone.lower;
      
      // SL di level 0.382 dari rentang harga
      newSupplyZone.sl = lowPoint + (zoneHeight * 0.382);
      
      // Hitung jarak Entry-SL untuk menentukan TP
      double entrySlDistance = MathAbs(newSupplyZone.entry - newSupplyZone.sl);
      
      // TP pada level 1.5 dari jarak Entry-SL
      newSupplyZone.tp = newSupplyZone.entry - (entrySlDistance * 1.5);
      
      newSupplyZone.valid = true;
      newSupplyZone.used = false;
      
      // Update area Supply terakhir
      lastSupplyZone = newSupplyZone;
      
      // Kirim notifikasi Telegram
      if (EnableTelegramNotifications) {
         string message = "Area Distribusi (Supply) terdeteksi pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
         message += "Upper: " + DoubleToString(newSupplyZone.upper, _Digits) + "\n";
         message += "Lower: " + DoubleToString(newSupplyZone.lower, _Digits) + "\n";
         message += "Entry: " + DoubleToString(newSupplyZone.entry, _Digits) + "\n";
         message += "SL: " + DoubleToString(newSupplyZone.sl, _Digits) + "\n";
         message += "TP: " + DoubleToString(newSupplyZone.tp, _Digits);
         SendTelegramMessage(message);
      }
      
      Print("Area Distribusi (Supply) terdeteksi pada ", _Symbol, ", ", TimeframeToString(Period()));
   }
}

//+------------------------------------------------------------------+
//| Deteksi area Demand                                              |
//+------------------------------------------------------------------+
void DetectDemandZone() {
   // Area Demand terbentuk ketika:
   // 1. Ada candle bearish dengan harga penutupan terendah dalam rentang tertentu
   // 2. Candle tepat sebelumnya juga merupakan candle bearish
   
   int lowestCloseIndex = -1;
   double lowestClose = DBL_MAX;
   
   // Cari candle dengan close terendah
   for (int i = 1; i <= LookbackPeriod; i++) {
      double close = iClose(_Symbol, Period(), i);
      if (close < lowestClose) {
         lowestClose = close;
         lowestCloseIndex = i;
      }
   }
   
   // Jika tidak ditemukan candle dengan close terendah, return
   if (lowestCloseIndex < 1) return;
   
   // Cek apakah candle tersebut dan candle sebelumnya adalah bearish
   double lowestOpen = iOpen(_Symbol, Period(), lowestCloseIndex);
   double prevOpen = iOpen(_Symbol, Period(), lowestCloseIndex + 1);
   double prevClose = iClose(_Symbol, Period(), lowestCloseIndex + 1);
   
   // Bearish candle: Close < Open
   if (lowestClose < lowestOpen && prevClose < prevOpen) {
      // Area Demand ditemukan, buat objek Demand
      SDZone newDemandZone;
      newDemandZone.type = ZONE_DEMAND;
      newDemandZone.time = iTime(_Symbol, Period(), lowestCloseIndex);
      
      // Titik-titik kunci untuk perhitungan Entry, SL, dan TP
      double lowPoint = iLow(_Symbol, Period(), lowestCloseIndex);
      double highPoint = iHigh(_Symbol, Period(), lowestCloseIndex + 1);
      double zoneHeight = highPoint - lowPoint;
      
      // Definisikan area Demand (dari level 0.618 hingga level 1)
      newDemandZone.lower = lowPoint;
      newDemandZone.upper = lowPoint + (zoneHeight * 0.618);
      
      // Entry di level 0.618
      newDemandZone.entry = newDemandZone.upper;
      
      // SL di level 0.382 dari rentang harga
      newDemandZone.sl = highPoint - (zoneHeight * 0.382);
      
      // Hitung jarak Entry-SL untuk menentukan TP
      double entrySlDistance = MathAbs(newDemandZone.entry - newDemandZone.sl);
      
      // TP pada level 1.5 dari jarak Entry-SL
      newDemandZone.tp = newDemandZone.entry + (entrySlDistance * 1.5);
      
      newDemandZone.valid = true;
      newDemandZone.used = false;
      
      // Update area Demand terakhir
      lastDemandZone = newDemandZone;
      
      // Kirim notifikasi Telegram
      if (EnableTelegramNotifications) {
         string message = "Area Akumulasi (Demand) terdeteksi pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
         message += "Upper: " + DoubleToString(newDemandZone.upper, _Digits) + "\n";
         message += "Lower: " + DoubleToString(newDemandZone.lower, _Digits) + "\n";
         message += "Entry: " + DoubleToString(newDemandZone.entry, _Digits) + "\n";
         message += "SL: " + DoubleToString(newDemandZone.sl, _Digits) + "\n";
         message += "TP: " + DoubleToString(newDemandZone.tp, _Digits);
         SendTelegramMessage(message);
      }
      
      Print("Area Akumulasi (Demand) terdeteksi pada ", _Symbol, ", ", TimeframeToString(Period()));
   }
}

//+------------------------------------------------------------------+
//| Pasang order untuk area yang terdeteksi                          |
//+------------------------------------------------------------------+
void PlaceOrders() {
   // Pasang order untuk area Supply
   if (lastSupplyZone.valid && !lastSupplyZone.used) {
      // Area Supply valid, pasang sell limit
      double lotSize = CalculateLotSize(ZONE_SUPPLY);
      
      // Jika lotSize valid, pasang order
      if (lotSize > 0) {
         // Kirim pesan sebelum memasang order
         Print("Memasang Sell Limit pada ", _Symbol, " di harga ", DoubleToString(lastSupplyZone.entry, _Digits),
               ", SL: ", DoubleToString(lastSupplyZone.sl, _Digits), ", TP: ", DoubleToString(lastSupplyZone.tp, _Digits));
         
         // Jika harga saat ini di bawah entry, tidak perlu memasang order
         if (symbolInfo.Ask() < lastSupplyZone.entry) {
            Print("Harga saat ini (", DoubleToString(symbolInfo.Ask(), _Digits), ") di bawah level entry Supply (", 
                  DoubleToString(lastSupplyZone.entry, _Digits), "). Tidak memasang order.");
            lastSupplyZone.used = true;
            return;
         }
         
         // Pasang sell limit
         if (trade.SellLimit(lotSize, lastSupplyZone.entry, _Symbol, lastSupplyZone.sl, lastSupplyZone.tp)) {
            supplyOrderTicket = trade.ResultOrder();
            lastSupplyZone.used = true;
            
            Print("Sell Limit berhasil dipasang. Ticket: ", supplyOrderTicket);
            
            // Kirim notifikasi Telegram
            if (EnableTelegramNotifications) {
               string message = "Sell Limit berhasil dipasang pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
               message += "Entry: " + DoubleToString(lastSupplyZone.entry, _Digits) + "\n";
               message += "SL: " + DoubleToString(lastSupplyZone.sl, _Digits) + "\n";
               message += "TP: " + DoubleToString(lastSupplyZone.tp, _Digits) + "\n";
               message += "Lot Size: " + DoubleToString(lotSize, 2);
               SendTelegramMessage(message);
            }
         } else {
            Print("Gagal memasang Sell Limit. Error: ", GetLastError());
         }
      }
   }
   
   // Pasang order untuk area Demand
   if (lastDemandZone.valid && !lastDemandZone.used) {
      // Area Demand valid, pasang buy limit
      double lotSize = CalculateLotSize(ZONE_DEMAND);
      
      // Jika lotSize valid, pasang order
      if (lotSize > 0) {
         // Kirim pesan sebelum memasang order
         Print("Memasang Buy Limit pada ", _Symbol, " di harga ", DoubleToString(lastDemandZone.entry, _Digits),
               ", SL: ", DoubleToString(lastDemandZone.sl, _Digits), ", TP: ", DoubleToString(lastDemandZone.tp, _Digits));
         
         // Jika harga saat ini di atas entry, tidak perlu memasang order
         if (symbolInfo.Bid() > lastDemandZone.entry) {
            Print("Harga saat ini (", DoubleToString(symbolInfo.Bid(), _Digits), ") di atas level entry Demand (", 
                  DoubleToString(lastDemandZone.entry, _Digits), "). Tidak memasang order.");
            lastDemandZone.used = true;
            return;
         }
         
         // Pasang buy limit
         if (trade.BuyLimit(lotSize, lastDemandZone.entry, _Symbol, lastDemandZone.sl, lastDemandZone.tp)) {
            demandOrderTicket = trade.ResultOrder();
            lastDemandZone.used = true;
            
            Print("Buy Limit berhasil dipasang. Ticket: ", demandOrderTicket);
            
            // Kirim notifikasi Telegram
            if (EnableTelegramNotifications) {
               string message = "Buy Limit berhasil dipasang pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
               message += "Entry: " + DoubleToString(lastDemandZone.entry, _Digits) + "\n";
               message += "SL: " + DoubleToString(lastDemandZone.sl, _Digits) + "\n";
               message += "TP: " + DoubleToString(lastDemandZone.tp, _Digits) + "\n";
               message += "Lot Size: " + DoubleToString(lotSize, 2);
               SendTelegramMessage(message);
            }
         } else {
            Print("Gagal memasang Buy Limit. Error: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Cek status pending order                                         |
//+------------------------------------------------------------------+
void CheckPendingOrders() {
   // Cek status order Supply
   if (supplyOrderTicket > 0) {
      if (!OrderSelect(supplyOrderTicket)) {
         // Order tidak ditemukan, reset
         supplyOrderTicket = 0;
         ResetSupplyZone();
      }
   }
   
   // Cek status order Demand
   if (demandOrderTicket > 0) {
      if (!OrderSelect(demandOrderTicket)) {
         // Order tidak ditemukan, reset
         demandOrderTicket = 0;
         ResetDemandZone();
      }
   }
}

//+------------------------------------------------------------------+
//| Cek dan update trailing stop                                     |
//+------------------------------------------------------------------+
void CheckAndUpdateTrailingStop() {
   // Jika trailing stop tidak diaktifkan, return
   if (!EnableTrailingStop) return;
   
   // Ambil semua posisi terbuka
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      
      // Jika posisi ditemukan dan milik EA ini (magic number sama)
      if (ticket > 0 && PositionGetInteger(POSITION_MAGIC) == trade.RequestMagic()) {
         // Pastikan simbolnya sama
         if (PositionGetString(POSITION_SYMBOL) == _Symbol) {
            double positionOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentSL = PositionGetDouble(POSITION_SL);
            double currentTP = PositionGetDouble(POSITION_TP);
            double entrySlDistance = 0;
            
            ENUM_POSITION_TYPE positionType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            // Hitung jarak Entry-SL
            if (positionType == POSITION_TYPE_BUY) {
               entrySlDistance = MathAbs(positionOpenPrice - currentSL);
               
               // Cek apakah harga sudah mencapai level 1.0
               double level10 = positionOpenPrice + entrySlDistance;
               
               // Jika harga saat ini lebih besar dari level 1.0
               if (symbolInfo.Bid() > level10) {
                  // Hitung level trailing stop pada 0.764
                  double newSL = positionOpenPrice + (entrySlDistance * 0.764);
                  
                  // Jika SL baru lebih besar dari SL saat ini
                  if (newSL > currentSL) {
                     // Update trailing stop
                     if (trade.PositionModify(ticket, newSL, currentTP)) {
                        Print("Trailing Stop diperbarui untuk Buy position. Ticket: ", ticket, ", SL baru: ", DoubleToString(newSL, _Digits));
                        
                        // Kirim notifikasi Telegram
                        if (EnableTelegramNotifications) {
                           string message = "Trailing Stop diperbarui pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
                           message += "Ticket: " + IntegerToString(ticket) + "\n";
                           message += "SL Baru: " + DoubleToString(newSL, _Digits);
                           SendTelegramMessage(message);
                        }
                     } else {
                        Print("Gagal memperbarui Trailing Stop. Error: ", GetLastError());
                     }
                  }
               }
            } else if (positionType == POSITION_TYPE_SELL) {
               entrySlDistance = MathAbs(positionOpenPrice - currentSL);
               
               // Cek apakah harga sudah mencapai level 1.0
               double level10 = positionOpenPrice - entrySlDistance;
               
               // Jika harga saat ini lebih kecil dari level 1.0
               if (symbolInfo.Ask() < level10) {
                  // Hitung level trailing stop pada 0.764
                  double newSL = positionOpenPrice - (entrySlDistance * 0.764);
                  
                  // Jika SL baru lebih kecil dari SL saat ini
                  if (newSL < currentSL) {
                     // Update trailing stop
                     if (trade.PositionModify(ticket, newSL, currentTP)) {
                        Print("Trailing Stop diperbarui untuk Sell position. Ticket: ", ticket, ", SL baru: ", DoubleToString(newSL, _Digits));
                        
                        // Kirim notifikasi Telegram
                        if (EnableTelegramNotifications) {
                           string message = "Trailing Stop diperbarui pada " + _Symbol + ", " + TimeframeToString(Period()) + "\n";
                           message += "Ticket: " + IntegerToString(ticket) + "\n";
                           message += "SL Baru: " + DoubleToString(newSL, _Digits);
                           SendTelegramMessage(message);
                        }
                     } else {
                        Print("Gagal memperbarui Trailing Stop. Error: ", GetLastError());
                     }
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Hitung lot size berdasarkan risk management                      |
//+------------------------------------------------------------------+
double CalculateLotSize(ENUM_ZONE_TYPE zoneType) {
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double maxRiskAmount = accountBalance * (RiskPercentage / 100.0);
   double entryPrice = 0.0;
   double stopLossPrice = 0.0;
   
   // Set harga entry dan SL berdasarkan tipe area
   if (zoneType == ZONE_SUPPLY) {
      entryPrice = lastSupplyZone.entry;
      stopLossPrice = lastSupplyZone.sl;
   } else if (zoneType == ZONE_DEMAND) {
      entryPrice = lastDemandZone.entry;
      stopLossPrice = lastDemandZone.sl;
   }
   
   // Hitung jarak SL dalam pips
   double slDistance = MathAbs(entryPrice - stopLossPrice);
   double slDistanceInPips = slDistance / symbolInfo.Point();
   
   // Hitung nilai per pip
   double tickValue = symbolInfo.TickValue();
   double tickSize = symbolInfo.TickSize();
   double pipValue = (tickValue * symbolInfo.Point()) / tickSize;
   
   // Hitung lot size berdasarkan risiko maksimal
   double lotSize = maxRiskAmount / (slDistanceInPips * pipValue);
   
   // Normalisasi lot size berdasarkan aturan broker
   double minLot = symbolInfo.LotsMin();
   double maxLot = symbolInfo.LotsMax();
   double lotStep = symbolInfo.LotsStep();
   
   // Round lot size ke lot step terdekat
   lotSize = MathFloor(lotSize / lotStep) * lotStep;
   
   // Pastikan lot size berada dalam rentang yang diizinkan
   lotSize = MathMax(minLot, MathMin(maxLot, lotSize));
   
   return lotSize;
}

//+------------------------------------------------------------------+
//| Reset area Supply                                                |
//+------------------------------------------------------------------+
void ResetSupplyZone() {
   lastSupplyZone.valid = false;
   lastSupplyZone.used = false;
}

//+------------------------------------------------------------------+
//| Reset area Demand                                                |
//+------------------------------------------------------------------+
void ResetDemandZone() {
   lastDemandZone.valid = false;
   lastDemandZone.used = false;
}

//+------------------------------------------------------------------+
//| Kirim pesan ke Telegram                                          |
//+------------------------------------------------------------------+
void SendTelegramMessage(string message) {
   // Jika notifikasi Telegram tidak diaktifkan, return
   if (!EnableTelegramNotifications) return;
   
   // Jika token atau chat ID kosong, return
   if (TelegramToken == "" || TelegramChatID == "") {
      Print("Telegram Token atau Chat ID tidak dikonfigurasi");
      return;
   }
   
   string url = "https://api.telegram.org/bot" + TelegramToken + "/sendMessage";
   string params = "chat_id=" + TelegramChatID + "&text=" + message;
   
   // Kirim pesan ke Telegram
   char post[], result[];
   StringToCharArray(params, post);
   
   int res = WebRequest("POST", url, "Content-Type: application/x-www-form-urlencoded", 5000, post, result, NULL);
   
   if (res == -1) {
      Print("Error saat mengirim pesan Telegram: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Konversi timeframe ke string                                     |
//+------------------------------------------------------------------+
string TimeframeToString(ENUM_TIMEFRAMES timeframe) {
   switch (timeframe) {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return "Unknown";
   }
}
