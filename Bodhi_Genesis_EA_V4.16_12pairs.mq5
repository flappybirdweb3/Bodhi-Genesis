//+------------------------------------------------------------------+
//|                                     Bodhi_Genesis_EA_V4.16.mq5    |
//|                  🕉️ BODHI GENESIS V4.16 - 12 PAIRS SESSION FILTER |
//|                        Karma × Trend + 12 Pairs Support           |
//+------------------------------------------------------------------+
#property copyright "Bodhi Genesis"
#property link      "https://github.com/bodhi-genesis"
#property version   "4.16"
#property description "V4.16: 12 Pairs + Optimized Session Filter (7h-20h)"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

// ═══════════════════════════════════════════════════════════════════
// AUDIT MODULE - BỘ PHẬN KIỂM TOÁN
// ═══════════════════════════════════════════════════════════════════
#include "Bodhi_Audit_Module.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                   |
//+------------------------------------------------------------------+

input group "═══════════ 🕉️ GENERAL ═══════════"
input int      MagicNumber = 8402;
input string   TradeComment = "BodhiV3";
input bool     EnableTrading = true;
input string   ServerVersion = "V3.0";

input group "═══════════ 📊 CORE TRADING ═══════════"
input int      RSI_Period = 14;
input int      ADX_Period = 14;
input double   ADX_Strong_Trend = 25;
input bool     EnableSingularity = true;
input bool     EnablePriceAction = true;      // Detect PA patterns

input group "═══════════ 🌟 SINGULARITY (Karma-based) ═══════════"
input int      Karma_L1 = 20;       // MONK level
input int      Karma_L2 = 50;       // ARHAT level
input int      Karma_L3 = 100;      // BODHISATTVA level
input int      Karma_L4 = 200;      // BUDDHA level
input double   Mult_L1 = 1.0;       // MONK: x1.0
input double   Mult_L2 = 1.25;      // ARHAT: x1.25
input double   Mult_L3 = 1.5;       // BODHISATTVA: x1.5
input double   Mult_L4 = 2.0;       // BUDDHA: x2.0

input group "═══════════ 💰 FUND MANAGEMENT ═══════════"
input double   RiskPercent = 0.1;             // Base risk 0.1% (karma multiplier tự scale)
input double   MaxDailyLossPercent = 2.9;
input double   MaxTotalLossPercent = 9.0;
input double   DailyTargetPercent = 1.0;
input int      MaxConsecutiveLosses = 2;
input bool     EnableKarmaBonus = true;
input int      SilaStreakForBonus = 5;
input int      MeritForBonus = 50;

input group "═══════════ 📈 TREND ADAPTIVE SYSTEM ═══════════"
// Trend Strength Thresholds
input double   ADX_Strong = 40;               // ADX >= 40 = Strong trend
input double   ADX_Moderate = 25;             // ADX 25-40 = Moderate trend
input double   TEMA_Dist_Strong = 0.5;        // TEMA distance >= 0.5% = Strong
input double   TEMA_Dist_Moderate = 0.1;      // TEMA distance 0.1-0.5% = Moderate

// Trend Lot Multipliers (combined with Karma)
input double   Trend_Mult_Strong = 1.5;       // Strong trend: Lot x1.5
input double   Trend_Mult_Moderate = 1.0;     // Moderate trend: Lot x1.0
input double   Trend_Mult_Weak = 0.5;         // Weak trend: Lot x0.5
input double   Final_Mult_Max = 2.5;          // Max combined multiplier (Karma x Trend)
input double   Final_Mult_Min = 0.5;          // Min combined multiplier

// Trend Max Trades Per Day
input int      MaxTrades_Strong = 5;          // Strong: Max 5 trades
input int      MaxTrades_Moderate = 3;        // Moderate: Max 3 trades
input int      MaxTrades_Weak = 1;            // Weak: Max 1 trade

input group "═══════════ 🔍 AUDIT MODULE - BỘ PHẬN KIỂM TOÁN ═══════════"
input bool     EnableAudit = true;            // Enable Audit Module
input bool     AuditBlockCritical = true;    // Block trade on CRITICAL (future)
input bool     AuditShowDashboard = true;     // Show Audit status on Dashboard

// Audit Thresholds - Risk
input double   Audit_MaxDD_Warning = 2.0;     // DD Warning level (%)
input double   Audit_MaxDD_Danger = 4.0;      // DD Danger level (%)
input double   Audit_MaxDD_Critical = 6.0;    // DD Critical level (%)

// Audit Thresholds - Behavior
input double   Audit_MinConfidence = 60.0;    // Min AI Confidence (%)
input int      Audit_MinSecBetweenTrades = 300; // Min seconds between trades
input int      Audit_MaxConsecutiveLosses = 2;  // Max consecutive losses before warning

// Audit Thresholds - Market
input double   Audit_MaxSpreadPips = 5.0;     // Max spread (pips)
input double   Audit_MinADX = 20.0;           // Min ADX to trade

input group "═══════════ 🎯 TRADE SETTINGS ═══════════"
input int      StopLoss = 150;                 // Default SL (pips) if ATR not used
input int      TakeProfit = 300;               // Default TP (pips) if ATR not used
input int      MaxTradesPerDay = 10;

input group "═══════════ 🎭 DUAL SL/TP SYSTEM ═══════════"
// TRẬN THẬT (Shadow - EA quản lý)
input double   ATR_SL_Real = 1.5;              // Real SL = ATR × 1.5 (shadow)
input double   ATR_TP1_Real = 2.0;             // TP1 = ATR × 2.0 (partial close)
input double   ATR_TP2_Real = 4.0;             // TP2 = ATR × 4.0 (full close / trail)

// TRẬN GIẢ (Fake - cho sàn thấy, tránh bị hunt)
input double   ATR_SL_Fake = 8.0;              // Fake SL = ATR × 8 (xa, an toàn)
input double   ATR_TP_Fake = 12.0;             // Fake TP = ATR × 12 (không bao giờ chạm)

// PARTIAL CLOSE SETTINGS
input double   PartialCloseRatio = 0.6;        // Close 60% at TP1 (bỏ túi)
input double   PartialCloseRatio_Strong = 0.5; // Close 50% if trend strong
input bool     TrailAfterPartial = true;       // Trail remaining after partial
input double   TrailATR_Multiplier = 1.0;      // Trail distance = ATR × 1.0

// ATR SETTINGS
input int      ATR_Period = 14;                // ATR period
input ENUM_TIMEFRAMES ATR_Timeframe = PERIOD_H4; // ATR timeframe

input group "═══════════ 🛡️ RISK MANAGEMENT ═══════════"
input bool     UseSmartExit = true;
input int      SmartExit_RSI_Bull = 70;
input int      SmartExit_RSI_Bear = 30;
input int      SmartExit_MinHoldMins = 5;    // Minimum hold time before Smart Exit (minutes)
input int      MaxHoldHours = 8;             // Force close after X hours (0 = disabled)
input bool     UseDualSLTP = true;             // Enable Dual SL/TP system
input bool     UseSpreadProtection = true;
input double   SpreadMultiplier = 1.5;
input double   PanicThreshold = 1.0;           // FSS-5: Panic close if equity drops X% in 1 tick

input group "═══════════ 🕐 SESSION FILTER (12 PAIRS) ═══════════"
input bool     UseSessionFilter = true;

// EUR Cluster (London + NY overlap)
input int      EURUSD_StartHour = 7;
input int      EURUSD_EndHour = 20;
input int      EURGBP_StartHour = 7;
input int      EURGBP_EndHour = 16;
input int      EURJPY_StartHour = 7;
input int      EURJPY_EndHour = 16;

// GBP Cluster (London + NY overlap)
input int      GBPUSD_StartHour = 7;
input int      GBPUSD_EndHour = 20;
input int      GBPJPY_StartHour = 7;
input int      GBPJPY_EndHour = 16;

// USD Majors (Multi-session)
input int      USDJPY_StartHour = 0;
input int      USDJPY_EndHour = 16;
input int      USDCAD_StartHour = 13;
input int      USDCAD_EndHour = 20;
input int      AUDUSD_StartHour = 0;
input int      AUDUSD_EndHour = 9;

// Commodities (NY session - best liquidity)
input int      XAUUSD_StartHour = 13;
input int      XAUUSD_EndHour = 20;
input int      XAGUSD_StartHour = 13;
input int      XAGUSD_EndHour = 20;

// Indices (US equity hours)
input int      US30_StartHour = 13;
input int      US30_EndHour = 20;

// Oceania (Early Asia)
input int      NZDUSD_StartHour = 0;
input int      NZDUSD_EndHour = 9;


input group "═══════════ 📰 NEWS FILTER ═══════════"
input bool     UseNewsFilter = true;              // Enable news filter
input int      NewsMinutesBefore = 30;            // Stop trading X minutes before news
input int      NewsMinutesAfter = 60;             // Resume trading X minutes after news
input bool     FilterHighImpact = true;           // Filter High impact news (RED)
input bool     FilterMediumImpact = false;        // Filter Medium impact news (ORANGE)
input string   NewsCurrencies = "USD,EUR,GBP";    // Currencies to monitor
// Note: Add https://ec.forexprostools.com to MT5 Options > Expert Advisors > Allow WebRequest

input group "═══════════ 🇺🇸 US30 / INDICES ═══════════"
input bool     Use_Indices_Logic = false;
input double   US30_RiskPercent = 0.2;         // 0.2% cho US30
input double   US30_Equity_Per_Lot = 10000;
input bool     Auto_Adjust_To_MinLot = true;
input double   Max_Lot_Indices = 3.0;
input double   FSS5_Sensitivity = 0.5;

input group "═══════════ 🥇 XAUUSD / GOLD ═══════════"
input double   XAUUSD_Max_Lot = 0.1;           // Max lot cho XAUUSD (safety - volatile)
input double   XAUUSD_Equity_Per_Lot = 5000;   // $5000 equity per 0.1 lot

input group "═══════════ 🎨 DASHBOARD ═══════════"
input bool     ShowDashboard = true;
input int      DashboardX = 10;
input int      DashboardY = 30;
input color    ColorTitle = clrGold;
input color    ColorActive = clrLime;
input color    ColorWarning = clrYellow;
input color    ColorDanger = clrRed;
input color    ColorInfo = clrWhite;
input color    ColorMuted = clrGray;

input group "═══════════ 🌐 SERVER CONNECTION ═══════════"
input bool     UseAIServer = true;            // Connect to AI server
input string   ServerHost = "127.0.0.1";      // Server host (use IP, not localhost)
input int      ServerPort = 9999;             // Server port
input int      ServerTimeout = 5000;          // Timeout (ms)
input int      SignalCheckInterval = 60;      // Check signal every N seconds

input group "═══════════ 💓 HEARTBEAT & AI FILTER ═══════════"
input bool     EnableServerConnection = true;
input string   ServerURL = "http://127.0.0.1:9999";  // Server URL (use IP, not localhost)
input int      HeartbeatInterval = 60;               // Seconds between heartbeats (was 30)
input int      HeartbeatTimeout = 10;                // Timeout in seconds
input int      MaxMissedHeartbeats = 3;              // Max missed before alert
input double   MinAIConfidence = 55.0;               // Min confidence to trade (%)
input bool     UseFallbackSignal = false;            // V4.14: Use local signal when AI=HOLD? (THUAN THIEN = false)
input int      CooldownAfterLoss = 15;               // Minutes to wait after loss (anti-revenge)

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                   |
//+------------------------------------------------------------------+

CTrade trade;
CPositionInfo posInfo;
CAccountInfo accInfo;

// Indicator handles
int h_RSI_M5, h_RSI_M15, h_RSI_H1, h_RSI_H4;
int h_ADX_M5, h_ADX_H1, h_ADX_H4;
int h_ATR;
int h_EMA_D1;  // EMA 50 D1 for TEMA calculation

// Indicator cache (reduce CopyBuffer calls)
datetime LastIndicatorCacheTime = 0;
double Cache_RSI_M5 = 50;
double Cache_RSI_M15 = 50;
double Cache_RSI_H1 = 50;
double Cache_RSI_H4 = 50;
double Cache_ADX_M5 = 0;
double Cache_ADX_H1 = 0;
double Cache_ADX_H4 = 0;
double Cache_ATR_H4 = 0;

// TEMA D1 cache (expensive calculation)
datetime LastTemaBarTime = 0;
double CachedTemaD1 = 0;

// Spread tracking
double SpreadHistory[];
int SpreadIndex = 0;
double AverageSpread = 0;
double CurrentSpread = 0;

// Daily tracking
double DayStartEquity;
double DayStartBalance;
int DailyTrades;           // Trades opened today (for MaxTradesPerDay limit)
int DailyTradesCompleted;  // Trades completed today (for dashboard W/L stats)
int DailyWins;
int DailyLosses;
double DailyPnL;
double DailyDrawdown;
bool DailyStopped;
bool Killed;

// Karma tracking (Enhanced)
int ConsecutiveLosses;
int Karma;
int SilaStreak;
int TotalMerit;
int TotalDemerit;
bool HasBonus;
datetime LastLossTime = 0;        // Thời điểm loss gần nhất (cho Revenge detection)
datetime LastTradeTime = 0;       // Thời điểm trade gần nhất (cho FOMO detection)
double LastTradeConfidence = 0;   // Confidence của trade gần nhất
bool LastTradeWasLoss = false;    // Trade trước có phải loss không

// Tam Bao
int Chua, Kinh, Tang;
double PhatBalance, PhapBalance, TangBalance;

// News Filter
datetime NextNewsTime = 0;
string NextNewsTitle = "";
int NextNewsImpact = 0;
datetime LastNewsCheck = 0;

// DUAL SL/TP System
// TRẬN THẬT (Shadow - EA quản lý)
double RealSL_Price = 0;      // SL thật (shadow)
double RealTP1_Price = 0;     // TP1 - partial close
double RealTP2_Price = 0;     // TP2 - full close / trail
double TrailSL_Price = 0;     // Trailing SL after partial
int Position_Type = 0;        // 1=BUY, -1=SELL
double Position_OpenPrice = 0;
bool IsTrailing = false;      // Đang trail sau partial close

// Partial close
bool PartialClosed = false;

// AI Status
bool AIConnected = false;
double AIConfidence = 0;
string LastSignal = "HOLD";
int TotalPredictions = 0;
datetime LastPredictionTime;
// LastTradeTime và LastLossTime đã định nghĩa ở Karma tracking section

// Heartbeat tracking
datetime LastHeartbeatTime = 0;
datetime LastHeartbeatSuccess = 0;
int MissedHeartbeats = 0;
bool ServerOnline = false;
string ServerStatus = "OFFLINE";
string AIModelStatus = "UNKNOWN";

// Position Cache (avoid multiple PositionsTotal loops)
static bool CachedHasPosition = false;
static datetime LastPositionCheck = 0;

// ═══════════════════════════════════════════════════════════════
// TRUNG ĐẠO KARMA TRACKING - For Server Sync
// ═══════════════════════════════════════════════════════════════
datetime TradeEntryTime = 0;        // Thời điểm mở trade
double TradeEntryPrice = 0;         // Giá entry
double TradeMaxDrawdownPips = 0;    // Max drawdown trong trade (pips)
bool TradeHadSL = false;            // Trade có đặt SL không
bool TradeHadTP = false;            // Trade có đặt TP không

int LastTradeDay;

//+------------------------------------------------------------------+
//| Karma Functions                                                    |
//+------------------------------------------------------------------+
string GetKarmaLevel()
{
   if(Karma >= Karma_L4) return "BUDDHA";
   if(Karma >= Karma_L3) return "BODHISATTVA";
   if(Karma >= Karma_L2) return "ARHAT";
   if(Karma >= Karma_L1) return "MONK";
   return "NOVICE";
}

double GetKarmaMultiplier()
{
   if(Karma >= Karma_L4) return Mult_L4;
   if(Karma >= Karma_L3) return Mult_L3;
   if(Karma >= Karma_L2) return Mult_L2;
   if(Karma >= Karma_L1) return Mult_L1;
   return 1.0;
}

//+------------------------------------------------------------------+
//| Trend Strength Functions - ADAPTIVE PARAMS                        |
//+------------------------------------------------------------------+
// Global trend tracking
string CurrentTrendStrength = "MODERATE";
double CurrentTrendMult = 1.0;
int CurrentMaxTrades = 3;
double CurrentSL_ATR_Mult = 1.5;
double CurrentTP_ATR_Mult = 2.0;

string GetTrendStrength()
{
   /*
    * Trend Strength Classification:
    * - STRONG: ADX >= 40 + TEMA distance >= 0.5% + RSI extreme
    * - MODERATE: ADX 25-40 + TEMA distance 0.1-0.5%
    * - WEAK: ADX < 25 OR TEMA distance < 0.1%
    */
   
   double adx_h4 = GetADX(h_ADX_H4);
   double rsi_h4 = GetRSI(h_RSI_H4);
   
   // Get TEMA distance
   double tema_d1 = GetTEMA_D1();
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double tema_dist_pct = 0;
   if(tema_d1 > 0)
      tema_dist_pct = MathAbs((currentPrice - tema_d1) / tema_d1 * 100);
   
   // STRONG: High ADX + Far from TEMA + RSI extreme
   if(adx_h4 >= ADX_Strong && tema_dist_pct >= TEMA_Dist_Strong)
   {
      if(rsi_h4 > 60 || rsi_h4 < 40)
         return "STRONG";
   }
   
   // MODERATE: Medium ADX or medium TEMA distance
   if(adx_h4 >= ADX_Moderate && tema_dist_pct >= TEMA_Dist_Moderate)
   {
      return "MODERATE";
   }
   
   // WEAK: Low ADX or too close to TEMA
   return "WEAK";
}

double GetTrendMultiplier()
{
   string strength = GetTrendStrength();
   
   if(strength == "STRONG")
      return Trend_Mult_Strong;
   else if(strength == "MODERATE")
      return Trend_Mult_Moderate;
   else
      return Trend_Mult_Weak;
}

int GetTrendMaxTrades()
{
   string strength = GetTrendStrength();
   
   if(strength == "STRONG")
      return MaxTrades_Strong;
   else if(strength == "MODERATE")
      return MaxTrades_Moderate;
   else
      return MaxTrades_Weak;
}

void GetTrendAdaptiveSLTP(double &sl_mult, double &tp1_mult, double &tp2_mult)
{
   /*
    * Adaptive SL/TP based on Trend Strength:
    * - STRONG: Wider SL (2.0x), Further TP (4.0x) - ride the trend
    * - MODERATE: Normal SL (1.5x), Medium TP (2.0x)
    * - WEAK: Tight SL (1.0x), Short TP (1.2x) - quick in/out
    */
   
   string strength = GetTrendStrength();
   
   if(strength == "STRONG")
   {
      sl_mult = 2.0;
      tp1_mult = 3.0;
      tp2_mult = 5.0;
   }
   else if(strength == "MODERATE")
   {
      sl_mult = 1.5;
      tp1_mult = 2.0;
      tp2_mult = 4.0;
   }
   else  // WEAK
   {
      sl_mult = 1.0;
      tp1_mult = 1.2;
      tp2_mult = 2.0;
   }
}

double GetFinalMultiplier()
{
   /*
    * Final Lot Multiplier = Karma × Trend (with caps)
    * 
    * Karma = Internal factor (trader's track record)
    * Trend = External factor (market condition)
    * 
    * Cap: 0.5 - 2.5 to prevent extreme positions
    */
   
   double karmaMult = GetKarmaMultiplier();
   double trendMult = GetTrendMultiplier();
   
   double finalMult = karmaMult * trendMult;
   
   // Apply caps
   if(finalMult > Final_Mult_Max)
      finalMult = Final_Mult_Max;
   if(finalMult < Final_Mult_Min)
      finalMult = Final_Mult_Min;
   
   return finalMult;
}

void UpdateTrendParams()
{
   /*
    * Update all trend-adaptive parameters
    * Called at start of each tick/signal check
    */
   
   CurrentTrendStrength = GetTrendStrength();
   CurrentTrendMult = GetTrendMultiplier();
   CurrentMaxTrades = GetTrendMaxTrades();
   
   double sl, tp1, tp2;
   GetTrendAdaptiveSLTP(sl, tp1, tp2);
   CurrentSL_ATR_Mult = sl;
   CurrentTP_ATR_Mult = tp1;
}

string GetPathName()
{
   if(Killed) return "NIRVANA";
   if(DailyStopped) return "RESTING";
   
   // Check cooldown (anti-revenge)
   if(LastLossTime > 0 && CooldownAfterLoss > 0)
   {
      int minutesSinceLoss = (int)((TimeCurrent() - LastLossTime) / 60);
      if(minutesSinceLoss < CooldownAfterLoss)
         return "COOLDOWN";
   }
   
   if(HasPosition()) return "TRADING";
   if(!IsInSession()) return "SLEEPING";
   return "MEDITATING";
}

string GetPathState()
{
   string path = GetPathName();
   if(path == "NIRVANA") return "Giai Thoat";
   if(path == "RESTING") return "Nghi Ngoi";
   if(path == "COOLDOWN") return "Binh Tinh";  // Calming down
   if(path == "TRADING") return "Hanh Dong";
   if(path == "SLEEPING") return "Ngu Nghi";
   return "Thien Dinh";
}

//+------------------------------------------------------------------+
//| Dashboard Display - OPTIMIZED                                      |
//+------------------------------------------------------------------+
static bool DashboardCreated = false;
static datetime LastDashUpdate = 0;
const int DASH_UPDATE_INTERVAL = 1;  // Update every 1 second

void UpdateDashboard()
{
   if(!ShowDashboard) return;
   
   // Throttle: Update max 1x/second
   if(TimeCurrent() - LastDashUpdate < DASH_UPDATE_INTERVAL) return;
   LastDashUpdate = TimeCurrent();
   
   string p = "BD_";
   
   // Create labels ONCE
   if(!DashboardCreated)
   {
      CreateDashboardLabels();
      DashboardCreated = true;
   }
   
   // Update text only (no delete/recreate)
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   DailyDrawdown = (DayStartEquity > 0) ? MathMax(0, (DayStartEquity - equity) / DayStartEquity * 100) : 0;
   
   // Update colors based on state
   color ddCol = (DailyDrawdown > MaxDailyLossPercent * 0.7) ? ColorDanger : (DailyDrawdown > MaxDailyLossPercent * 0.5) ? ColorWarning : ColorActive;
   
   ObjectSetString(0, p+"dd", OBJPROP_TEXT, StringFormat("Daily Drawdown: %.2f%% / %.1f%%", DailyDrawdown, MaxDailyLossPercent));
   ObjectSetInteger(0, p+"dd", OBJPROP_COLOR, ddCol);
   
   ObjectSetString(0, p+"pnl", OBJPROP_TEXT, StringFormat("Today's Profit: $%.2f", DailyPnL));
   ObjectSetInteger(0, p+"pnl", OBJPROP_COLOR, DailyPnL >= 0 ? ColorActive : ColorDanger);
   
   ObjectSetString(0, p+"allow", OBJPROP_TEXT, StringFormat("Allowable Loss Today: $%.2f", DayStartEquity * MaxDailyLossPercent / 100));
   
   double maxLeft = MathMax(0, DayStartBalance * MaxTotalLossPercent / 100 - MathMax(0, DayStartBalance - equity));
   ObjectSetString(0, p+"max", OBJPROP_TEXT, StringFormat("Max Permitted Loss Left: $%.2f", maxLeft));
   
   int maxLoss = GetMaxConsecutiveLoss();
   ObjectSetString(0, p+"cons", OBJPROP_TEXT, StringFormat("Consecutive Loss: %d / %d (This pair)", ConsecutiveLosses, maxLoss));
   ObjectSetInteger(0, p+"cons", OBJPROP_COLOR, (ConsecutiveLosses >= maxLoss - 1) ? ColorDanger : ColorInfo);
   
   double wr = (DailyTradesCompleted > 0) ? (double)DailyWins / DailyTradesCompleted * 100 : 0;
   ObjectSetString(0, p+"trd", OBJPROP_TEXT, StringFormat("Trades: %d (W:%d L:%d | %.0f%%)", DailyTradesCompleted, DailyWins, DailyLosses, wr));
   
   ObjectSetString(0, p+"spr", OBJPROP_TEXT, StringFormat("Spread: %.1f (Avg: %.1f)", CurrentSpread, AverageSpread));
   ObjectSetInteger(0, p+"spr", OBJPROP_COLOR, (CurrentSpread > AverageSpread * SpreadMultiplier && AverageSpread > 0) ? ColorDanger : ColorActive);
   
   string kLvl = GetKarmaLevel();
   double kMult = GetKarmaMultiplier();
   color kCol = clrGray;
   if(kLvl == "BUDDHA") kCol = clrMagenta;
   else if(kLvl == "BODHISATTVA") kCol = clrGold;
   else if(kLvl == "ARHAT") kCol = clrCyan;
   else if(kLvl == "MONK") kCol = ColorActive;
   ObjectSetString(0, p+"krm", OBJPROP_TEXT, StringFormat("Karma: %s (Vol x%.2f)", kLvl, kMult));
   ObjectSetInteger(0, p+"krm", OBJPROP_COLOR, kCol);
   
   // Trend Adaptive Display
   UpdateTrendParams();
   double tMult = GetTrendMultiplier();
   double fMult = GetFinalMultiplier();
   color tCol = clrGray;
   if(CurrentTrendStrength == "STRONG") tCol = clrLime;
   else if(CurrentTrendStrength == "MODERATE") tCol = clrYellow;
   else tCol = clrOrange;
   ObjectSetString(0, p+"tnd", OBJPROP_TEXT, StringFormat("Trend: %s (x%.2f) | Final: x%.2f | Max: %d", 
         CurrentTrendStrength, tMult, fMult, CurrentMaxTrades));
   ObjectSetInteger(0, p+"tnd", OBJPROP_COLOR, tCol);
   
   // 🔍 AUDIT STATUS DISPLAY
   if(EnableAudit && AuditShowDashboard)
   {
      string auditStr = g_Audit.GetAuditSummary();
      color auditCol = clrLime;
      if(g_Audit.HasCriticalIssues()) auditCol = clrRed;
      else if(g_Audit.HasDangerIssues()) auditCol = clrOrange;
      else if(g_Audit.HasWarnings()) auditCol = clrGold;
      
      ObjectSetString(0, p+"aud", OBJPROP_TEXT, StringFormat("🔍 Audit: %s", auditStr));
      ObjectSetInteger(0, p+"aud", OBJPROP_COLOR, auditCol);
      
      // Show last critical message if any
      string critMsg = g_Audit.GetLastCriticalMessage();
      if(critMsg != "")
      {
         ObjectSetString(0, p+"audmsg", OBJPROP_TEXT, StringFormat("   └─ %s", critMsg));
         ObjectSetInteger(0, p+"audmsg", OBJPROP_COLOR, clrRed);
      }
      else
      {
         ObjectSetString(0, p+"audmsg", OBJPROP_TEXT, " ");  // Space instead of empty to hide "Label"
      }
   }
   else
   {
      // Hide audit labels when disabled
      ObjectSetString(0, p+"aud", OBJPROP_TEXT, " ");
      ObjectSetString(0, p+"audmsg", OBJPROP_TEXT, " ");
   }
   
   ObjectSetString(0, p+"pth", OBJPROP_TEXT, StringFormat("Path: %s (%s)", GetPathName(), GetPathState()));
   ObjectSetString(0, p+"tam", OBJPROP_TEXT, StringFormat("Tam Bao: Chua:%d Kinh:%d Tang:%d", Chua, Kinh, Tang));
   
   string khatStr = "□ KHAT THUC...";
   if(CachedHasPosition) khatStr = "□ DANG TRADE...";
   else if(DailyStopped) khatStr = "□ NGHI NGOI...";
   else if(IsNewsTime()) khatStr = "□ TIN DO: " + GetNewsStatus();
   else if(!IsInSession()) khatStr = "□ CHO PHIEN...";
   ObjectSetString(0, p+"kht", OBJPROP_TEXT, khatStr);
   ObjectSetInteger(0, p+"kht", OBJPROP_COLOR, IsNewsTime() ? ColorWarning : ColorMuted);
   
   double targetPct = (DayStartEquity > 0) ? DailyPnL / DayStartEquity * 100 : 0;
   ObjectSetString(0, p+"tgt", OBJPROP_TEXT, StringFormat("Target: %.2f%% / %.1f%%", targetPct, DailyTargetPercent));
   ObjectSetInteger(0, p+"tgt", OBJPROP_COLOR, targetPct >= DailyTargetPercent ? ColorActive : ColorInfo);
   
   double needAmt = MathMax(0, DayStartEquity * DailyTargetPercent / 100 - DailyPnL);
   ObjectSetString(0, p+"ned", OBJPROP_TEXT, StringFormat("Need: $%.2f", needAmt));
   
   // AI Status
   string aiStr = StringFormat("□ AI Heart: %.1f%% | %s | Last: %s", AIConfidence, AIConnected ? "CONNECTED" : "OFFLINE", LastSignal);
   ObjectSetString(0, p+"ai", OBJPROP_TEXT, aiStr);
   ObjectSetInteger(0, p+"ai", OBJPROP_COLOR, AIConnected ? ColorInfo : ColorDanger);
   
   int minsAgo = (int)((TimeCurrent() - LastPredictionTime) / 60);
   ObjectSetString(0, p+"pred", OBJPROP_TEXT, StringFormat("Predictions: %d | Latest: %.1f%% | %dm ago", TotalPredictions, AIConfidence, minsAgo));
   
   // NO ChartRedraw() - MT5 auto refreshes
}

void CreateDashboardLabels()
{
   int y = DashboardY;
   int lh = 18;
   string p = "BD_";
   
   // Header (static)
   CreateLabel(p+"h1", DashboardX, y, "QUA KHO KHONG TRUY TAM", ColorTitle, 10); y += lh;
   CreateLabel(p+"h2", DashboardX, y, "TUONG LAI KHONG UOC VONG", ColorTitle, 10); y += lh;
   CreateLabel(p+"h3", DashboardX, y, "HAY NHIN VAO HIEN TAI:", ColorTitle, 10); y += lh + 5;
   
   // System Info
   CreateLabel(p+"sym", DashboardX, y, StringFormat("BODHI %s (M:%d)", _Symbol, MagicNumber), ColorInfo, 10); y += lh;
   CreateLabel(p+"sta", DashboardX, y, "Status: " + (EnableTrading ? "ACTIVE" : "PAUSED"), EnableTrading ? ColorActive : ColorWarning, 10); y += lh;
   CreateLabel(p+"srv", DashboardX, y, "Server: ONLINE (" + ServerVersion + ")", ColorActive, 10); y += lh;
   
   // Dynamic labels (will be updated)
   CreateLabel(p+"ai", DashboardX, y, "", ColorInfo, 10); y += lh;
   CreateLabel(p+"pred", DashboardX, y, "", ColorMuted, 10); y += lh + 5;
   CreateLabel(p+"dd", DashboardX, y, "", ColorActive, 10); y += lh;
   CreateLabel(p+"pnl", DashboardX, y, "", ColorActive, 10); y += lh;
   CreateLabel(p+"allow", DashboardX, y, "", ColorWarning, 10); y += lh;
   CreateLabel(p+"max", DashboardX, y, "", ColorInfo, 10); y += lh;
   CreateLabel(p+"cons", DashboardX, y, "", ColorInfo, 10); y += lh + 5;
   CreateLabel(p+"trd", DashboardX, y, "", ColorInfo, 10); y += lh;
   CreateLabel(p+"spr", DashboardX, y, "", ColorActive, 10); y += lh;
   CreateLabel(p+"krm", DashboardX, y, "", clrGray, 10); y += lh;
   CreateLabel(p+"tnd", DashboardX, y, "", ColorInfo, 10); y += lh;  // Trend Adaptive
   CreateLabel(p+"aud", DashboardX, y, "", ColorInfo, 10); y += lh;  // Audit Status
   CreateLabel(p+"audmsg", DashboardX, y, "", clrRed, 9); y += lh;   // Audit Message
   CreateLabel(p+"pth", DashboardX, y, "", ColorInfo, 10); y += lh;
   CreateLabel(p+"tam", DashboardX, y, "", ColorWarning, 10); y += lh + 5;
   CreateLabel(p+"kht", DashboardX, y, "", ColorWarning, 10); y += lh;
   CreateLabel(p+"tgt", DashboardX, y, "", ColorInfo, 10); y += lh;
   CreateLabel(p+"ned", DashboardX, y, "", ColorInfo, 10);
}

void CreateLabel(string name, int x, int y, string text, color clr, int size)
{
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
}

//+------------------------------------------------------------------+
//| Heartbeat - Server Connection                                      |
//+------------------------------------------------------------------+
void CheckHeartbeat()
{
   if(!EnableServerConnection) 
   {
      // Offline mode - simulate
      AIConnected = false;
      ServerOnline = false;
      ServerStatus = "DISABLED";
      return;
   }
   
   // Check if it's time for heartbeat
   if(TimeCurrent() - LastHeartbeatTime < HeartbeatInterval)
      return;
   
   LastHeartbeatTime = TimeCurrent();
   
   // Send heartbeat request
   bool success = SendHeartbeat();
   
   if(success)
   {
      MissedHeartbeats = 0;
      LastHeartbeatSuccess = TimeCurrent();
      ServerOnline = true;
      AIConnected = true;
      ServerStatus = "ONLINE";
      
      Print("💓 Heartbeat OK | Server: ", ServerStatus, " | AI: ", AIModelStatus, " | Confidence: ", AIConfidence, "%");
   }
   else
   {
      MissedHeartbeats++;
      ServerOnline = false;
      AIConnected = false;
      ServerStatus = "OFFLINE";
      
      // Alert based on missed heartbeats
      if(MissedHeartbeats == 1)
      {
         Print("⚠️ [HEARTBEAT] Missed heartbeat #1 - Server may be slow");
      }
      else if(MissedHeartbeats == 2)
      {
         Print("⚠️ [HEARTBEAT] Missed heartbeat #2 - Connection unstable!");
      }
      else if(MissedHeartbeats >= MaxMissedHeartbeats)
      {
         Print("🚨 [HEARTBEAT] CRITICAL: ", MissedHeartbeats, " missed heartbeats!");
         Print("🚨 [HEARTBEAT] Server URL: ", ServerURL);
         Print("🚨 [HEARTBEAT] Last successful: ", TimeToString(LastHeartbeatSuccess));
         Print("🚨 [HEARTBEAT] AI predictions UNAVAILABLE - Trading with local signals only!");
         
         // Alert popup every 5 missed
         if(MissedHeartbeats % 5 == 0)
         {
            Alert("BODHI: Server connection lost! ", MissedHeartbeats, " missed heartbeats. Check server!");
         }
      }
   }
}

bool SendHeartbeat()
{
   string url = ServerURL + "/api/heartbeat";
   string headers = "Content-Type: application/json\r\n";
   
   // Build request body
   string body = StringFormat(
      "{\"symbol\":\"%s\",\"magic\":%d,\"equity\":%.2f,\"time\":\"%s\"}",
      _Symbol, MagicNumber, AccountInfoDouble(ACCOUNT_EQUITY), TimeToString(TimeCurrent())
   );
   
   char post[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(body, post, 0, StringLen(body));
   ArrayResize(post, StringLen(body));
   
   // Reset WebRequest
   ResetLastError();
   
   int timeout = HeartbeatTimeout * 1000;  // Convert to ms
   int res = WebRequest("POST", url, headers, timeout, post, result, resultHeaders);
   
   if(res == -1)
   {
      int error = GetLastError();
      if(error == 4060)
      {
         // URL not allowed - need to add in MT5 Options
         if(MissedHeartbeats == 0)
         {
            Print("❌ [HEARTBEAT] WebRequest failed! Error 4060");
            Print("❌ [HEARTBEAT] Please add '", ServerURL, "' to Tools > Options > Expert Advisors > Allow WebRequest");
         }
      }
      else
      {
         Print("❌ [HEARTBEAT] WebRequest failed! Error: ", error);
      }
      return false;
   }
   
   if(res != 200)
   {
      Print("❌ [HEARTBEAT] Server returned HTTP ", res);
      return false;
   }
   
   // Parse response
   string response = CharArrayToString(result);
   
   // Extract AI confidence from response
   // Expected: {"status":"ok","ai_confidence":98.7,"ai_model":"ready","predictions":48}
   if(!ParseHeartbeatResponse(response))
   {
      Print("⚠️ [HEARTBEAT] Failed to parse response: ", response);
      return false;
   }
   
   return true;
}

bool ParseHeartbeatResponse(string response)
{
   // Simple JSON parsing
   // Look for "ai_confidence":XX.X
   int confPos = StringFind(response, "\"ai_confidence\":");
   if(confPos >= 0)
   {
      int start = confPos + 16;
      int end = StringFind(response, ",", start);
      if(end < 0) end = StringFind(response, "}", start);
      if(end > start)
      {
         string confStr = StringSubstr(response, start, end - start);
         AIConfidence = StringToDouble(confStr);
      }
   }
   
   // Look for "ai_model":"ready"
   int modelPos = StringFind(response, "\"ai_model\":");
   if(modelPos >= 0)
   {
      int start = modelPos + 12;
      int end = StringFind(response, "\"", start);
      if(end > start)
      {
         AIModelStatus = StringSubstr(response, start, end - start);
      }
   }
   
   // Look for "predictions":XX
   int predPos = StringFind(response, "\"predictions\":");
   if(predPos >= 0)
   {
      int start = predPos + 14;
      int end = StringFind(response, ",", start);
      if(end < 0) end = StringFind(response, "}", start);
      if(end > start)
      {
         string predStr = StringSubstr(response, start, end - start);
         TotalPredictions = (int)StringToInteger(predStr);
      }
   }
   
   // Look for "last_signal":"BUY/SELL/HOLD"
   int sigPos = StringFind(response, "\"last_signal\":");
   if(sigPos >= 0)
   {
      // "last_signal":" = 15 chars, then find the closing "
      int start = sigPos + 15;  // Position after ":"
      // Skip the opening quote if present
      if(StringGetCharacter(response, start) == '"')
         start++;
      int end = StringFind(response, "\"", start);
      if(end > start)
      {
         LastSignal = StringSubstr(response, start, end - start);
         AIModelStatus = LastSignal;  // Update AIModelStatus for display
         Print("📡 Parsed signal: ", LastSignal);
      }
   }
   else
   {
      // Fallback: try to find signal in different format "signal":1 or "signal":-1
      int numSigPos = StringFind(response, "\"signal\":");
      if(numSigPos >= 0)
      {
         int start = numSigPos + 9;
         string sigChar = StringSubstr(response, start, 2);
         if(StringFind(sigChar, "1") >= 0 && StringFind(sigChar, "-") < 0)
         {
            LastSignal = "BUY";
            AIModelStatus = "BUY";
         }
         else if(StringFind(sigChar, "-1") >= 0)
         {
            LastSignal = "SELL";
            AIModelStatus = "SELL";
         }
         else
         {
            LastSignal = "HOLD";
            AIModelStatus = "HOLD";
         }
      }
   }
   
   // Check status
   if(StringFind(response, "\"status\":\"ok\"") >= 0)
   {
      LastPredictionTime = TimeCurrent();
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Request AI Signal from Server - FULL TỨ THỜI + KARMA DATA        |
//+------------------------------------------------------------------+
int GetAISignal()
{
   if(!EnableServerConnection || !ServerOnline)
      return 0;  // No AI signal available
   
   string url = ServerURL + "/api/signal";
   string headers = "Content-Type: application/json\r\n";
   
   // ═══════════════════════════════════════════════════════════════
   // FULL TỨ THỜI DATA - D1 → H4 → H1 → M15 → M5
   // ═══════════════════════════════════════════════════════════════
   double rsi_m5 = GetRSI(h_RSI_M5);
   double rsi_m15 = GetRSI(h_RSI_M15);
   double rsi_h1 = GetRSI(h_RSI_H1);
   double rsi_h4 = GetRSI(h_RSI_H4);
   double adx_m5 = GetADX(h_ADX_M5);
   double adx_h4 = GetADX(h_ADX_H4);
   
   // TEMA D1 - Main Trend (Tự Nhiên)
   int mainTrend = GetMainTrend_D1();
   double tema_d1 = GetTEMA_D1();
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // ═══════════════════════════════════════════════════════════════
   // KARMA & RISK DATA - For PPO Validator
   // ═══════════════════════════════════════════════════════════════
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double currentDD = (DayStartEquity > 0) ? (DayStartEquity - equity) / DayStartEquity * 100 : 0;
   
   // Build FULL request body
   string body = StringFormat(
      "{"
      "\"symbol\":\"%s\","
      "\"rsi_m5\":%.2f,\"rsi_m15\":%.2f,\"rsi_h1\":%.2f,\"rsi_h4\":%.2f,"
      "\"adx_m5\":%.2f,\"adx_h4\":%.2f,"
      "\"main_trend\":%d,\"tema_d1\":%.5f,\"current_price\":%.5f,"
      "\"karma\":%d,\"sila_streak\":%d,"
      "\"trades_today\":%d,\"consecutive_losses\":%d,"
      "\"current_drawdown\":%.2f,\"daily_pnl\":%.2f,"
      "\"equity\":%.2f"
      "}",
      _Symbol,
      rsi_m5, rsi_m15, rsi_h1, rsi_h4,
      adx_m5, adx_h4,
      mainTrend, tema_d1, currentPrice,
      Karma, SilaStreak,
      DailyTrades, ConsecutiveLosses,
      currentDD, DailyPnL,
      equity
   );
   
   char post[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(body, post, 0, StringLen(body));
   ArrayResize(post, StringLen(body));
   
   int res = WebRequest("POST", url, headers, HeartbeatTimeout * 1000, post, result, resultHeaders);
   
   if(res != 200)
   {
      Print("⚠️ [AI] Failed to get AI signal, using local signal");
      return 0;
   }
   
   string response = CharArrayToString(result);
   
   // Parse signal: {"signal":1,"confidence":95.5} or {"signal":-1} or {"signal":0}
   int sigPos = StringFind(response, "\"signal\":");
   if(sigPos >= 0)
   {
      int start = sigPos + 9;
      int end = StringFind(response, ",", start);
      if(end < 0) end = StringFind(response, "}", start);
      if(end > start)
      {
         string sigStr = StringSubstr(response, start, end - start);
         int aiSignal = (int)StringToInteger(sigStr);
         
         // Update confidence if available
         int confPos = StringFind(response, "\"confidence\":");
         if(confPos >= 0)
         {
            start = confPos + 13;
            end = StringFind(response, ",", start);
            if(end < 0) end = StringFind(response, "}", start);
            if(end > start)
            {
               AIConfidence = StringToDouble(StringSubstr(response, start, end - start));
            }
         }
         
         TotalPredictions++;
         LastPredictionTime = TimeCurrent();
         
         if(aiSignal == 1) LastSignal = "BUY";
         else if(aiSignal == -1) LastSignal = "SELL";
         else LastSignal = "HOLD";
         
         Print("🤖 [AI] Signal: ", LastSignal, " | Confidence: ", AIConfidence, "%");
         return aiSignal;
      }
   }
   
   return 0;
}

//+------------------------------------------------------------------+
//| Log Trade Result to Server (Karma Sync)                           |
//+------------------------------------------------------------------+
void LogTradeToServer(double profit, double profitPips, string tradeType, double lot, bool isClean, double durationMins, double openPrice=0, double closePrice=0)
{
   if(!EnableServerConnection || !ServerOnline)
      return;
   
   string url = ServerURL + "/api/trade";
   string headers = "Content-Type: application/json\r\n";
   
   // ═══════════════════════════════════════════════════════════════
   // ENHANCED KARMA DETECTION - Nhân Quả Nghiệp Báo
   // ═══════════════════════════════════════════════════════════════
   
   bool isWin = (profit > 0);
   bool isLoss = (profit < 0);
   
   // ═══════════════════════════════════════════════════════════════
   // 🔍 AUDIT MODULE - POST-TRADE UPDATE
   // Cập nhật Bộ phận Kiểm toán sau khi đóng lệnh
   // ═══════════════════════════════════════════════════════════════
   if(EnableAudit)
   {
      g_Audit.UpdateAfterTrade(isWin);
      Print("🔍 [AUDIT] Trade result recorded: ", isWin ? "WIN" : "LOSS");
   }
   
   // ─────────────────────────────────────────────────────────────
   // FOMO DETECTION (Tham - Greed)
   // Vào lệnh với confidence thấp HOẶC vào quá nhanh sau trade trước
   // ─────────────────────────────────────────────────────────────
   bool isFomo = false;
   string fomoReason = "";
   
   // Case 1: Confidence quá thấp (< 50%) - Tham, muốn vào dù signal yếu
   if(AIConfidence < 50 && AIConfidence > 0)
   {
      isFomo = true;
      fomoReason = "Low confidence (" + DoubleToString(AIConfidence, 1) + "%)";
   }
   
   // Case 2: Vào lệnh quá nhanh sau trade trước (< 5 phút cho H4 timeframe)
   if(LastTradeTime > 0)
   {
      int minutesSinceTrade = (int)((TimeCurrent() - LastTradeTime) / 60);
      if(minutesSinceTrade < 5)
      {
         isFomo = true;
         fomoReason = "Too fast (" + IntegerToString(minutesSinceTrade) + " min since last trade)";
      }
   }
   
   // ─────────────────────────────────────────────────────────────
   // REVENGE DETECTION (Sân - Anger)
   // Vào lệnh mới trong thời gian cooldown sau loss
   // ─────────────────────────────────────────────────────────────
   bool isRevenge = false;
   string revengeReason = "";
   
   // Case 1: Vào lệnh trong cooldown period sau loss
   if(LastLossTime > 0 && LastTradeWasLoss)
   {
      int minutesSinceLoss = (int)((TimeCurrent() - LastLossTime) / 60);
      if(minutesSinceLoss < CooldownAfterLoss)
      {
         isRevenge = true;
         revengeReason = "Within cooldown (" + IntegerToString(minutesSinceLoss) + "/" + IntegerToString(CooldownAfterLoss) + " min)";
      }
   }
   
   // Case 2: Đang có streak loss và vào lệnh ngay (muốn gỡ)
   if(ConsecutiveLosses >= 2)
   {
      isRevenge = true;
      revengeReason = "Consecutive losses: " + IntegerToString(ConsecutiveLosses);
   }
   
   // ─────────────────────────────────────────────────────────────
   // FOLLOWED AI DETECTION (Chánh Kiến - Right View)
   // ─────────────────────────────────────────────────────────────
   bool followedAI = false;
   
   // Followed AI nếu: trade cùng hướng với AI signal
   // HOẶC AI nói HOLD và không vào lệnh mới (trade này là close position)
   if(LastSignal == tradeType)
      followedAI = true;
   else if(LastSignal == "HOLD" && tradeType != "BUY" && tradeType != "SELL")
      followedAI = true;
   
   // ─────────────────────────────────────────────────────────────
   // LOG KARMA ANALYSIS
   // ─────────────────────────────────────────────────────────────
   Print("═══════════════════════════════════════════════════════════");
   Print("☸️ KARMA ANALYSIS - ", _Symbol);
   Print("───────────────────────────────────────────────────────────");
   Print("   Result: ", isWin ? "✅ WIN" : "❌ LOSS", " | ", DoubleToString(profit/_Point/10, 1), " pips");
   Print("   AI Signal: ", LastSignal, " | Confidence: ", DoubleToString(AIConfidence, 1), "%");
   Print("   Trade Type: ", tradeType, " | Followed AI: ", followedAI ? "YES ✅" : "NO ❌");
   
   if(isFomo)
      Print("   🔥 FOMO DETECTED: ", fomoReason, " (-3 karma)");
   if(isRevenge)
      Print("   😤 REVENGE DETECTED: ", revengeReason, " (-5 karma)");
   if(!followedAI && isLoss)
      Print("   🙈 IGNORED AI: Loss after ignoring signal (-2 karma)");
   
   Print("   Current Karma: ", Karma, " | Sila Streak: ", SilaStreak);
   Print("═══════════════════════════════════════════════════════════");
   
   // ─────────────────────────────────────────────────────────────
   // BUILD JSON PAYLOAD - WITH TRUNG ĐẠO FIELDS
   // ─────────────────────────────────────────────────────────────
   double atr = GetATR_Value();
   
   string body = StringFormat(
      "{\"symbol\":\"%s\",\"magic\":%d,\"type\":\"%s\",\"lot\":%.2f,"
      "\"open_price\":%.5f,\"close_price\":%.5f,"
      "\"profit_pips\":%.1f,\"profit_money\":%.2f,"
      "\"is_clean\":%s,\"is_fomo\":%s,\"is_revenge\":%s,\"followed_ai\":%s,"
      "\"ai_signal\":\"%s\",\"ai_confidence\":%.1f,"
      "\"karma_before\":%d,\"sila_streak\":%d,"
      "\"consecutive_losses\":%d,\"minutes_since_loss\":%d,"
      "\"rsi_m5\":%.1f,\"rsi_h1\":%.1f,\"rsi_h4\":%.1f,\"adx_h4\":%.1f,\"adx_m5\":%.1f,"
      "\"atr\":%.6f,\"drawdown_pips\":%.1f,\"duration_mins\":%.1f,"
      "\"had_sl\":%s,\"had_tp\":%s,\"trades_today\":%d}",
      _Symbol, MagicNumber, tradeType, lot,
      openPrice, closePrice,
      profitPips, profit,
      isClean ? "true" : "false",
      isFomo ? "true" : "false",
      isRevenge ? "true" : "false",
      followedAI ? "true" : "false",
      LastSignal, AIConfidence,
      Karma, SilaStreak,
      ConsecutiveLosses,
      (LastLossTime > 0) ? (int)((TimeCurrent() - LastLossTime) / 60) : 9999,
      GetRSI(h_RSI_M5), GetRSI(h_RSI_H1), GetRSI(h_RSI_H4), GetADX(h_ADX_H4), GetADX(h_ADX_M5),
      atr, TradeMaxDrawdownPips, durationMins,
      TradeHadSL ? "true" : "false",
      TradeHadTP ? "true" : "false",
      DailyTrades
   );
   
   char post[];
   char result[];
   string resultHeaders;
   
   StringToCharArray(body, post, 0, StringLen(body));
   ArrayResize(post, StringLen(body));
   
   int res = WebRequest("POST", url, headers, HeartbeatTimeout * 1000, post, result, resultHeaders);
   
   if(res == 200)
   {
      string response = CharArrayToString(result);
      
      // Parse karma from server response and SYNC
      int karmaPos = StringFind(response, "\"karma_after\":");
      if(karmaPos >= 0)
      {
         int start = karmaPos + 14;
         int end = StringFind(response, ",", start);
         if(end < 0) end = StringFind(response, "}", start);
         if(end > start)
         {
            int serverKarma = (int)StringToInteger(StringSubstr(response, start, end - start));
            Karma = serverKarma;  // Sync karma from server
            Print("📡 [SYNC] Karma synced from server: ", Karma);
         }
      }
      
      // Parse sila_streak from server
      int silaPos = StringFind(response, "\"sila_streak\":");
      if(silaPos >= 0)
      {
         int start = silaPos + 14;
         int end = StringFind(response, ",", start);
         if(end < 0) end = StringFind(response, "}", start);
         if(end > start)
         {
            int serverSila = (int)StringToInteger(StringSubstr(response, start, end - start));
            SilaStreak = serverSila;
         }
      }
      
      // Parse level from server
      int levelPos = StringFind(response, "\"level\":\"");
      if(levelPos >= 0)
      {
         int start = levelPos + 9;
         int end = StringFind(response, "\"", start);
         if(end > start)
         {
            string level = StringSubstr(response, start, end - start);
            Print("🕉️ [LEVEL] Current Level: ", level);
         }
      }
   }
   else
   {
      Print("⚠️ [LOG] Failed to log trade: HTTP ", res);
   }
   
   // ─────────────────────────────────────────────────────────────
   // UPDATE LOCAL TRACKING FOR NEXT TRADE
   // ─────────────────────────────────────────────────────────────
   LastTradeTime = TimeCurrent();
   LastTradeConfidence = AIConfidence;
   
   if(isLoss)
   {
      LastLossTime = TimeCurrent();
      LastTradeWasLoss = true;
   }
   else
   {
      LastTradeWasLoss = false;
      // Reset LastLossTime nếu win (không còn trong trạng thái "cần cooldown")
      if(ConsecutiveLosses == 0)
         LastLossTime = 0;
   }
}

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("🕉️ BODHI GENESIS V4.13 + AUDIT Initializing...");
   
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   if(!InitIndicators())
   {
      Print("❌ Failed to initialize indicators!");
      return INIT_FAILED;
   }
   
   ArrayResize(SpreadHistory, 100);
   ArrayInitialize(SpreadHistory, 0);
   
   ResetDaily();
   ResetKarma();
   LastTradeDay = 0;
   LastPredictionTime = TimeCurrent();
   
   // ═══════════════════════════════════════════════════════════════
   // AUDIT MODULE INITIALIZATION
   // ═══════════════════════════════════════════════════════════════
   if(EnableAudit)
   {
      g_Audit.Configure(
         Audit_MaxDD_Warning,
         Audit_MaxDD_Danger,
         Audit_MaxDD_Critical,
         Audit_MaxConsecutiveLosses,
         Audit_MinConfidence,
         Audit_MinSecBetweenTrades
      );
      g_Audit.SetDayStartEquity(AccountInfoDouble(ACCOUNT_EQUITY));
      Print("🔍 [AUDIT] Module initialized - Bộ phận Kiểm toán sẵn sàng!");
   }
   
   Print("✅ BODHI GENESIS V4.14 Ready!");
   Print("   Features: Audit, Dashboard, PriceAction, VirtualSL, PartialClose, TamBao");
   Print("   V4.14 Fix: Daily reset logic + THUAN THIEN mode (UseFallbackSignal=", UseFallbackSignal ? "ON" : "OFF", ")");
   
   return INIT_SUCCEEDED;
}

bool InitIndicators()
{
   h_RSI_M5 = iRSI(_Symbol, PERIOD_M5, RSI_Period, PRICE_CLOSE);
   h_RSI_M15 = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   h_RSI_H1 = iRSI(_Symbol, PERIOD_H1, RSI_Period, PRICE_CLOSE);
   h_RSI_H4 = iRSI(_Symbol, PERIOD_H4, RSI_Period, PRICE_CLOSE);
   
   h_ADX_M5 = iADX(_Symbol, PERIOD_M5, ADX_Period);
   h_ADX_H1 = iADX(_Symbol, PERIOD_H1, ADX_Period);
   h_ADX_H4 = iADX(_Symbol, PERIOD_H4, ADX_Period);
   
   h_ATR = iATR(_Symbol, PERIOD_H4, 14);
   
   // TEMA D1 50 = Main Trend (Tự Nhiên)
   h_EMA_D1 = iMA(_Symbol, PERIOD_D1, 50, 0, MODE_EMA, PRICE_CLOSE);
   
   return (h_RSI_M5 != INVALID_HANDLE && h_RSI_H4 != INVALID_HANDLE && h_ADX_H4 != INVALID_HANDLE && h_EMA_D1 != INVALID_HANDLE);
}

//+------------------------------------------------------------------+
//| Price Action Pattern Detection                                     |
//+------------------------------------------------------------------+
int DetectPriceAction()
{
   if(!EnablePriceAction) return 0;
   
   double open1 = iOpen(_Symbol, PERIOD_H4, 1);
   double high1 = iHigh(_Symbol, PERIOD_H4, 1);
   double low1 = iLow(_Symbol, PERIOD_H4, 1);
   double close1 = iClose(_Symbol, PERIOD_H4, 1);
   
   double open2 = iOpen(_Symbol, PERIOD_H4, 2);
   double high2 = iHigh(_Symbol, PERIOD_H4, 2);
   double low2 = iLow(_Symbol, PERIOD_H4, 2);
   double close2 = iClose(_Symbol, PERIOD_H4, 2);
   
   double body1 = MathAbs(close1 - open1);
   double range1 = high1 - low1;
   double upperWick1 = high1 - MathMax(open1, close1);
   double lowerWick1 = MathMin(open1, close1) - low1;
   
   if(range1 > 0)
   {
      // Bullish Pin Bar
      if(lowerWick1 > body1 * 2 && lowerWick1 > upperWick1 * 2)
         return 1;
      
      // Bearish Pin Bar
      if(upperWick1 > body1 * 2 && upperWick1 > lowerWick1 * 2)
         return -1;
   }
   
   // Inside Bar
   if(high1 < high2 && low1 > low2)
   {
      if(close1 > open1) return 1;
      if(close1 < open1) return -1;
   }
   
   // Engulfing
   if(close1 > open1 && close2 < open2 && close1 > open2 && open1 < close2)
      return 1;
   if(close1 < open1 && close2 > open2 && close1 < open2 && open1 > close2)
      return -1;
   
   return 0;
}

//+------------------------------------------------------------------+
//| Check Singularity                                                  |
//+------------------------------------------------------------------+
bool IsSingularity(int direction)
{
   if(!EnableSingularity) return false;
   
   double rsi_m15 = GetRSI(h_RSI_M15);
   double rsi_h1 = GetRSI(h_RSI_H1);
   double rsi_h4 = GetRSI(h_RSI_H4);
   double adx_h4 = GetADX(h_ADX_H4);
   int pa = DetectPriceAction();
   
   if(direction == 1)
   {
      if(rsi_m15 > 45 && rsi_h1 > 50 && rsi_h4 > 50 && adx_h4 > ADX_Strong_Trend)
      {
         Print("🌟 SINGULARITY BUY! M15:", rsi_m15, " H1:", rsi_h1, " H4:", rsi_h4, " ADX:", adx_h4, " PA:", pa);
         return true;
      }
   }
   else if(direction == -1)
   {
      if(rsi_m15 < 55 && rsi_h1 < 50 && rsi_h4 < 50 && adx_h4 > ADX_Strong_Trend)
      {
         Print("🌟 SINGULARITY SELL! M15:", rsi_m15, " H1:", rsi_h1, " H4:", rsi_h4, " ADX:", adx_h4, " PA:", pa);
         return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Get ATR-based SL/TP                                                |
//+------------------------------------------------------------------+
double GetATR_Value()
{
   RefreshIndicatorCache();
   return Cache_ATR_H4;
}

double GetDynamicSL()
{
   double atr = GetATR_Value();
   if(atr > 0)
      return atr * ATR_SL_Real;
   
   return StopLoss * _Point;  // Fallback
}

double GetDynamicTP()
{
   double atr = GetATR_Value();
   if(atr > 0)
      return atr * ATR_TP1_Real;
   
   return TakeProfit * _Point;  // Fallback
}

//+------------------------------------------------------------------+
//| DUAL SL/TP Management - TRẬN THẬT (Shadow)                         |
//| EA tự quản lý: Real SL, Partial Close TP1, Trail TP2               |
//+------------------------------------------------------------------+
void CheckDualSLTP()
{
   if(!UseDualSLTP) return;
   if(RealSL_Price == 0) return;
   if(!HasPosition()) 
   {
      // Reset all levels
      ResetDualLevels();
      return;
   }
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double currentPrice = (Position_Type == 1) ? bid : ask;
   
   // ═══════════════════════════════════════════════════════════════
   // CHECK REAL SL (Shadow SL)
   // ═══════════════════════════════════════════════════════════════
   bool hitSL = false;
   if(Position_Type == 1)  // BUY
      hitSL = (bid <= RealSL_Price);
   else if(Position_Type == -1)  // SELL
      hitSL = (ask >= RealSL_Price);
   
   if(hitSL)
   {
      double pips = MathAbs(Position_OpenPrice - currentPrice) / _Point;
      Print("═══════════════════════════════════════════════════════════");
      Print("🛡️ REAL SL HIT (Shadow) | Price: ", currentPrice, " | SL: ", RealSL_Price);
      Print("   Loss: ", DoubleToString(pips, 1), " pips");
      Print("═══════════════════════════════════════════════════════════");
      CloseAllPositions();
      ResetDualLevels();
      return;
   }
   
   // ═══════════════════════════════════════════════════════════════
   // CHECK TP1 - PARTIAL CLOSE (bỏ túi trước)
   // ═══════════════════════════════════════════════════════════════
   if(!PartialClosed && RealTP1_Price != 0)
   {
      bool hitTP1 = false;
      if(Position_Type == 1)  // BUY
         hitTP1 = (bid >= RealTP1_Price);
      else if(Position_Type == -1)  // SELL
         hitTP1 = (ask <= RealTP1_Price);
      
      if(hitTP1)
      {
         // Determine close ratio based on trend strength
         double closeRatio = IsTrendStrong() ? PartialCloseRatio_Strong : PartialCloseRatio;
         
         for(int i = PositionsTotal() - 1; i >= 0; i--)
         {
            if(posInfo.SelectByIndex(i))
            {
               if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;
               
               double vol = posInfo.Volume();
               double closeVol = NormalizeDouble(vol * closeRatio, 2);
               double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
               
               if(closeVol < minVol) closeVol = minVol;
               if(closeVol >= vol) closeVol = vol * 0.5;  // At least keep 50%
               
               if(trade.PositionClosePartial(posInfo.Ticket(), closeVol))
               {
                  double pips = MathAbs(currentPrice - Position_OpenPrice) / _Point;
                  
                  Print("═══════════════════════════════════════════════════════════");
                  Print("💰 TP1 HIT - PARTIAL CLOSE | ", DoubleToString(closeRatio*100, 0), "% BỎ TÚI!");
                  Print("   Profit: +", DoubleToString(pips, 1), " pips");
                  Print("   Closed: ", closeVol, " lots | Remaining: ", DoubleToString(vol - closeVol, 2), " lots");
                  
                  // Move SL to breakeven
                  RealSL_Price = Position_OpenPrice;
                  Print("   🔒 SL moved to BREAKEVEN: ", Position_OpenPrice);
                  
                  // Start trailing
                  if(TrailAfterPartial)
                  {
                     IsTrailing = true;
                     double atr = GetATR_Value();
                     if(Position_Type == 1)
                        TrailSL_Price = currentPrice - atr * TrailATR_Multiplier;
                     else
                        TrailSL_Price = currentPrice + atr * TrailATR_Multiplier;
                     Print("   📈 Trail SL started at: ", TrailSL_Price);
                  }
                  
                  Print("   Còn lại thả trôi follow trend → TP2: ", RealTP2_Price);
                  Print("═══════════════════════════════════════════════════════════");
                  
                  PartialClosed = true;
               }
            }
         }
      }
   }
   
   // ═══════════════════════════════════════════════════════════════
   // TRAILING SL (sau partial close)
   // ═══════════════════════════════════════════════════════════════
   if(IsTrailing && PartialClosed)
   {
      double atr = GetATR_Value();
      double trailDist = atr * TrailATR_Multiplier;
      
      if(Position_Type == 1)  // BUY - trail up
      {
         double newTrail = currentPrice - trailDist;
         if(newTrail > TrailSL_Price && newTrail > RealSL_Price)
         {
            TrailSL_Price = newTrail;
            RealSL_Price = newTrail;  // Update shadow SL
            Print("📈 Trail SL updated: ", DoubleToString(TrailSL_Price, _Digits));
         }
         
         // Check if hit trailing SL
         if(bid <= TrailSL_Price)
         {
            double pips = MathAbs(currentPrice - Position_OpenPrice) / _Point;
            Print("═══════════════════════════════════════════════════════════");
            Print("📈 TRAIL SL HIT - CLOSE REMAINING");
            Print("   Final profit: +", DoubleToString(pips, 1), " pips");
            Print("═══════════════════════════════════════════════════════════");
            CloseAllPositions();
            ResetDualLevels();
            return;
         }
      }
      else if(Position_Type == -1)  // SELL - trail down
      {
         double newTrail = currentPrice + trailDist;
         if(newTrail < TrailSL_Price && newTrail < RealSL_Price)
         {
            TrailSL_Price = newTrail;
            RealSL_Price = newTrail;
            Print("📈 Trail SL updated: ", DoubleToString(TrailSL_Price, _Digits));
         }
         
         if(ask >= TrailSL_Price)
         {
            double pips = MathAbs(Position_OpenPrice - currentPrice) / _Point;
            Print("═══════════════════════════════════════════════════════════");
            Print("📈 TRAIL SL HIT - CLOSE REMAINING");
            Print("   Final profit: +", DoubleToString(pips, 1), " pips");
            Print("═══════════════════════════════════════════════════════════");
            CloseAllPositions();
            ResetDualLevels();
            return;
         }
      }
   }
   
   // ═══════════════════════════════════════════════════════════════
   // CHECK TP2 - FULL CLOSE (phần còn lại)
   // ═══════════════════════════════════════════════════════════════
   if(PartialClosed && RealTP2_Price != 0)
   {
      bool hitTP2 = false;
      if(Position_Type == 1)
         hitTP2 = (bid >= RealTP2_Price);
      else if(Position_Type == -1)
         hitTP2 = (ask <= RealTP2_Price);
      
      if(hitTP2)
      {
         double pips = MathAbs(currentPrice - Position_OpenPrice) / _Point;
         Print("═══════════════════════════════════════════════════════════");
         Print("🎯 TP2 HIT - FULL CLOSE! TREND FOLLOWED TO THE END!");
         Print("   Total profit: +", DoubleToString(pips, 1), " pips");
         Print("═══════════════════════════════════════════════════════════");
         CloseAllPositions();
         ResetDualLevels();
      }
   }
}

void ResetDualLevels()
{
   RealSL_Price = 0;
   RealTP1_Price = 0;
   RealTP2_Price = 0;
   TrailSL_Price = 0;
   Position_Type = 0;
   Position_OpenPrice = 0;
   PartialClosed = false;
   IsTrailing = false;
}

// Legacy function
void CheckVirtualSL()
{
   CheckDualSLTP();
}

//+------------------------------------------------------------------+
//| Partial Close - Now handled by CheckDualSLTP()                     |
//+------------------------------------------------------------------+
void CheckPartialClose()
{
   // Partial close logic is now integrated into CheckDualSLTP()
   // This function kept for backward compatibility
   // Do nothing here - CheckDualSLTP handles everything
}

//+------------------------------------------------------------------+
//| Smart Exit                                                         |
//+------------------------------------------------------------------+
bool ShouldSmartExit()
{
   if(!UseSmartExit || !HasPosition()) return false;
   
   double rsi = GetRSI(h_RSI_M15);
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() != _Symbol || posInfo.Magic() != MagicNumber) continue;
         
         // ═══════════════════════════════════════════════════════════
         // CHECK MINIMUM HOLD TIME - Tránh đóng ngay sau khi vào lệnh
         // ═══════════════════════════════════════════════════════════
         datetime posOpenTime = (datetime)posInfo.Time();
         int holdMins = (int)((TimeCurrent() - posOpenTime) / 60);
         
         if(holdMins < SmartExit_MinHoldMins)
         {
            // Chưa đủ thời gian hold, không exit
            continue;
         }
         
         if(posInfo.PositionType() == POSITION_TYPE_BUY && rsi > SmartExit_RSI_Bull)
         {
            Print("🧠 Smart Exit BUY: RSI M15 = ", rsi, " | Hold: ", holdMins, " mins");
            return true;
         }
         if(posInfo.PositionType() == POSITION_TYPE_SELL && rsi < SmartExit_RSI_Bear)
         {
            Print("🧠 Smart Exit SELL: RSI M15 = ", rsi, " | Hold: ", holdMins, " mins");
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| FSS-5 Panic (Anti Flash Crash)                                     |
//| - Detects sudden equity drop                                        |
//| - Detects price gap/spike                                           |
//| - US30 has lower sensitivity to avoid false triggers               |
//+------------------------------------------------------------------+
static double FSS_LastEquity = 0;
static double FSS_LastPrice = 0;
static datetime FSS_LastCheck = 0;

bool CheckPanic()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Initialize on first run
   if(FSS_LastEquity == 0) 
   { 
      FSS_LastEquity = equity; 
      FSS_LastPrice = currentPrice;
      FSS_LastCheck = TimeCurrent();
      return false; 
   }
   
   // Calculate drops
   double equityDrop = (FSS_LastEquity - equity) / FSS_LastEquity * 100;
   double priceDrop = MathAbs(FSS_LastPrice - currentPrice) / FSS_LastPrice * 100;
   
   // Adjust threshold for indices (less sensitive)
   double threshold = Use_Indices_Logic ? PanicThreshold * FSS5_Sensitivity : PanicThreshold;
   
   // Flash crash detection: sudden price spike > 0.5% in one tick
   bool isFlashCrash = (priceDrop > 0.5);
   
   // Update last values
   FSS_LastEquity = equity;
   FSS_LastPrice = currentPrice;
   FSS_LastCheck = TimeCurrent();
   
   // Trigger panic if equity drops OR flash crash detected
   if(equityDrop >= threshold)
   {
      Print("🚨 FSS-5 PANIC! Equity drop: ", DoubleToString(equityDrop, 2), "% | Threshold: ", DoubleToString(threshold, 2), "%");
      CloseAllPositions();
      DailyStopped = true;
      Alert("FSS-5 TRIGGERED! Equity dropped ", DoubleToString(equityDrop, 2), "% - All positions closed!");
      return true;
   }
   
   if(isFlashCrash && HasPosition())
   {
      Print("🚨 FSS-5 FLASH CRASH! Price spike: ", DoubleToString(priceDrop, 2), "%");
      Print("   Last: ", FSS_LastPrice, " Current: ", currentPrice);
      CloseAllPositions();
      Alert("FSS-5 FLASH CRASH! Price spiked ", DoubleToString(priceDrop, 2), "% - All positions closed!");
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| NEWS FILTER - MT5 Calendar + Forex Factory                         |
//| Blocks trading before/after high impact news for XAU/US30          |
//+------------------------------------------------------------------+
bool IsNewsTime()
{
   if(!UseNewsFilter) return false;
   
   // Only filter for XAU and US30 (volatile during news)
   bool isVolatileSymbol = (StringFind(_Symbol, "XAU") >= 0 || 
                            StringFind(_Symbol, "US30") >= 0 || 
                            StringFind(_Symbol, "DJ") >= 0 ||
                            StringFind(_Symbol, "USA30") >= 0);
   if(!isVolatileSymbol) return false;
   
   datetime currentTime = TimeCurrent();
   
   // Check MT5 built-in Calendar every 5 minutes
   if(currentTime - LastNewsCheck > 300)
   {
      LastNewsCheck = currentTime;
      CheckUpcomingNews();
   }
   
   // If we have upcoming news, check if we're in the blackout window
   if(NextNewsTime > 0)
   {
      int minutesToNews = (int)((NextNewsTime - currentTime) / 60);
      int minutesAfterNews = (int)((currentTime - NextNewsTime) / 60);
      
      // Before news window
      if(minutesToNews > 0 && minutesToNews <= NewsMinutesBefore)
      {
         Print("📰 NEWS FILTER: ", minutesToNews, " minutes until ", NextNewsTitle);
         return true;
      }
      
      // After news window  
      if(minutesAfterNews >= 0 && minutesAfterNews <= NewsMinutesAfter)
      {
         Print("📰 NEWS FILTER: ", minutesAfterNews, " minutes after ", NextNewsTitle);
         return true;
      }
      
      // News passed, reset
      if(minutesAfterNews > NewsMinutesAfter)
      {
         NextNewsTime = 0;
         NextNewsTitle = "";
      }
   }
   
   return false;
}

void CheckUpcomingNews()
{
   // Use MT5 built-in Economic Calendar
   MqlCalendarValue values[];
   datetime from = TimeCurrent();
   datetime to = from + 24 * 60 * 60;  // Next 24 hours
   
   // Get calendar events
   int count = CalendarValueHistory(values, from, to);
   
   if(count <= 0) return;
   
   datetime nearestNews = 0;
   string nearestTitle = "";
   int nearestImpact = 0;
   
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event;
      if(!CalendarEventById(values[i].event_id, event)) continue;
      
      MqlCalendarCountry country;
      if(!CalendarCountryById(event.country_id, country)) continue;
      
      // Check if currency matches our filter
      string currency = country.currency;
      if(StringFind(NewsCurrencies, currency) < 0) continue;
      
      // Check impact level
      // CALENDAR_IMPORTANCE_NONE=0, LOW=1, MODERATE=2, HIGH=3
      bool isHighImpact = (event.importance == CALENDAR_IMPORTANCE_HIGH);
      bool isMediumImpact = (event.importance == CALENDAR_IMPORTANCE_MODERATE);
      
      if(FilterHighImpact && isHighImpact)
      {
         if(nearestNews == 0 || values[i].time < nearestNews)
         {
            nearestNews = values[i].time;
            nearestTitle = event.name + " (" + currency + ")";
            nearestImpact = 3;
         }
      }
      
      if(FilterMediumImpact && isMediumImpact)
      {
         if(nearestNews == 0 || values[i].time < nearestNews)
         {
            nearestNews = values[i].time;
            nearestTitle = event.name + " (" + currency + ")";
            nearestImpact = 2;
         }
      }
   }
   
   // Update globals
   if(nearestNews > 0)
   {
      NextNewsTime = nearestNews;
      NextNewsTitle = nearestTitle;
      NextNewsImpact = nearestImpact;
      
      int minutesToNews = (int)((NextNewsTime - TimeCurrent()) / 60);
      Print("📰 Next News: ", NextNewsTitle, " in ", minutesToNews, " minutes");
   }
}

string GetNewsStatus()
{
   if(!UseNewsFilter) return "OFF";
   
   if(NextNewsTime == 0) return "CLEAR";
   
   datetime currentTime = TimeCurrent();
   int minutesToNews = (int)((NextNewsTime - currentTime) / 60);
   
   if(minutesToNews > 0 && minutesToNews <= NewsMinutesBefore)
      return "BLOCKED (-" + IntegerToString(minutesToNews) + "m)";
   
   int minutesAfterNews = (int)((currentTime - NextNewsTime) / 60);
   if(minutesAfterNews >= 0 && minutesAfterNews <= NewsMinutesAfter)
      return "BLOCKED (+" + IntegerToString(minutesAfterNews) + "m)";
   
   if(minutesToNews > NewsMinutesBefore)
      return "OK (" + IntegerToString(minutesToNews) + "m)";
   
   return "CLEAR";
}

//+------------------------------------------------------------------+
//| Spread Management - OPTIMIZED                                      |
//+------------------------------------------------------------------+
void UpdateSpread()
{
   CurrentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point * 10;
   
   // Running average (EMA-like) - NO LOOP!
   if(AverageSpread == 0)
      AverageSpread = CurrentSpread;
   else
      AverageSpread = (AverageSpread * 99 + CurrentSpread) / 100;
}

bool IsSpreadOK()
{
   if(!UseSpreadProtection) return true;
   if(AverageSpread == 0) return true;
   
   if(CurrentSpread > AverageSpread * SpreadMultiplier)
   {
      Print("⚠️ Spread too high: ", CurrentSpread, " > ", AverageSpread * SpreadMultiplier);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Calculate TEMA D1 50 - Main Trend Direction (Tự Nhiên)           |
//| TEMA = 3*EMA1 - 3*EMA2 + EMA3                                    |
//+------------------------------------------------------------------+
double GetTEMA_D1()
{
   datetime barTime = iTime(_Symbol, PERIOD_D1, 0);
   if(barTime == 0) return 0;
   
   if(barTime == LastTemaBarTime && CachedTemaD1 != 0)
      return CachedTemaD1;
   
   // Get enough D1 close prices to calculate triple EMA
   double close[];
   ArraySetAsSeries(close, true);
   int copied = CopyClose(_Symbol, PERIOD_D1, 0, 200, close);
   if(copied < 150) return 0;  // Need enough history
   
   int period = 50;
   double alpha = 2.0 / (period + 1);
   
   // Initialize EMAs with first close
   double ema1 = close[copied - 1];
   double ema2 = ema1;
   double ema3 = ema1;
   
   // Calculate EMAs from oldest to newest
   for(int i = copied - 2; i >= 0; i--)
   {
      ema1 = alpha * close[i] + (1 - alpha) * ema1;
      ema2 = alpha * ema1 + (1 - alpha) * ema2;
      ema3 = alpha * ema2 + (1 - alpha) * ema3;
   }
   
   // TEMA = 3*EMA1 - 3*EMA2 + EMA3
   CachedTemaD1 = 3 * ema1 - 3 * ema2 + ema3;
   LastTemaBarTime = barTime;
   return CachedTemaD1;
}

//+------------------------------------------------------------------+
//| Get Main Trend from TEMA D1 50                                   |
//| Returns: 1 = Bullish (Price > TEMA), -1 = Bearish, 0 = Neutral   |
//+------------------------------------------------------------------+
int GetMainTrend_D1()
{
   double tema = GetTEMA_D1();
   if(tema == 0) return 0;  // Error
   
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double distance = (currentPrice - tema) / tema * 100;  // % distance
   
   // Require at least 0.1% distance for clear trend
   if(distance > 0.1)
      return 1;   // Bullish - Price above TEMA
   else if(distance < -0.1)
      return -1;  // Bearish - Price below TEMA
   else
      return 0;   // Neutral - Too close to TEMA
}

//+------------------------------------------------------------------+
//| Get Signal - TỨ THỜI HỢP NHẤT + TEMA D1                          |
//+------------------------------------------------------------------+
int GetSignal()
{
   // ═══════════════════════════════════════════════════════════════
   // ANTI-REVENGE CHECK - Cooldown after loss
   // ═══════════════════════════════════════════════════════════════
   if(LastLossTime > 0 && CooldownAfterLoss > 0)
   {
      int minutesSinceLoss = (int)((TimeCurrent() - LastLossTime) / 60);
      if(minutesSinceLoss < CooldownAfterLoss)
      {
         Print("⏳ [COOLDOWN] ", CooldownAfterLoss - minutesSinceLoss, " mins remaining - avoiding revenge trading");
         LastSignal = "COOLDOWN";
         return 0;
      }
   }
   
   // ═══════════════════════════════════════════════════════════════
   // TRY AI SIGNAL FIRST (if server connected)
   // ═══════════════════════════════════════════════════════════════
   if(UseAIServer && ServerOnline && AIConnected)
   {
      int aiSignal = GetAISignal();
      
      // ═══════════════════════════════════════════════════════════
      // CONFIDENCE FILTER - Only trade when AI is confident
      // ═══════════════════════════════════════════════════════════
      if(aiSignal != 0)
      {
         if(AIConfidence < MinAIConfidence)
         {
            Print("⚠️ [FILTER] AI Signal ", (aiSignal == 1 ? "BUY" : "SELL"), 
                  " rejected - Confidence ", DoubleToString(AIConfidence, 1), 
                  "% < ", DoubleToString(MinAIConfidence, 1), "%");
            LastSignal = "FILTERED";
            return 0;  // Reject low confidence signals
         }
         
         // AI gave confident signal, use it!
         Print("✅ [AI] Signal ", (aiSignal == 1 ? "BUY" : "SELL"), 
               " ACCEPTED - Confidence ", DoubleToString(AIConfidence, 1), "%");
         return aiSignal;
      }
      
      // ═══════════════════════════════════════════════════════════
      // V4.14: AI said HOLD - Check if fallback allowed
      // THUẬN THIÊN: UseFallbackSignal = false → Only trade with AI
      // ═══════════════════════════════════════════════════════════
      if(!UseFallbackSignal)
      {
         Print("🕉️ [THUAN THIEN] AI = HOLD | Fallback disabled - Waiting for AI signal");
         LastSignal = "HOLD";
         return 0;  // No fallback, respect AI decision
      }
      
      Print("⚠️ [FALLBACK] AI = HOLD | Using local signal (UseFallbackSignal=true)");
   }
   
   // ═══════════════════════════════════════════════════════════════
   // FALLBACK: LOCAL SIGNAL (RSI/ADX) + TEMA D1 FILTER
   // Only reached if: Server offline OR UseFallbackSignal=true
   // ═══════════════════════════════════════════════════════════════
   
   // V4.14: If server should be used but is offline, and fallback disabled
   if(UseAIServer && (!ServerOnline || !AIConnected) && !UseFallbackSignal)
   {
      Print("🕉️ [THUAN THIEN] Server OFFLINE | Fallback disabled - No trade until reconnect");
      LastSignal = "OFFLINE";
      return 0;
   }
   
   double rsi_m5 = GetRSI(h_RSI_M5);
   double rsi_m15 = GetRSI(h_RSI_M15);
   double rsi_h1 = GetRSI(h_RSI_H1);
   double rsi_h4 = GetRSI(h_RSI_H4);
   double adx_m5 = GetADX(h_ADX_M5);
   
   // ═══════════════════════════════════════════════════════════════
   // TEMA D1 50 - Main Trend (Tự Nhiên / Đạo)
   // Mưa thuận gió hòa: Chỉ trade thuận main trend
   // ═══════════════════════════════════════════════════════════════
   int mainTrend = GetMainTrend_D1();
   double tema_d1 = GetTEMA_D1();
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   int pa = DetectPriceAction();
   
   // Update AI stats (for local signals)
   TotalPredictions++;
   LastPredictionTime = TimeCurrent();
   
   // ═══════════════════════════════════════════════════════════════
   // TỨ THỜI HỢP NHẤT + TEMA D1 FILTER
   // D1 TEMA (Tự Nhiên) → H4 (Đạo) → H1 (Trời) → M15 (Đất) → M5 (Người)
   // ═══════════════════════════════════════════════════════════════
   
   // BUY: Main trend bullish/neutral + H4 không bearish, H1 bull, M15 confirm, M5 pullback
   bool buy = (mainTrend >= 0 && rsi_h4 >= 45 && rsi_h1 > 50 && rsi_m15 > 45 && rsi_m5 < 45 && adx_m5 > 20);
   
   // SELL: Main trend bearish/neutral + H4 không bullish, H1 bear, M15 confirm, M5 pullback
   bool sell = (mainTrend <= 0 && rsi_h4 <= 55 && rsi_h1 < 50 && rsi_m15 < 55 && rsi_m5 > 55 && adx_m5 > 20);
   
   // Combine with PA
   if(buy && (pa == 1 || pa == 0))
   {
      LastSignal = "BUY";
      Print("🕉️ [LOCAL] TỨ THỜI + TEMA | TREND:", (mainTrend==1?"BULL":"NEUT"), " TEMA:", DoubleToString(tema_d1, _Digits), 
            " | H4:", rsi_h4, " H1:", rsi_h1, " M15:", rsi_m15, " M5:", rsi_m5);
      return 1;
   }
   
   if(sell && (pa == -1 || pa == 0))
   {
      LastSignal = "SELL";
      Print("🕉️ [LOCAL] TỨ THỜI + TEMA | TREND:", (mainTrend==-1?"BEAR":"NEUT"), " TEMA:", DoubleToString(tema_d1, _Digits),
            " | H4:", rsi_h4, " H1:", rsi_h1, " M15:", rsi_m15, " M5:", rsi_m5);
      return -1;
   }
   
   // Log rejection if signal blocked by main trend
   if((rsi_h4 >= 45 && rsi_h1 > 50 && rsi_m15 > 45 && rsi_m5 < 45 && adx_m5 > 20) && mainTrend == -1)
   {
      Print("⚠️ [TEMA FILTER] BUY blocked - Main trend BEARISH | Price:", currentPrice, " < TEMA:", DoubleToString(tema_d1, _Digits));
   }
   if((rsi_h4 <= 55 && rsi_h1 < 50 && rsi_m15 < 55 && rsi_m5 > 55 && adx_m5 > 20) && mainTrend == 1)
   {
      Print("⚠️ [TEMA FILTER] SELL blocked - Main trend BULLISH | Price:", currentPrice, " > TEMA:", DoubleToString(tema_d1, _Digits));
   }
   
   LastSignal = "HOLD";
   return 0;
}

//+------------------------------------------------------------------+
//| Calculate Lot (FIXED - uses ATR-based SL for proper sizing)        |
//+------------------------------------------------------------------+
double CalculateLot()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Base risk 0.15% - Karma multiplier sẽ scale lot sau
   double risk = Use_Indices_Logic ? US30_RiskPercent : RiskPercent;
   double riskAmt = equity * risk / 100;
   
   double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // Get REAL SL distance (ATR-based) for lot calculation
   double atr = GetATR_Value();
   double slDistance = 0;
   
   if(atr > 0)
   {
      slDistance = atr * ATR_SL_Real;  // Use REAL SL (shadow), not fake
   }
   else
   {
      slDistance = StopLoss * point;  // Fallback to fixed SL
   }
   
   // Convert SL distance to points for lot calculation
   double slPoints = slDistance / point;
   
   // Calculate lot based on risk
   double rawLot = 0;
   if(tickVal > 0 && slPoints > 0)
   {
      // Correct formula: Lot = Risk$ / (SL_points × TickValue_per_point)
      double tickValuePerPoint = tickVal / (tickSize / point);
      rawLot = riskAmt / (slPoints * tickValuePerPoint);
   }
   
   // Debug log - Final Multiplier = Karma × Trend
   Print("📊 LOT CALC | Base Risk: ", DoubleToString(risk, 2), "% | Equity: $", DoubleToString(equity, 0), 
         " | Risk$: $", DoubleToString(riskAmt, 2),
         " | SL: ", DoubleToString(slPoints, 1), " pts",
         " | Base Lot: ", DoubleToString(rawLot, 3),
         " | Karma x", DoubleToString(GetKarmaMultiplier(), 2),
         " | Trend(", CurrentTrendStrength, ") x", DoubleToString(GetTrendMultiplier(), 2),
         " | Final x", DoubleToString(GetFinalMultiplier(), 2));
   
   // US30 special handling
   if(Use_Indices_Logic)
   {
      double baseLot = equity / US30_Equity_Per_Lot;
      
      // Use smaller of risk-based or equity-based
      if(rawLot > baseLot)
      {
         Print("📊 US30: Capping lot from ", DoubleToString(rawLot, 3), " to ", DoubleToString(baseLot, 3));
         rawLot = baseLot;
      }
      
      rawLot = MathMin(rawLot, Max_Lot_Indices);
      
      if(Auto_Adjust_To_MinLot && rawLot < minLot)
      {
         Print("📊 US30: Adjusting to MinLot ", minLot, " (was ", DoubleToString(rawLot, 3), ")");
         rawLot = minLot;
      }
   }
   
   // XAUUSD special handling - Dynamic lot theo Equity + Final multiplier (Karma × Trend)
   if(StringFind(_Symbol, "XAU") >= 0 || StringFind(_Symbol, "GOLD") >= 0)
   {
      // Dynamic lot theo equity tiers
      // $10,000 = 0.01 lot
      // $20,000 = 0.02 lot
      // $30,000-$50,000 = 0.03 lot
      // $50,001-$100,000 = 0.05 lot
      // >$100,000 = 0.05 lot (cap)
      
      double goldBaseLot = 0.01;  // Default
      
      if(equity >= 100000)
         goldBaseLot = 0.05;
      else if(equity >= 50001)
         goldBaseLot = 0.05;
      else if(equity >= 30000)
         goldBaseLot = 0.03;
      else if(equity >= 20000)
         goldBaseLot = 0.02;
      else
         goldBaseLot = 0.01;
      
      // Apply Final multiplier = Karma × Trend (capped)
      double karmaMult = GetKarmaMultiplier();
      double trendMult = GetTrendMultiplier();
      double finalMult = GetFinalMultiplier();
      double goldLot = goldBaseLot * finalMult;
      
      Print("🥇 XAUUSD: Equity=$", DoubleToString(equity, 0), 
            " | Base=", DoubleToString(goldBaseLot, 2),
            " | Karma=", GetKarmaLevel(), " (x", DoubleToString(karmaMult, 2), ")",
            " | Trend=", CurrentTrendStrength, " (x", DoubleToString(trendMult, 2), ")",
            " | Final=x", DoubleToString(finalMult, 2),
            " | Lot=", DoubleToString(goldLot, 2));
      
      rawLot = goldLot;
      
      // Apply hard cap
      if(rawLot > XAUUSD_Max_Lot)
      {
         Print("🥇 XAUUSD: Hard cap to ", DoubleToString(XAUUSD_Max_Lot, 2));
         rawLot = XAUUSD_Max_Lot;
      }
      
      // Minimum lot
      if(rawLot < 0.01)
         rawLot = 0.01;
   }
   
   // Forex: Also auto-adjust to minLot if too small
   // NHƯNG không áp dụng cho XAUUSD nếu minLot > calculated lot (vì XAUUSD cần risk-based)
   bool isGold = (StringFind(_Symbol, "XAU") >= 0 || StringFind(_Symbol, "GOLD") >= 0);
   if(!Use_Indices_Logic && rawLot < minLot && !isGold)
   {
      Print("📊 FOREX: Adjusting to MinLot ", minLot, " (was ", DoubleToString(rawLot, 3), ")");
      rawLot = minLot;
   }
   
   // Normalize
   rawLot = MathFloor(rawLot / lotStep) * lotStep;
   
   // Chỉ force minLot nếu không phải XAUUSD, hoặc nếu minLot <= 0.01
   if(!isGold || minLot <= 0.01)
      rawLot = MathMax(rawLot, minLot);
   else if(isGold && rawLot < minLot)
   {
      // XAUUSD: Nếu broker minLot > calculated lot → KHÔNG TRADE (quá rủi ro)
      Print("⚠️ XAUUSD: Lot ", DoubleToString(rawLot, 3), " < Broker MinLot ", DoubleToString(minLot, 2), " - TOO RISKY!");
      return 0;  // Không trade
   }
   
   rawLot = MathMin(rawLot, maxLot);
   
   Print("📊 FINAL LOT: ", DoubleToString(rawLot, 2));
   
   return NormalizeDouble(rawLot, 2);
}

//+------------------------------------------------------------------+
//| Check if trend is strong (for partial close ratio)                 |
//+------------------------------------------------------------------+
bool IsTrendStrong()
{
   double rsi_h4 = GetRSI(h_RSI_H4);
   double adx_h4 = GetADX(h_ADX_H4);
   
   // Strong trend: RSI clearly directional + ADX > 25
   bool strongBull = (rsi_h4 > 60 && adx_h4 > 25);
   bool strongBear = (rsi_h4 < 40 && adx_h4 > 25);
   
   return strongBull || strongBear;
}

//+------------------------------------------------------------------+
//| Execute Trade with DUAL SL/TP System                               |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal)
{
   // ═══════════════════════════════════════════════════════════════
   // UPDATE TREND ADAPTIVE PARAMS
   // ═══════════════════════════════════════════════════════════════
   UpdateTrendParams();
   
   // Check max trades for current trend strength
   if(DailyTrades >= CurrentMaxTrades)
   {
      Print("⚠️ [TREND LIMIT] Max trades reached for ", CurrentTrendStrength, " trend: ", CurrentMaxTrades);
      return;
   }
   
   double lot = CalculateLot();
   
   // ═══════════════════════════════════════════════════════════════
   // APPLY FINAL MULTIPLIER (Karma × Trend) - All trades
   // ═══════════════════════════════════════════════════════════════
   bool isGold = (StringFind(_Symbol, "XAU") >= 0 || StringFind(_Symbol, "GOLD") >= 0);
   if(!isGold)  // XAUUSD already has multiplier in CalculateLot
   {
      double finalMult = GetFinalMultiplier();
      lot = NormalizeDouble(lot * finalMult, 2);
      Print("📊 [ADAPTIVE] Lot × ", DoubleToString(finalMult, 2), 
            " (Karma:", GetKarmaLevel(), " × Trend:", CurrentTrendStrength, ") = ", DoubleToString(lot, 2));
   }
   
   // Singularity bonus (extra multiplier on top)
   if(IsSingularity(signal))
   {
      lot = NormalizeDouble(lot * 1.2, 2);  // Extra 20% for Singularity
      Print("🌟 SINGULARITY BONUS! +20% → Lot:", lot);
   }
   
   // Cap
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(Use_Indices_Logic) maxLot = MathMin(maxLot, Max_Lot_Indices);
   lot = MathMin(lot, maxLot);
   
   // Min lot check
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lot < minLot)
   {
      Print("⚠️ Lot ", DoubleToString(lot, 3), " < MinLot ", DoubleToString(minLot, 2), " - Using MinLot");
      lot = minLot;
   }
   
   // ═══════════════════════════════════════════════════════════════
   // ADAPTIVE SL/TP BASED ON TREND STRENGTH
   // ═══════════════════════════════════════════════════════════════
   double sl_mult, tp1_mult, tp2_mult;
   GetTrendAdaptiveSLTP(sl_mult, tp1_mult, tp2_mult);
   
   // Get ATR for SL/TP calculation
   double atr = GetATR_Value();
   if(atr == 0) atr = StopLoss * _Point;  // Fallback
   
   // ═══════════════════════════════════════════════════════════════
   // 🔍 AUDIT MODULE - PRE-TRADE CHECK
   // Bộ phận Kiểm toán kiểm tra TRƯỚC khi vào lệnh
   // ═══════════════════════════════════════════════════════════════
   if(EnableAudit)
   {
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      double slPips = atr * sl_mult / _Point;
      double tpPips = atr * tp1_mult / _Point;
      double riskAmount = lot * slPips * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      int mainTrend = GetMainTrend_D1();
      double adx_h4 = GetADX(h_ADX_H4);
      double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      // Run comprehensive pre-trade audit
      bool auditPassed = g_Audit.RunPreTradeAudit(
         lot,                   // Lot size
         equity,                // Current equity
         riskAmount,            // Risk amount $
         slPips,                // SL in pips
         tpPips,                // TP in pips
         currentPrice,          // Entry price
         signal,                // Direction (1=BUY, -1=SELL)
         AIConfidence,          // AI confidence
         mainTrend,             // Main trend direction
         CurrentTrendStrength,  // Trend strength
         adx_h4                 // ADX H4 value
      );
      
      // Future: Block trade if critical issues and AuditBlockCritical is enabled
      if(!auditPassed && AuditBlockCritical)
      {
         Print("🚨 [AUDIT] Trade BLOCKED by Audit Module - Critical issues detected!");
         return;
      }
      
      // Log warning even if not blocking
      if(g_Audit.HasCriticalIssues())
      {
         Print("🚨 [AUDIT] WARNING: Critical issues detected but trade NOT blocked (AuditBlockCritical=false)");
      }
   }
   
   double price;
   double fakeSL, fakeTP;  // TRẬN GIẢ - cho sàn thấy
   double realSL, realTP1, realTP2;  // TRẬN THẬT - EA quản lý
   
   if(signal == 1)  // BUY
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      // TRẬN THẬT (Shadow - EA quản lý) - USE ADAPTIVE SL/TP
      realSL = price - atr * sl_mult;              // SL thật (adaptive)
      realTP1 = price + atr * tp1_mult;            // TP1 - partial close (adaptive)
      realTP2 = price + atr * tp2_mult;            // TP2 - full close (adaptive)
      
      // TRẬN GIẢ (Fake - broker thấy, xa để không bị hunt)
      fakeSL = price - atr * ATR_SL_Fake;           // SL giả (xa)
      fakeTP = price + atr * ATR_TP_Fake;           // TP giả (xa)
      
      // Gửi lệnh với FAKE SL/TP (nếu dùng Dual system)
      double orderSL = UseDualSLTP ? fakeSL : realSL;
      double orderTP = UseDualSLTP ? fakeTP : realTP2;
      
      if(trade.Buy(lot, _Symbol, price, orderSL, orderTP, TradeComment))
      {
         Print("═══════════════════════════════════════════════════════════");
         Print("✅ BUY EXECUTED | Lot: ", lot, " | Karma: ", GetKarmaLevel(), " | Trend: ", CurrentTrendStrength);
         Print("───────────────────────────────────────────────────────────");
         Print("📈 ADAPTIVE PARAMS (", CurrentTrendStrength, " trend):");
         Print("   Lot Mult: x", DoubleToString(GetFinalMultiplier(), 2), " | Max Trades: ", CurrentMaxTrades);
         Print("   SL: ", DoubleToString(sl_mult, 1), "x ATR | TP1: ", DoubleToString(tp1_mult, 1), "x | TP2: ", DoubleToString(tp2_mult, 1), "x");
         Print("───────────────────────────────────────────────────────────");
         Print("🎭 DUAL SL/TP SYSTEM:");
         Print("   TRẬN GIẢ (Broker thấy):");
         Print("      Fake SL: ", DoubleToString(fakeSL, _Digits), " (", DoubleToString(atr * ATR_SL_Fake / _Point, 0), " pips)");
         Print("      Fake TP: ", DoubleToString(fakeTP, _Digits), " (", DoubleToString(atr * ATR_TP_Fake / _Point, 0), " pips)");
         Print("   TRẬN THẬT (EA quản lý - ADAPTIVE):");
         Print("      Real SL: ", DoubleToString(realSL, _Digits), " (", DoubleToString(atr * sl_mult / _Point, 0), " pips)");
         Print("      TP1:     ", DoubleToString(realTP1, _Digits), " (", DoubleToString(atr * tp1_mult / _Point, 0), " pips) → Chốt ", DoubleToString(PartialCloseRatio*100,0), "%");
         Print("      TP2:     ", DoubleToString(realTP2, _Digits), " (", DoubleToString(atr * tp2_mult / _Point, 0), " pips) → Trail còn lại");
         Print("═══════════════════════════════════════════════════════════");
         
         // Lưu các mức giá THẬT để EA quản lý
         DailyTrades++;
         PartialClosed = false;
         IsTrailing = false;
         Position_Type = 1;  // BUY
         Position_OpenPrice = price;
         RealSL_Price = realSL;
         RealTP1_Price = realTP1;
         RealTP2_Price = realTP2;
         TrailSL_Price = 0;
         
         // ═══════════════════════════════════════════════════════════
         // TRUNG ĐẠO TRACKING - Set ngay khi mở lệnh
         // ═══════════════════════════════════════════════════════════
         TradeEntryTime = TimeCurrent();
         TradeEntryPrice = price;
         TradeMaxDrawdownPips = 0;
         TradeHadSL = true;   // Luôn có SL (Real hoặc Fake)
         TradeHadTP = true;   // Luôn có TP (Real hoặc Fake)
         Print("📊 Trung Đạo: Entry tracked at ", price);
      }
   }
   else if(signal == -1)  // SELL
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      // TRẬN THẬT (Shadow - EA quản lý) - USE ADAPTIVE SL/TP
      realSL = price + atr * sl_mult;              // SL thật (adaptive)
      realTP1 = price - atr * tp1_mult;            // TP1 - partial close (adaptive)
      realTP2 = price - atr * tp2_mult;            // TP2 - full close (adaptive)
      
      // TRẬN GIẢ (Fake - broker thấy)
      fakeSL = price + atr * ATR_SL_Fake;
      fakeTP = price - atr * ATR_TP_Fake;
      
      double orderSL = UseDualSLTP ? fakeSL : realSL;
      double orderTP = UseDualSLTP ? fakeTP : realTP2;
      
      if(trade.Sell(lot, _Symbol, price, orderSL, orderTP, TradeComment))
      {
         Print("═══════════════════════════════════════════════════════════");
         Print("✅ SELL EXECUTED | Lot: ", lot, " | Karma: ", GetKarmaLevel(), " | Trend: ", CurrentTrendStrength);
         Print("───────────────────────────────────────────────────────────");
         Print("📈 ADAPTIVE PARAMS (", CurrentTrendStrength, " trend):");
         Print("   Lot Mult: x", DoubleToString(GetFinalMultiplier(), 2), " | Max Trades: ", CurrentMaxTrades);
         Print("   SL: ", DoubleToString(sl_mult, 1), "x ATR | TP1: ", DoubleToString(tp1_mult, 1), "x | TP2: ", DoubleToString(tp2_mult, 1), "x");
         Print("───────────────────────────────────────────────────────────");
         Print("🎭 DUAL SL/TP SYSTEM:");
         Print("   TRẬN GIẢ (Broker thấy):");
         Print("      Fake SL: ", DoubleToString(fakeSL, _Digits), " (", DoubleToString(atr * ATR_SL_Fake / _Point, 0), " pips)");
         Print("      Fake TP: ", DoubleToString(fakeTP, _Digits), " (", DoubleToString(atr * ATR_TP_Fake / _Point, 0), " pips)");
         Print("   TRẬN THẬT (EA quản lý - ADAPTIVE):");
         Print("      Real SL: ", DoubleToString(realSL, _Digits), " (", DoubleToString(atr * sl_mult / _Point, 0), " pips)");
         Print("      TP1:     ", DoubleToString(realTP1, _Digits), " (", DoubleToString(atr * tp1_mult / _Point, 0), " pips) → Chốt ", DoubleToString(PartialCloseRatio*100,0), "%");
         Print("      TP2:     ", DoubleToString(realTP2, _Digits), " (", DoubleToString(atr * tp2_mult / _Point, 0), " pips) → Trail còn lại");
         Print("═══════════════════════════════════════════════════════════");
         
         DailyTrades++;
         PartialClosed = false;
         IsTrailing = false;
         Position_Type = -1;  // SELL
         Position_OpenPrice = price;
         RealSL_Price = realSL;
         RealTP1_Price = realTP1;
         RealTP2_Price = realTP2;
         TrailSL_Price = 0;
         
         // ═══════════════════════════════════════════════════════════
         // TRUNG ĐẠO TRACKING - Set ngay khi mở lệnh
         // ═══════════════════════════════════════════════════════════
         TradeEntryTime = TimeCurrent();
         TradeEntryPrice = price;
         TradeMaxDrawdownPips = 0;
         TradeHadSL = true;   // Luôn có SL (Real hoặc Fake)
         TradeHadTP = true;   // Luôn có TP (Real hoặc Fake)
         Print("📊 Trung Đạo: Entry tracked at ", price);
      }
   }
}

//+------------------------------------------------------------------+
//| Helper Functions                                                   |
//+------------------------------------------------------------------+
double ReadIndicatorValue(int handle, double fallback)
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(handle != INVALID_HANDLE && CopyBuffer(handle, 0, 0, 1, buf) > 0)
      return buf[0];
   return fallback;
}

void RefreshIndicatorCache()
{
   datetime now = TimeCurrent();
   if(now == LastIndicatorCacheTime) return;
   LastIndicatorCacheTime = now;
   
   Cache_RSI_M5 = ReadIndicatorValue(h_RSI_M5, Cache_RSI_M5);
   Cache_RSI_M15 = ReadIndicatorValue(h_RSI_M15, Cache_RSI_M15);
   Cache_RSI_H1 = ReadIndicatorValue(h_RSI_H1, Cache_RSI_H1);
   Cache_RSI_H4 = ReadIndicatorValue(h_RSI_H4, Cache_RSI_H4);
   
   Cache_ADX_M5 = ReadIndicatorValue(h_ADX_M5, Cache_ADX_M5);
   Cache_ADX_H1 = ReadIndicatorValue(h_ADX_H1, Cache_ADX_H1);
   Cache_ADX_H4 = ReadIndicatorValue(h_ADX_H4, Cache_ADX_H4);
   
   Cache_ATR_H4 = ReadIndicatorValue(h_ATR, Cache_ATR_H4);
}

double GetRSI(int handle, int shift = 0)
{
   if(shift != 0)
   {
      double buf[];
      ArraySetAsSeries(buf, true);
      if(CopyBuffer(handle, 0, shift, 1, buf) > 0) return buf[0];
      return 50;
   }
   
   RefreshIndicatorCache();
   if(handle == h_RSI_M5) return Cache_RSI_M5;
   if(handle == h_RSI_M15) return Cache_RSI_M15;
   if(handle == h_RSI_H1) return Cache_RSI_H1;
   if(handle == h_RSI_H4) return Cache_RSI_H4;
   return 50;
}

double GetADX(int handle, int shift = 0)
{
   if(shift != 0)
   {
      double buf[];
      ArraySetAsSeries(buf, true);
      if(CopyBuffer(handle, 0, shift, 1, buf) > 0) return buf[0];
      return 0;
   }
   
   RefreshIndicatorCache();
   if(handle == h_ADX_M5) return Cache_ADX_M5;
   if(handle == h_ADX_H1) return Cache_ADX_H1;
   if(handle == h_ADX_H4) return Cache_ADX_H4;
   return 0;
}

bool HasPosition()
{
   // Cache for 1 second to avoid multiple loops per tick
   if(TimeCurrent() - LastPositionCheck < 1 && LastPositionCheck > 0)
      return CachedHasPosition;
   
   LastPositionCheck = TimeCurrent();
   CachedHasPosition = false;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
         {
            CachedHasPosition = true;
            return true;
         }
   }
   return false;
}

void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
            trade.PositionClose(posInfo.Ticket());
   }
}

bool IsInSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   int startHour, endHour;
   string symbol = _Symbol;
   
   // ═══════════════════════════════════════════════════════════════
   // 12 PAIRS SESSION FILTER - Symbol-specific optimized hours
   // ═══════════════════════════════════════════════════════════════
   
   // EUR Cluster
   if(StringFind(symbol, "EURUSD") >= 0)
   {
      startHour = EURUSD_StartHour;
      endHour = EURUSD_EndHour;
   }
   else if(StringFind(symbol, "EURGBP") >= 0)
   {
      startHour = EURGBP_StartHour;
      endHour = EURGBP_EndHour;
   }
   else if(StringFind(symbol, "EURJPY") >= 0)
   {
      startHour = EURJPY_StartHour;
      endHour = EURJPY_EndHour;
   }
   // GBP Cluster
   else if(StringFind(symbol, "GBPUSD") >= 0)
   {
      startHour = GBPUSD_StartHour;
      endHour = GBPUSD_EndHour;
   }
   else if(StringFind(symbol, "GBPJPY") >= 0)
   {
      startHour = GBPJPY_StartHour;
      endHour = GBPJPY_EndHour;
   }
   // USD Majors
   else if(StringFind(symbol, "USDJPY") >= 0)
   {
      startHour = USDJPY_StartHour;
      endHour = USDJPY_EndHour;
   }
   else if(StringFind(symbol, "USDCAD") >= 0)
   {
      startHour = USDCAD_StartHour;
      endHour = USDCAD_EndHour;
   }
   else if(StringFind(symbol, "AUDUSD") >= 0)
   {
      startHour = AUDUSD_StartHour;
      endHour = AUDUSD_EndHour;
   }
   // Commodities
   else if(StringFind(symbol, "XAU") >= 0 || StringFind(symbol, "GOLD") >= 0)
   {
      startHour = XAUUSD_StartHour;
      endHour = XAUUSD_EndHour;
   }
   else if(StringFind(symbol, "XAG") >= 0 || StringFind(symbol, "SILVER") >= 0)
   {
      startHour = XAGUSD_StartHour;
      endHour = XAGUSD_EndHour;
   }
   // Indices
   else if(StringFind(symbol, "US30") >= 0 || StringFind(symbol, "DJ") >= 0 || StringFind(symbol, "DOW") >= 0)
   {
      startHour = US30_StartHour;
      endHour = US30_EndHour;
   }
   // Oceania
   else if(StringFind(symbol, "NZDUSD") >= 0)
   {
      startHour = NZDUSD_StartHour;
      endHour = NZDUSD_EndHour;
   }
   else
   {
      // Default fallback (London + NY)
      startHour = 7;
      endHour = 20;
      Print("⚠️ [SESSION] Unknown symbol: ", symbol, " - Using default 7-20h");
   }
   
   return (dt.hour >= startHour && dt.hour <= endHour);
}

void ResetDaily()
{
   DayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   DayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   DailyTrades = 0;
   DailyTradesCompleted = 0;  // V4.14 FIX: Reset completed trades counter
   DailyWins = 0;
   DailyLosses = 0;
   DailyPnL = 0;
   DailyDrawdown = 0;
   DailyStopped = false;
   ConsecutiveLosses = 0;
   UpdateBonusStatus();
   
   Print("📅 [DAILY RESET] New day started | Equity: $", DoubleToString(DayStartEquity, 2));
}

void ResetKarma()
{
   Karma = 0;
   SilaStreak = 0;
   TotalMerit = 0;
   HasBonus = false;
   Chua = Kinh = Tang = 0;
   PhatBalance = PhapBalance = TangBalance = 0;
}

void UpdateBonusStatus()
{
   HasBonus = EnableKarmaBonus && (SilaStreak >= SilaStreakForBonus || TotalMerit >= MeritForBonus);
}

int GetMaxConsecutiveLoss()
{
   return HasBonus ? MaxConsecutiveLosses + 1 : MaxConsecutiveLosses;
}

bool CanTrade()
{
   if(!EnableTrading || Killed || DailyStopped) return false;
   if(DailyTrades >= MaxTradesPerDay) return false;
   if(ConsecutiveLosses >= GetMaxConsecutiveLoss()) return false;
   
   // News filter (for XAU and US30)
   if(IsNewsTime()) return false;
   
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dailyLoss = (DayStartEquity - equity) / DayStartEquity * 100;
   double totalLoss = (DayStartBalance - equity) / DayStartBalance * 100;
   
   if(dailyLoss >= MaxDailyLossPercent) { DailyStopped = true; return false; }
   if(totalLoss >= MaxTotalLossPercent) { Killed = true; return false; }
   
   return true;
}

//+------------------------------------------------------------------+
//| OnTick                                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // New day
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day != LastTradeDay)
   {
      ResetDaily();
      LastTradeDay = dt.day;
      
      // Update Audit day start equity
      if(EnableAudit)
         g_Audit.SetDayStartEquity(AccountInfoDouble(ACCOUNT_EQUITY));
   }
   
   // Update spread
   UpdateSpread();
   
   // ═══════════════════════════════════════════════════════════════
   // TRUNG ĐẠO: Track max drawdown during trade
   // ═══════════════════════════════════════════════════════════════
   if(HasPosition())
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(posInfo.SelectByIndex(i))
         {
            if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
            {
               double currentProfit = posInfo.Profit();
               if(currentProfit < 0)
               {
                  // Convert to pips
                  double ddPips = MathAbs(posInfo.PriceCurrent() - posInfo.PriceOpen()) / _Point;
                  if(_Digits == 3 || _Digits == 5) ddPips /= 10;  // Adjust for 5-digit brokers
                  
                  if(ddPips > TradeMaxDrawdownPips)
                     TradeMaxDrawdownPips = ddPips;
               }
               
               // ═══════════════════════════════════════════════════════════════
               // 🔍 AUDIT MODULE - IN-TRADE MONITORING (every 60 seconds)
               // ═══════════════════════════════════════════════════════════════
               static datetime lastInTradeAudit = 0;
               if(EnableAudit && TimeCurrent() - lastInTradeAudit >= 60)
               {
                  double equity = AccountInfoDouble(ACCOUNT_EQUITY);
                  AuditResult floatResult = g_Audit.AuditFloatingPL(currentProfit, equity);
                  
                  if(floatResult.severity >= AUDIT_WARNING)
                  {
                     g_Audit.ClearResults();
                     g_Audit.AddResult(floatResult);
                  }
                  
                  lastInTradeAudit = TimeCurrent();
               }
            }
         }
      }
   }
   
   // ═══════════════════════════════════════════════════════════════
   // SERVER HEARTBEAT - Check connection to AI server
   // ═══════════════════════════════════════════════════════════════
   CheckHeartbeat();
   
   // Update dashboard
   UpdateDashboard();
   
   // Panic check
   if(CheckPanic()) return;
   
   // Virtual SL check
   CheckVirtualSL();
   
   // Smart exit
   if(ShouldSmartExit())
   {
      CloseAllPositions();
      return;
   }
   
   // ═══════════════════════════════════════════════════════════════
   // MAX HOLD TIME - Force close after X hours (avoid overnight)
   // ═══════════════════════════════════════════════════════════════
   if(MaxHoldHours > 0 && HasPosition())
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         if(posInfo.SelectByIndex(i))
         {
            if(posInfo.Symbol() == _Symbol && posInfo.Magic() == MagicNumber)
            {
               datetime posOpenTime = (datetime)posInfo.Time();
               int holdHours = (int)((TimeCurrent() - posOpenTime) / 3600);
               
               if(holdHours >= MaxHoldHours)
               {
                  Print("⏰ [MAX HOLD] Closing position after ", holdHours, " hours (max: ", MaxHoldHours, "h)");
                  Print("   P/L: $", DoubleToString(posInfo.Profit(), 2));
                  trade.PositionClose(posInfo.Ticket());
               }
            }
         }
      }
   }
   
   // Partial close
   CheckPartialClose();
   
   // Can trade?
   if(!CanTrade()) return;
   
   // New bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;
   
   // Session
   if(UseSessionFilter && !IsInSession()) return;
   
   // Spread
   if(!IsSpreadOK()) return;
   
   // Position
   if(HasPosition()) return;
   
   // Signal
   int signal = GetSignal();
   if(signal != 0)
   {
      // ═══════════════════════════════════════════════════════════
      // PRE-ENTRY RSI FILTER - Không vào lệnh khi RSI đã quá mức
      // Tránh bị Smart Exit đóng ngay sau khi vào
      // ═══════════════════════════════════════════════════════════
      if(UseSmartExit)
      {
         double rsi = GetRSI(h_RSI_M15);
         if(signal == 1 && rsi > SmartExit_RSI_Bull - 10)  // BUY khi RSI > 60 → skip
         {
            Print("⚠️ [PRE-ENTRY] Skip BUY - RSI M15 = ", DoubleToString(rsi, 1), " already near overbought");
            return;
         }
         if(signal == -1 && rsi < SmartExit_RSI_Bear + 10)  // SELL khi RSI < 40 → skip
         {
            Print("⚠️ [PRE-ENTRY] Skip SELL - RSI M15 = ", DoubleToString(rsi, 1), " already near oversold");
            return;
         }
      }
      
      // ═══════════════════════════════════════════════════════════
      // H4 TREND CONFIRMATION - Trade theo xu hướng lớn
      // ═══════════════════════════════════════════════════════════
      double rsi_h4 = GetRSI(h_RSI_H4);
      if(signal == 1 && rsi_h4 < 45)  // BUY nhưng H4 bearish bias
      {
         Print("⚠️ [TREND] Skip BUY - RSI H4 = ", DoubleToString(rsi_h4, 1), " < 45 (bearish bias)");
         return;
      }
      if(signal == -1 && rsi_h4 > 55)  // SELL nhưng H4 bullish bias
      {
         Print("⚠️ [TREND] Skip SELL - RSI H4 = ", DoubleToString(rsi_h4, 1), " > 55 (bullish bias)");
         return;
      }
      
      ExecuteTrade(signal);
   }
}

//+------------------------------------------------------------------+
//| OnTradeTransaction                                                 |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong deal = trans.deal;
      if(HistoryDealSelect(deal))
      {
         if(HistoryDealGetInteger(deal, DEAL_MAGIC) == MagicNumber)
         {
            ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal, DEAL_ENTRY);
            
            // ═══════════════════════════════════════════════════════════
            // TRADE CLOSED - Send to server with Trung Đạo data
            // (Entry tracking đã move sang ExecuteTrade để tránh timing issue)
            // ═══════════════════════════════════════════════════════════
            if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_OUT_BY)
            {
               double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
               ProcessResult(profit, deal);
            }
         }
      }
   }
}

void ProcessResult(double profit, ulong dealTicket = 0)
{
   DailyPnL += profit;
   DailyTradesCompleted++;  // V4.14 FIX: Count completed trades for dashboard
   
   string tradeType = (profit > 0) ? "WIN" : "LOSS";
   bool isClean = true;  // Can be improved with more detection
   
   // ═══════════════════════════════════════════════════════════════
   // TRUNG ĐẠO: Calculate duration
   // ═══════════════════════════════════════════════════════════════
   double durationMins = 0;
   if(TradeEntryTime > 0)
   {
      datetime closeTime = TimeCurrent();
      if(dealTicket > 0 && HistoryDealSelect(dealTicket))
         closeTime = (datetime)HistoryDealGetInteger(dealTicket, DEAL_TIME);
      durationMins = (double)(closeTime - TradeEntryTime) / 60.0;
   }
   
   // Calculate profit in pips (từ profit_money và pip value)
   double profitPips = 0;
   double lot = 0.01;  // Default
   
   // Lấy lot THỰC TẾ từ deal
   if(dealTicket > 0 && HistoryDealSelect(dealTicket))
   {
      lot = HistoryDealGetDouble(dealTicket, DEAL_VOLUME);
      if(lot <= 0) lot = 0.01;  // Fallback
      
      // Lấy price để tính pips chính xác hơn
      double dealPrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
      long dealType = HistoryDealGetInteger(dealTicket, DEAL_TYPE);
      
      // Nếu có entry price, tính pips từ price difference
      if(TradeEntryPrice > 0 && dealPrice > 0)
      {
         double priceDiff = 0;
         if(dealType == DEAL_TYPE_SELL)  // Close Buy
            priceDiff = dealPrice - TradeEntryPrice;
         else if(dealType == DEAL_TYPE_BUY)  // Close Sell
            priceDiff = TradeEntryPrice - dealPrice;
         
         // Convert to pips
         double pipSize = (_Digits == 3 || _Digits == 5) ? _Point * 10 : _Point;
         if(StringFind(_Symbol, "XAU") >= 0 || StringFind(_Symbol, "GOLD") >= 0)
            pipSize = 0.1;  // Gold: 1 pip = $0.1
         else if(StringFind(_Symbol, "US30") >= 0 || StringFind(_Symbol, "DJ") >= 0)
            pipSize = 1.0;  // US30: 1 pip = 1 point
            
         profitPips = priceDiff / pipSize;
      }
   }
   
   // Fallback: tính từ profit money nếu không có price data
   if(profitPips == 0 && profit != 0)
   {
      // Lấy pip value cho symbol
      double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double pointValue = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
      // Tính pip value per 1.0 lot
      double pipValue = 0;
      if(tickSize > 0 && pointValue > 0)
      {
         // Số points trong 1 pip
         double pointsPerPip = (_Digits == 3 || _Digits == 5) ? 10.0 : 1.0;
         pipValue = tickValue * pointsPerPip * (pointValue / tickSize);
      }
   
      // Fallback pip values nếu không tính được
      if(pipValue <= 0)
      {
         if(StringFind(_Symbol, "XAU") >= 0)
            pipValue = 10.0;   // Gold: ~$10 per pip (0.1) per 1.0 lot
         else if(StringFind(_Symbol, "US30") >= 0 || StringFind(_Symbol, "DJ") >= 0)
         {
            // US30: pip value phụ thuộc contract size của broker
            // Typical: $1 per point per 1.0 lot, nhưng có thể là $0.1 hoặc $10
            // Tính từ tick value nếu có
            double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
            double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
            if(tv > 0 && ts > 0)
               pipValue = tv / ts;  // $ per point per 1.0 lot
            else
               pipValue = 1.0;      // Default fallback
         }
         else if(StringFind(_Symbol, "JPY") >= 0)
            pipValue = 7.0;    // JPY pairs: ~$7 per pip per 1.0 lot
         else
            pipValue = 10.0;   // Standard forex ~$10 per pip per 1.0 lot
      }
   
      // Tính pips từ profit, pip value VÀ lot size
      if(pipValue > 0 && lot > 0)
         profitPips = profit / (pipValue * lot);
   }
   
   if(profit > 0)
   {
      DailyWins++;
      ConsecutiveLosses = 0;
      SilaStreak++;
      Karma += 2;
      TotalMerit += 2;
      
      // Reset cooldown on win
      LastLossTime = 0;
      
      // Tam Bao - 10% cúng dường
      double donation = profit * 0.1;
      PhatBalance += donation * 0.4;
      PhapBalance += donation * 0.35;
      TangBalance += donation * 0.25;
      
      if(PhatBalance >= 100) { PhatBalance -= 100; Chua++; }
      if(PhapBalance >= 50) { PhapBalance -= 50; Kinh++; }
      if(TangBalance >= 25) { TangBalance -= 25; Tang++; }
      
      Print("✅ WIN! Karma +2 | Consecutive losses reset | Cooldown cleared");
   }
   else
   {
      DailyLosses++;
      ConsecutiveLosses++;
      SilaStreak = 0;
      Karma -= 1;  // Bad karma
      
      // Set cooldown timer to prevent revenge trading
      LastLossTime = TimeCurrent();
      Print("❌ LOSS! Karma -1 | Cooldown started: ", CooldownAfterLoss, " minutes");
      
      if(ConsecutiveLosses >= GetMaxConsecutiveLoss())
      {
         DailyStopped = true;
         if(HasBonus) HasBonus = false;
         Print("🛑 MAX CONSECUTIVE LOSSES! Trading stopped for today.");
      }
   }
   
   UpdateBonusStatus();
   
   // ═══════════ LOG TO SERVER (KARMA SYNC + TRUNG ĐẠO) ═══════════
   // Lấy close price từ deal nếu có
   double closePrice = 0;
   if(dealTicket > 0 && HistoryDealSelect(dealTicket))
      closePrice = HistoryDealGetDouble(dealTicket, DEAL_PRICE);
   
   LogTradeToServer(profit, profitPips, tradeType, lot, isClean, durationMins, TradeEntryPrice, closePrice);
   
   // Reset Dual SL/TP levels
   ResetDualLevels();
   
   // Reset Trung Đạo tracking
   TradeEntryTime = 0;
   TradeEntryPrice = 0;
   TradeMaxDrawdownPips = 0;
   TradeHadSL = false;
   TradeHadTP = false;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "BD_");
   
   IndicatorRelease(h_RSI_M5);
   IndicatorRelease(h_RSI_M15);
   IndicatorRelease(h_RSI_H1);
   IndicatorRelease(h_RSI_H4);
   IndicatorRelease(h_ADX_M5);
   IndicatorRelease(h_ADX_H1);
   IndicatorRelease(h_ADX_H4);
   IndicatorRelease(h_ATR);
   
   Print("🕉️ BODHI V4.14 TRUNG ĐẠO Stopped");
   Print("   Karma:", Karma, " Level:", GetKarmaLevel());
   Print("   🙏 Tam Bảo: 🛕", Chua, " 📜", Kinh, " 🙏", Tang);
}
//+------------------------------------------------------------------+
