// Lightweight audit module fallback for compilation.
#pragma once

enum AuditSeverity
{
   AUDIT_SEVERITY_OK = 0,
   AUDIT_SEVERITY_WARNING = 1,
   AUDIT_SEVERITY_DANGER = 2,
   AUDIT_SEVERITY_CRITICAL = 3
};

struct AuditResult
{
   AuditSeverity severity;
   string message;
};

class AuditModule
{
private:
   double max_dd_warning;
   double max_dd_danger;
   double max_dd_critical;
   int max_consecutive_losses;
   double min_confidence;
   int min_seconds_between_trades;
   double day_start_equity;
   int consecutive_losses;
   datetime last_trade_time;
   string last_critical_message;
   bool has_warning;
   bool has_danger;
   bool has_critical;

   void ResetFlags()
   {
      has_warning = false;
      has_danger = false;
      has_critical = false;
      last_critical_message = "";
   }

   void AddIssue(AuditSeverity severity, string message)
   {
      AuditResult result;
      result.severity = severity;
      result.message = message;
      UpdateFlags(result);
   }

   void UpdateFlags(const AuditResult &result)
   {
      if(result.severity == AUDIT_SEVERITY_CRITICAL)
      {
         has_critical = true;
         if(result.message != "")
            last_critical_message = result.message;
      }
      else if(result.severity == AUDIT_SEVERITY_DANGER)
      {
         has_danger = true;
      }
      else if(result.severity == AUDIT_SEVERITY_WARNING)
      {
         has_warning = true;
      }
   }

public:
   AuditModule()
   {
      max_dd_warning = 0.0;
      max_dd_danger = 0.0;
      max_dd_critical = 0.0;
      max_consecutive_losses = 0;
      min_confidence = 0.0;
      min_seconds_between_trades = 0;
      day_start_equity = 0.0;
      consecutive_losses = 0;
      last_trade_time = 0;
      ResetFlags();
   }

   void Configure(double dd_warning,
                  double dd_danger,
                  double dd_critical,
                  int max_losses,
                  double min_conf,
                  int min_seconds)
   {
      max_dd_warning = dd_warning;
      max_dd_danger = dd_danger;
      max_dd_critical = dd_critical;
      max_consecutive_losses = max_losses;
      min_confidence = min_conf;
      min_seconds_between_trades = min_seconds;
   }

   void SetDayStartEquity(double equity)
   {
      day_start_equity = equity;
   }

   void UpdateAfterTrade(bool is_win)
   {
      if(is_win)
         consecutive_losses = 0;
      else
         consecutive_losses++;
      last_trade_time = TimeCurrent();
   }

   bool RunPreTradeAudit(double lot,
                         double equity,
                         double risk_amount,
                         double sl_pips,
                         double tp_pips,
                         double entry_price,
                         int direction,
                         double ai_confidence,
                         int main_trend,
                         string trend_strength,
                         double adx_h4)
   {
      ResetFlags();

      if(lot <= 0.0)
         AddIssue(AUDIT_SEVERITY_WARNING, "Lot size is zero");
      if(sl_pips <= 0.0 || tp_pips <= 0.0)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid SL/TP pips");
      if(entry_price <= 0.0)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid entry price");
      if(direction != 1 && direction != -1)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid trade direction");
      if(main_trend != 1 && main_trend != 0 && main_trend != -1)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid main trend");
      if(trend_strength == "")
         AddIssue(AUDIT_SEVERITY_WARNING, "Missing trend strength");
      if(adx_h4 < 0.0)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid ADX value");
      if(max_consecutive_losses > 0 && consecutive_losses >= max_consecutive_losses)
         AddIssue(AUDIT_SEVERITY_DANGER, "Consecutive loss limit reached");
      if(min_seconds_between_trades > 0 && last_trade_time > 0 &&
         (TimeCurrent() - last_trade_time) < min_seconds_between_trades)
      {
         AddIssue(AUDIT_SEVERITY_WARNING, "Trades too frequent");
      }

      double dd_pct = 0.0;
      if(day_start_equity > 0.0 && equity < day_start_equity)
         dd_pct = (day_start_equity - equity) / day_start_equity * 100.0;

      if(max_dd_warning > 0.0 && dd_pct >= max_dd_warning)
         AddIssue(AUDIT_SEVERITY_WARNING, "Daily drawdown warning");
      if(max_dd_danger > 0.0 && dd_pct >= max_dd_danger)
         AddIssue(AUDIT_SEVERITY_DANGER, "Daily drawdown danger");
      if(max_dd_critical > 0.0 && dd_pct >= max_dd_critical)
         AddIssue(AUDIT_SEVERITY_CRITICAL, "Daily drawdown critical");

      if(min_confidence > 0.0 && ai_confidence > 0.0 && ai_confidence < min_confidence)
         AddIssue(AUDIT_SEVERITY_WARNING, "AI confidence below minimum");

      if(risk_amount < 0.0)
         AddIssue(AUDIT_SEVERITY_WARNING, "Invalid risk amount");

      return !has_critical;
   }

   AuditResult AuditFloatingPL(double current_profit, double equity)
   {
      AuditResult result;
      result.severity = AUDIT_SEVERITY_OK;
      result.message = "";

      if(current_profit < 0.0 && equity > 0.0)
      {
         result.severity = AUDIT_SEVERITY_WARNING;
         result.message = "Floating P/L is negative";
      }

      return result;
   }

   void ClearResults()
   {
      ResetFlags();
   }

   void AddResult(const AuditResult &result)
   {
      UpdateFlags(result);
   }

   bool HasWarnings() { return has_warning; }
   bool HasDangerIssues() { return has_danger; }
   bool HasCriticalIssues() { return has_critical; }

   string GetLastCriticalMessage() { return last_critical_message; }

   string GetAuditSummary()
   {
      if(has_critical) return "CRITICAL";
      if(has_danger) return "DANGER";
      if(has_warning) return "WARNING";
      return "OK";
   }
};

AuditModule g_Audit;
