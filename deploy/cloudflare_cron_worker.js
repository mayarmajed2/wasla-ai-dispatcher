/**
 * Wasla AI — Zero-Cost Cloudflare Worker Cron Trigger
 *
 * Official Free Tier Limits (verified):
 * - No credit card required
 * - 100,000 requests/day, up to 3 Cron Triggers per Worker
 * - Invokes POST /api/scheduler/cron-tick every 10 minutes so due & missed
 *   60-minute review requests and 90-minute arrival checks execute reliably
 *   even if a free Render backend sleeps after 15 minutes of inactivity.
 *
 * Configure Worker secrets (never hardcode):
 *   wrangler secret put WASLA_BACKEND_URL
 *   wrangler secret put SCHEDULER_CRON_SECRET
 */

export default {
  async scheduled(event, env, ctx) {
    const baseUrl = (env.WASLA_BACKEND_URL || "").replace(/\/+$/, "");
    if (!baseUrl || !env.SCHEDULER_CRON_SECRET) {
      console.error("Missing WASLA_BACKEND_URL or SCHEDULER_CRON_SECRET");
      return;
    }
    const resp = await fetch(`${baseUrl}/api/scheduler/cron-tick`, {
      method: "POST",
      headers: {
        "X-Scheduler-Secret": env.SCHEDULER_CRON_SECRET,
        "Content-Type": "application/json",
      },
    });
    const text = await resp.text();
    console.log(`Wasla cron-tick HTTP ${resp.status}: ${text.slice(0, 300)}`);
  },
};
