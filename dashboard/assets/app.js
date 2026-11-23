let authToken = localStorage.getItem("wasla_token") || "";
let currentUser = null;
let allOrders = [];
let allMenuItems = [];
let activeStatusFilter = "ALL";
let directiveOrderId = null;
let audioContext = null;

const loginScreen = document.getElementById("loginScreen");
const appShell = document.getElementById("appShell");
const ordersContainer = document.getElementById("orders");
const modal = document.getElementById("directiveModal");
const directiveText = document.getElementById("directiveText");
const directiveType = document.getElementById("directiveType");
const directivePreview = document.getElementById("directivePreview");
const toastStack = document.getElementById("toastStack");

function showToast(msg) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = msg;
  toastStack.appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function escapeHtml(val) {
  const el = document.createElement("span");
  el.textContent = val ?? "";
  return el.innerHTML;
}

async function apiFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    logout();
    throw new Error("Unauthorized");
  }
  return res;
}

function logout() {
  authToken = "";
  currentUser = null;
  localStorage.removeItem("wasla_token");
  appShell.hidden = true;
  loginScreen.hidden = false;
}

document.getElementById("logoutBtn").addEventListener("click", logout);

document.getElementById("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errBox = document.getElementById("loginError");
  errBox.hidden = true;
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: document.getElementById("loginUser").value.trim(),
      password: document.getElementById("loginPass").value,
    }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    errBox.textContent = data.detail || "فشل تسجيل الدخول";
    errBox.hidden = false;
    return;
  }
  const data = await res.json();
  authToken = data.access_token;
  currentUser = data.user;
  localStorage.setItem("wasla_token", authToken);
  enterApp();
});

async function bootstrapAuth() {
  if (!authToken) {
    loginScreen.hidden = false;
    appShell.hidden = true;
    return;
  }
  try {
    const res = await apiFetch("/api/auth/me");
    if (!res.ok) throw new Error();
    currentUser = await res.json();
    enterApp();
  } catch {
    logout();
  }
}

function enterApp() {
  loginScreen.hidden = true;
  appShell.hidden = false;
  document.getElementById("currentUserLabel").textContent =
    `${currentUser.full_name} (${currentUser.role}) • بيانات تشغيلية حقيقية فقط`;
  document.getElementById("staffNavBtn").hidden = currentUser.role !== "admin";
  loadOrders();
  loadMetrics();
  connectWS();
}

// Language RTL / LTR Toggle
document.getElementById("langToggle").addEventListener("click", () => {
  const html = document.documentElement;
  const isRtl = html.dir === "rtl";
  html.dir = isRtl ? "ltr" : "rtl";
  html.lang = isRtl ? "en" : "ar";
});

// Sound
function playChime() {
  if (!audioContext) return;
  const now = audioContext.currentTime;
  [0, 0.13].forEach((offset, idx) => {
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = "sine";
    osc.frequency.value = idx ? 880 : 660;
    gain.gain.setValueAtTime(0.0001, now + offset);
    gain.gain.exponentialRampToValueAtTime(0.22, now + offset + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.22);
    osc.connect(gain);
    gain.connect(audioContext.destination);
    osc.start(now + offset);
    osc.stop(now + offset + 0.24);
  });
}

document.getElementById("enableSound").addEventListener("click", async () => {
  audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
  await audioContext.resume();
  playChime();
  document.getElementById("enableSound").textContent = "🔊 الصوت مفعّل";
});

// Navigation
const viewTitles = {
  ordersView: "الطلبات الحية",
  conversationsView: "محادثات العملاء",
  cxView: "تجربة العملاء والتقييمات والشكاوى",
  menuView: "إدارة المنيو والتوافر (MOKA)",
  analyticsView: "تحليلات المبيعات الفعلية",
  healthView: "حالة الذكاء الاصطناعي والتكاملات",
  staffView: "الموظفين والصلاحيات وسجل التدقيق",
  settingsView: "الإعدادات",
};

document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.view;
    document.getElementById(target).classList.add("active");
    document.getElementById("viewTitle").textContent = viewTitles[target] || "";
    if (target === "menuView") loadMenu();
    if (target === "conversationsView") loadConversations();
    if (target === "cxView") loadCustomerExperience();
    if (target === "analyticsView") loadMetrics();
    if (target === "healthView") loadHealth();
    if (target === "staffView") loadStaffAndLogs();
  });
});

function statusLabel(st) {
  return {
    SUBMITTED: "طلب جديد",
    IN_KITCHEN: "في المطبخ",
    DISPATCHED: "خرج للتوصيل",
    UNREACHABLE_ALERT: "لا يمكن الوصول",
    FULFILLED: "تم التسليم",
    CANCELLED: "ملغي",
  }[st] || st;
}

async function loadMetrics() {
  const res = await apiFetch("/api/metrics");
  if (!res.ok) return;
  const m = await res.json();
  document.getElementById("kpiNew").textContent = m.by_status.SUBMITTED || 0;
  document.getElementById("kpiKitchen").textContent = m.by_status.IN_KITCHEN || 0;
  document.getElementById("kpiDispatched").textContent = m.by_status.DISPATCHED || 0;
  document.getElementById("kpiFulfilled").textContent = m.by_status.FULFILLED || 0;
  document.getElementById("kpiRevenue").textContent = `${m.total_revenue_egp} ج.م`;
  document.getElementById("kpiHandoffRating").textContent = `${m.human_handoffs} / ${m.average_rating ?? "—"}`;
  document.getElementById("navNewBadge").textContent = m.by_status.SUBMITTED || 0;
  document.getElementById("navOutBadge").textContent = m.menu_out_of_stock || 0;

  document.getElementById("analyticsGrid").innerHTML = `
    <article class="kpi-card"><span class="kpi-label">إجمالي الطلبات الفعلية</span><strong class="kpi-value">${m.total_orders}</strong></article>
    <article class="kpi-card accent-orange"><span class="kpi-label">إجمالي الإيرادات الفعلية</span><strong class="kpi-value">${m.total_revenue_egp} ج.م</strong></article>
    <article class="kpi-card accent-green"><span class="kpi-label">الطلبات المكتملة</span><strong class="kpi-value">${m.by_status.FULFILLED || 0}</strong></article>
    <article class="kpi-card"><span class="kpi-label">الطلبات الملغاة</span><strong class="kpi-value">${m.by_status.CANCELLED || 0}</strong></article>
    <article class="kpi-card"><span class="kpi-label">أصناف المنيو المتوفرة</span><strong class="kpi-value">${m.menu_available} / ${m.menu_total}</strong></article>
    <article class="kpi-card"><span class="kpi-label">المحادثات النشطة</span><strong class="kpi-value">${m.active_conversations}</strong></article>
  `;
}

async function loadOrders() {
  const res = await apiFetch("/api/orders");
  if (!res.ok) return;
  allOrders = await res.json();
  renderOrders();
}

function actionButtons(order) {
  const actions = {
    SUBMITTED: [["إرسال للمطبخ", "IN_KITCHEN", "btn-blue"], ["إلغاء", "CANCELLED", "btn-outline"]],
    IN_KITCHEN: [["خرج للتوصيل", "DISPATCHED", "btn-blue"]],
    DISPATCHED: [["العميل لا يرد", "UNREACHABLE_ALERT", "btn-outline"], ["تم التسليم", "FULFILLED", "btn-primary"]],
    UNREACHABLE_ALERT: [["الطيار وصل للعميل", "DISPATCHED", "btn-blue"], ["إلغاء", "CANCELLED", "btn-outline"]],
  }[order.status] || [];
  return actions
    .map(([label, st, cls]) => `<button class="btn btn-sm ${cls}" data-status="${st}" data-order="${order.id}">${label}</button>`)
    .join("");
}

function renderOrders() {
  const q = (document.getElementById("orderSearch").value || "").trim().toLowerCase();
  const filtered = allOrders.filter((o) => {
    if (activeStatusFilter !== "ALL" && o.status !== activeStatusFilter) return false;
    if (!q) return true;
    return (
      String(o.id).includes(q) ||
      (o.customer_name || "").toLowerCase().includes(q) ||
      (o.customer_phone || "").includes(q)
    );
  });

  if (!filtered.length) {
    ordersContainer.innerHTML = `<div class="empty-state">لا توجد طلبات حتى الآن</div>`;
    return;
  }

  ordersContainer.innerHTML = filtered.map((order) => `
    <article class="order-card ${order.status === "SUBMITTED" ? "new" : ""}">
      <div class="order-head">
        <div>
          <span class="order-id">طلب #${order.id} • ${escapeHtml(order.data_source)}</span>
          <h2>${escapeHtml(order.customer_name)}</h2>
        </div>
        <span class="status-badge ${order.status.toLowerCase()}">${statusLabel(order.status)}</span>
      </div>
      <div class="meta-row">
        <span>📍 ${escapeHtml(order.delivery_address)}</span>
        <span>📞 ${escapeHtml(order.customer_phone)}${order.backup_phone ? ` | بديل: ${escapeHtml(order.backup_phone)}` : ""}</span>
      </div>
      <div class="items-list">
        ${(order.items || []).map((i) => `<div class="item-row"><span>${i.quantity} × ${escapeHtml(i.name)}</span><strong>${escapeHtml(i.line_total || "")} ج.م</strong></div>`).join("")}
      </div>
      <div class="total-bar"><span>الإجمالي</span><span>${escapeHtml(order.total_amount)} ج.م</span></div>
      <div class="actions">
        ${actionButtons(order)}
        <button class="btn btn-sm btn-dark" data-direct="${order.id}">💬 توجيه للعميل</button>
      </div>
    </article>
  `).join("");
}

document.getElementById("statusFilters").addEventListener("click", (e) => {
  const pill = e.target.closest(".pill");
  if (!pill) return;
  document.querySelectorAll("#statusFilters .pill").forEach((p) => p.classList.remove("active"));
  pill.classList.add("active");
  activeStatusFilter = pill.dataset.status;
  renderOrders();
});
document.getElementById("orderSearch").addEventListener("input", renderOrders);
document.getElementById("refreshBtn").addEventListener("click", () => {
  loadOrders();
  loadMetrics();
  showToast("تم التحديث");
});

ordersContainer.addEventListener("click", async (e) => {
  const statusBtn = e.target.closest("[data-status]");
  if (statusBtn) {
    await apiFetch(`/api/orders/${statusBtn.dataset.order}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: statusBtn.dataset.status }),
    });
    loadOrders();
    loadMetrics();
    return;
  }
  const directBtn = e.target.closest("[data-direct]");
  if (directBtn) {
    directiveOrderId = directBtn.dataset.direct;
    directiveText.value = "";
    directiveType.value = "free_text";
    directivePreview.hidden = true;
    modal.showModal();
  }
});

// Conversations
async function loadConversations() {
  const res = await apiFetch("/api/conversations");
  if (!res.ok) return;
  const rows = await res.json();
  const body = document.getElementById("conversationsBody");
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#64748b;padding:28px;">لا توجد محادثات حتى الآن</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((c) => `
    <tr>
      <td><strong>${escapeHtml(c.phone_number)}</strong></td>
      <td>${c.active_order_id ? `#${c.active_order_id}` : "—"}</td>
      <td>${escapeHtml((c.conversation_state || {}).last_intent || "—")}</td>
      <td><span class="status-badge ${c.needs_human_takeover ? "unreachable_alert" : "fulfilled"}">${c.needs_human_takeover ? "مطلوب كاشير" : "آلي"}</span></td>
      <td>${c.needs_human_takeover ? `<button class="btn btn-sm btn-primary" data-resolve="${escapeHtml(c.phone_number)}">تم التعامل</button>` : "—"}</td>
    </tr>
  `).join("");
}

document.getElementById("conversationsBody").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-resolve]");
  if (!btn) return;
  await apiFetch(`/api/conversations/${encodeURIComponent(btn.dataset.resolve)}/resolve-handoff`, { method: "POST" });
  loadConversations();
  loadMetrics();
});

// Menu & Inventory
async function loadMenu() {
  const res = await apiFetch("/api/menu/all");
  if (!res.ok) return;
  allMenuItems = await res.json();
  const cats = [...new Set(allMenuItems.map((m) => m.category))];
  document.getElementById("categoryFilter").innerHTML =
    `<option value="ALL">كل الأقسام (${allMenuItems.length})</option>` + cats.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  renderMenu();
}

function renderMenu() {
  const cat = document.getElementById("categoryFilter").value;
  const q = (document.getElementById("menuSearch").value || "").trim().toLowerCase();
  const rows = allMenuItems.filter((m) => (cat === "ALL" || m.category === cat) && (!q || m.name.toLowerCase().includes(q)));
  document.getElementById("menuTableBody").innerHTML = rows.map((m) => `
    <tr>
      <td>${m.id}</td>
      <td><strong>${escapeHtml(m.name)}</strong></td>
      <td>${escapeHtml(m.category)}</td>
      <td>${escapeHtml(m.base_price)} ج.م</td>
      <td><span class="status-badge ${m.is_available ? "fulfilled" : "unreachable_alert"}">${m.is_available ? "متوفر" : "نفذ"}</span></td>
      <td><button class="btn btn-sm ${m.is_available ? "btn-outline" : "btn-primary"}" data-toggle-item="${m.id}" data-next="${!m.is_available}">${m.is_available ? "إيقاف" : "تفعيل"}</button></td>
    </tr>
  `).join("");
}

document.getElementById("categoryFilter").addEventListener("change", renderMenu);
document.getElementById("menuSearch").addEventListener("input", renderMenu);
document.getElementById("menuTableBody").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-toggle-item]");
  if (!btn) return;
  await apiFetch(`/api/menu/${btn.dataset.toggleItem}/availability`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_available: btn.dataset.next === "true" }),
  });
  loadMenu();
  loadMetrics();
});

// Integration Health (100% honest live status from /api/ready)
async function loadHealth() {
  const res = await fetch("/api/ready");
  const d = await res.json();
  const badge = (ok) => `<span class="status-badge ${ok ? "fulfilled" : "unreachable_alert"}">${ok ? "متصل" : "غير متصل"}</span>`;
  document.getElementById("healthGrid").innerHTML = `
    <article class="kpi-card"><span class="kpi-label">قاعدة البيانات (${escapeHtml(d.database_dialect)})</span>${badge(d.database_connected)}</article>
    <article class="kpi-card"><span class="kpi-label">Google Gemini LLM</span>${badge(d.providers.gemini_llm_configured)}</article>
    <article class="kpi-card"><span class="kpi-label">Groq Whisper Audio</span>${badge(d.providers.groq_whisper_configured)}</article>
    <article class="kpi-card"><span class="kpi-label">Meta WhatsApp Cloud API (v27.0)</span>${badge(d.providers.meta_whatsapp_configured)}</article>
    <article class="kpi-card"><span class="kpi-label">توقيع HMAC للـ Webhook</span>${badge(d.providers.meta_webhook_hmac_enabled)}</article>
  `;
}

// Staff & Audit Logs
async function loadStaffAndLogs() {
  if (currentUser.role !== "admin") return;
  const [sRes, aRes] = await Promise.all([apiFetch("/api/staff"), apiFetch("/api/audit-logs")]);
  const staff = await sRes.json();
  const logs = await aRes.json();
  document.getElementById("staffTableBody").innerHTML = staff.map((u) => `
    <tr><td>${escapeHtml(u.username)}</td><td>${escapeHtml(u.full_name)}</td><td>${escapeHtml(u.role)}</td><td>${u.is_active ? "نشط" : "معطل"}</td></tr>
  `).join("");
  document.getElementById("auditTableBody").innerHTML = logs.map((l) => `
    <tr><td>${escapeHtml(l.created_at)}</td><td>${escapeHtml(l.actor_username)}</td><td>${escapeHtml(l.action)}</td><td>${escapeHtml(l.target)}</td></tr>
  `).join("");
}

document.getElementById("createStaffForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const res = await apiFetch("/api/staff", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: document.getElementById("staffUsername").value.trim(),
      full_name: document.getElementById("staffFullName").value.trim(),
      password: document.getElementById("staffPassword").value,
      role: document.getElementById("staffRole").value,
    }),
  });
  if (res.ok) {
    showToast("تم إنشاء الحساب بنجاح");
    e.target.reset();
    loadStaffAndLogs();
  } else {
    const err = await res.json().catch(() => ({}));
    showToast(err.detail || "تعذر إنشاء الحساب");
  }
});

// Directive Modal
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    directiveType.value = btn.dataset.preset;
    if (btn.dataset.preset === "unreachable_client") directiveText.value = "الكابتن وصل تحت المنزل والعميل لا يرد";
    if (btn.dataset.preset === "kitchen_delay") directiveText.value = "تأخير مطبخ 15 دقيقة بسبب ضغط الطلبات";
    if (btn.dataset.preset === "out_of_stock") directiveText.value = "الاستربس خلصان، اعرض كريب زنجر حار كبديل";
  });
});
document.getElementById("cancelDirective").addEventListener("click", () => modal.close());
document.getElementById("closeDirectiveX").addEventListener("click", () => modal.close());
document.getElementById("directiveForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const res = await apiFetch(`/api/orders/${directiveOrderId}/instruct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      directive: directiveText.value,
      directive_type: directiveType.value,
      extra_delay_minutes: directiveType.value === "kitchen_delay" ? 15 : null,
    }),
  });
  const data = await res.json();
  directivePreview.hidden = false;
  directivePreview.innerHTML = `<strong>الرسالة للعميل:</strong><br>${escapeHtml(data.customer_message_ar)}`;
});

// Customer Experience, Reviews, Complaints & Notification Health (Genuine DB Data Only)
async function loadCustomerExperience() {
  const [ovRes, cRes] = await Promise.all([
    apiFetch("/api/customer-experience/overview"),
    apiFetch("/api/complaints"),
  ]);
  if (!ovRes.ok || !cRes.ok) return;
  const ov = await ovRes.json();
  const tickets = await cRes.json();

  const activeComplaints = (ov.complaints.open || 0) + (ov.complaints.escalated || 0) + (ov.complaints.acknowledged || 0);
  const badge = document.getElementById("navComplaintsBadge");
  if (badge) badge.textContent = activeComplaints;

  document.getElementById("cxKpiGrid").innerHTML = `
    <article class="kpi-card accent-green"><span class="kpi-label">متوسط التقييم الحقيقي</span><strong class="kpi-value">${ov.reviews.average_rating ?? "—"} ⭐ (${ov.reviews.total_submitted})</strong></article>
    <article class="kpi-card accent-blue"><span class="kpi-label">نسبة رضا العملاء (4-5⭐)</span><strong class="kpi-value">${ov.reviews.satisfaction_rate_pct !== null ? ov.reviews.satisfaction_rate_pct + "%" : "—"}</strong></article>
    <article class="kpi-card accent-orange"><span class="kpi-label">شكاوى مفتوحة / مصعّدة</span><strong class="kpi-value">${ov.complaints.open + ov.complaints.escalated}</strong></article>
    <article class="kpi-card"><span class="kpi-label">شكاوى محلولة</span><strong class="kpi-value">${ov.complaints.resolved}</strong></article>
    <article class="kpi-card"><span class="kpi-label">متوسط الاستجابة / الحل</span><strong class="kpi-value">${ov.complaints.avg_response_minutes ?? "—"}د / ${ov.complaints.avg_resolution_minutes ?? "—"}د</strong></article>
    <article class="kpi-card accent-indigo"><span class="kpi-label">نجاح التوصيات (Upsell)</span><strong class="kpi-value">${ov.recommendations.acceptance_rate_pct !== null ? ov.recommendations.acceptance_rate_pct + "%" : "—"} (${ov.recommendations.added_revenue_egp} ج.م)</strong></article>
  `;

  const cBody = document.getElementById("complaintsTableBody");
  if (!tickets.length) {
    cBody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#64748b;padding:22px;">لا توجد شكاوى مسجلة حتى الآن</td></tr>`;
  } else {
    cBody.innerHTML = tickets.map((t) => `
      <tr>
        <td>#${t.id}</td>
        <td>${t.order_id ? `#${t.order_id}` : "—"}</td>
        <td>${escapeHtml(t.category)}</td>
        <td><span class="status-badge ${t.severity === "CRITICAL" || t.severity === "HIGH" ? "unreachable_alert" : "submitted"}">${escapeHtml(t.severity)}</span></td>
        <td>${escapeHtml(t.summary)}</td>
        <td>${escapeHtml(t.status)}</td>
        <td>${escapeHtml(t.escalation_channel || "—")}</td>
        <td>${t.human_takeover_active ? "نشط (AI متوقف)" : "آلي"}</td>
        <td style="display:flex;gap:4px;flex-wrap:wrap;">
          ${!t.owner_acknowledged_at ? `<button class="btn btn-sm btn-blue" data-ack-complaint="${t.id}">تأكيد الاستلام</button>` : ""}
          ${t.status !== "RESOLVED" ? `<button class="btn btn-sm btn-primary" data-resolve-complaint="${t.id}">تم الحل</button>` : "✅ محلولة"}
        </td>
      </tr>
    `).join("");
  }

  const rBody = document.getElementById("reviewsTableBody");
  if (!ov.reviews.recent.length) {
    rBody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#64748b;padding:20px;">لا توجد تقييمات حقيقية بعد</td></tr>`;
  } else {
    rBody.innerHTML = ov.reviews.recent.map((r) => `
      <tr>
        <td>#${r.order_id}</td>
        <td><strong>${r.rating} / 5 ⭐</strong></td>
        <td>${escapeHtml(r.comment || r.issue_reported || "—")}</td>
        <td>${escapeHtml(r.submitted_at || "")}</td>
      </tr>
    `).join("");
  }

  const nBody = document.getElementById("notifAttemptsBody");
  if (!ov.notifications.recent_attempts.length) {
    nBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:#64748b;padding:20px;">لا توجد محاولات إشعار بعد</td></tr>`;
  } else {
    nBody.innerHTML = ov.notifications.recent_attempts.map((n) => `
      <tr>
        <td>${escapeHtml(n.type)}</td>
        <td>${escapeHtml(n.channel)}</td>
        <td><span class="status-badge ${n.status === "SENT" ? "fulfilled" : "unreachable_alert"}">${escapeHtml(n.status)}</span></td>
        <td>${n.order_id ? `#${n.order_id}` : "—"}</td>
        <td>${escapeHtml(n.last_error || "OK")}</td>
      </tr>
    `).join("");
  }
}

document.getElementById("complaintsTableBody").addEventListener("click", async (e) => {
  const ackBtn = e.target.closest("[data-ack-complaint]");
  if (ackBtn) {
    await apiFetch(`/api/complaints/${ackBtn.dataset.ackComplaint}/acknowledge`, { method: "POST" });
    showToast("تم تأكيد استلام الشكوى");
    loadCustomerExperience();
    return;
  }
  const resBtn = e.target.closest("[data-resolve-complaint]");
  if (resBtn) {
    await apiFetch(`/api/complaints/${resBtn.dataset.resolveComplaint}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolution_note: "Resolved via dashboard", release_human_takeover: true }),
    });
    showToast("تم إغلاق الشكوى وإعادة تفعيل الرد الآلي");
    loadCustomerExperience();
  }
});

// WebSocket
function connectWS() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/ws/cashier`);
  const dot = document.getElementById("wsDot");
  const conn = document.getElementById("connection");
  ws.onopen = () => { dot.className = "dot online"; conn.textContent = "متصل لحظياً (WS)"; };
  ws.onclose = () => { dot.className = "dot offline"; conn.textContent = "إعادة الاتصال..."; setTimeout(connectWS, 2500); };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (["order_created", "order_modified", "customer_alert", "complaint_escalated", "cashier_notification_failed"].includes(data.event_type)) {
      playChime();
    }
    if (data.event_type === "complaint_escalated") {
      showToast(`🚨 شكوى عاجلة جديدة (تذكرة #${data.payload?.ticket_id || ""})`);
    }
    if (data.event_type === "cashier_notification_failed") {
      showToast(`⚠️ تعذر إرسال إشعار التليجرام للطلب #${data.order_id} — تمت جدولة إعادة المحاولة`);
    }
    loadOrders();
    loadMetrics();
  };
}

bootstrapAuth();
