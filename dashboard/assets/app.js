const ordersContainer = document.querySelector("#orders");
const modal = document.querySelector("#directiveModal");
const directiveText = document.querySelector("#directiveText");
const directiveType = document.querySelector("#directiveType");
const directivePreview = document.querySelector("#directivePreview");
const toastStack = document.querySelector("#toastStack");

let allOrders = [];
let allMenuItems = [];
let activeStatusFilter = "ALL";
let directiveOrderId = null;
let audioContext = null;

function showToast(message) {
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  toastStack.appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

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

document.querySelector("#enableSound").addEventListener("click", async () => {
  audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
  await audioContext.resume();
  playChime();
  document.querySelector("#enableSound").textContent = "🔊 الصوت مفعّل";
});

// Sidebar View Switching
const viewTitles = {
  ordersView: "لوحة الطلبات الحية",
  menuView: "إدارة توافر المنيو (MOKA)",
  aiView: "محاكي الوكيل الذكي (Wasla AI)",
};
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.view;
    document.getElementById(target).classList.add("active");
    document.getElementById("viewTitle").textContent = viewTitles[target] || "";
    if (target === "menuView" && !allMenuItems.length) loadMenu();
  });
});

function escapeHtml(val) {
  const el = document.createElement("span");
  el.textContent = val ?? "";
  return el.innerHTML;
}

function statusLabel(status) {
  return {
    SUBMITTED: "طلب جديد",
    IN_KITCHEN: "في المطبخ",
    DISPATCHED: "خرج للتوصيل",
    UNREACHABLE_ALERT: "لا يمكن الوصول",
    FULFILLED: "تم التسليم",
    CANCELLED: "ملغي",
  }[status] || status;
}

async function loadMetrics() {
  const res = await fetch("/api/metrics");
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
}

async function loadOrders() {
  const res = await fetch("/api/orders");
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
    FULFILLED: [["إعادة للاختبار", "RESET_DEMO", "btn-outline"]],
    CANCELLED: [["إعادة للاختبار", "RESET_DEMO", "btn-outline"]],
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
      (o.customer_phone || "").includes(q) ||
      (o.delivery_address || "").toLowerCase().includes(q)
    );
  });

  if (!filtered.length) {
    ordersContainer.innerHTML = `<div class="empty-state">لا توجد طلبات مطابقة حالياً. يمكنك إنشاء طلب تجريبي من "محاكي الوكيل الذكي".</div>`;
    return;
  }

  ordersContainer.innerHTML = filtered.map((order) => `
    <article class="order-card ${order.status === "SUBMITTED" ? "new" : ""}">
      <div class="order-head">
        <div>
          <span class="order-id">طلب #${order.id}${order.is_test ? ' • <strong style="color:#d97706;">🧪 تجريبي (DEMO)</strong>' : ''}</span>
          <h2>${escapeHtml(order.customer_name)}</h2>
        </div>
        <span class="status-badge ${order.status.toLowerCase()}">${statusLabel(order.status)}</span>
      </div>
      <div class="meta-row">
        <span>📍 ${escapeHtml(order.delivery_address)}</span>
        <span>📞 ${escapeHtml(order.customer_phone)}${order.backup_phone ? ` | بديل: ${escapeHtml(order.backup_phone)}` : ""}</span>
      </div>
      <div class="items-list">
        ${(order.items || []).map((item) => `
          <div class="item-row">
            <span>${item.quantity} × ${escapeHtml(item.name)}</span>
            <strong>${escapeHtml(item.line_total || "")} ج.م</strong>
          </div>
        `).join("")}
      </div>
      ${order.latest_modification ? `<div class="notice-box">✏️ تعديل العميل: ${escapeHtml(order.latest_modification.message)}</div>` : ""}
      ${order.feedback && order.feedback.rating ? `<div class="feedback-box">⭐ تقييم العميل: ${"★".repeat(order.feedback.rating)} ${escapeHtml(order.feedback.comment || "")}</div>` : ""}
      <div class="total-bar">
        <span>الإجمالي شامل التوصيل</span>
        <span>${escapeHtml(order.total_amount)} ج.م</span>
      </div>
      <div class="actions">
        ${actionButtons(order)}
        <button class="btn btn-sm btn-dark" data-direct="${order.id}">💬 توجيه ذكي</button>
      </div>
    </article>
  `).join("");
}

// Filters & Search
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
  if (allMenuItems.length) loadMenu();
  showToast("تم تحديث البيانات");
});

// Order Card Actions
ordersContainer.addEventListener("click", async (e) => {
  const statusBtn = e.target.closest("[data-status]");
  if (statusBtn) {
    const id = statusBtn.dataset.order;
    const st = statusBtn.dataset.status;
    if (st === "RESET_DEMO") {
      await fetch(`/api/orders/${id}/reset-demo`, { method: "POST" });
    } else {
      await fetch(`/api/orders/${id}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: st }),
      });
    }
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

// Cashier Directive Presets & Submission
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const type = btn.dataset.preset;
    directiveType.value = type;
    if (type === "unreachable_client") directiveText.value = "الكابتن وصل تحت المنزل والعميل لا يرد";
    if (type === "kitchen_delay") directiveText.value = "تأخير مطبخ 15 دقيقة بسبب ضغط الطلبات";
    if (type === "out_of_stock") directiveText.value = "الاستربس خلصان، اعرض كريب زنجر حار كبديل";
  });
});

document.getElementById("cancelDirective").addEventListener("click", () => modal.close());
document.getElementById("closeDirectiveX").addEventListener("click", () => modal.close());

document.getElementById("directiveForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    directive: directiveText.value,
    directive_type: directiveType.value,
    extra_delay_minutes: directiveType.value === "kitchen_delay" ? 15 : null,
  };
  const res = await fetch(`/api/orders/${directiveOrderId}/instruct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  directivePreview.hidden = false;
  directivePreview.innerHTML = `<strong>الرسالة الجاهزة للعميل (Wasla AI):</strong><br>${escapeHtml(data.customer_message_ar)}${data.warnings && data.warnings.length ? `<br><small>⚠️ ${escapeHtml(data.warnings.join(" | "))}</small>` : ""}`;
  showToast(`تم تجهيز توجيه طلب #${directiveOrderId}`);
});

// Menu Availability View
async function loadMenu() {
  const res = await fetch("/api/menu/all");
  if (!res.ok) return;
  allMenuItems = await res.json();
  const cats = [...new Set(allMenuItems.map((m) => m.category))];
  const select = document.getElementById("categoryFilter");
  select.innerHTML = `<option value="ALL">كل الأقسام (${allMenuItems.length})</option>` + cats.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  renderMenu();
}

function renderMenu() {
  const cat = document.getElementById("categoryFilter").value;
  const q = (document.getElementById("menuSearch").value || "").trim().toLowerCase();
  const rows = allMenuItems.filter((m) => {
    if (cat !== "ALL" && m.category !== cat) return false;
    if (q && !m.name.toLowerCase().includes(q) && !m.category.toLowerCase().includes(q)) return false;
    return true;
  });
  document.getElementById("menuTableBody").innerHTML = rows.map((m) => `
    <tr>
      <td>${m.id}</td>
      <td><strong>${escapeHtml(m.name)}</strong></td>
      <td>${escapeHtml(m.category)}</td>
      <td>${escapeHtml(m.base_price)} ج.م</td>
      <td><span class="status-badge ${m.is_available ? "fulfilled" : "unreachable_alert"}">${m.is_available ? "متوفر" : "نفذ"}</span></td>
      <td>
        <button class="btn btn-sm ${m.is_available ? "btn-outline" : "btn-primary"}" data-toggle-item="${m.id}" data-next="${!m.is_available}">
          ${m.is_available ? "تعيين غير متوفر" : "إعادة للتوفر"}
        </button>
      </td>
    </tr>
  `).join("");
}

document.getElementById("categoryFilter").addEventListener("change", renderMenu);
document.getElementById("menuSearch").addEventListener("input", renderMenu);
document.getElementById("menuTableBody").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-toggle-item]");
  if (!btn) return;
  const id = btn.dataset.toggleItem;
  const isAvailable = btn.dataset.next === "true";
  await fetch(`/api/menu/${id}/availability`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_available: isAvailable }),
  });
  await loadMenu();
  await loadMetrics();
  showToast(isAvailable ? "تم إعادة الصنف للمنيو" : "تم إيقاف الصنف مؤقتاً");
});

// Wasla AI Live Simulator
document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chatInput");
  const phone = document.getElementById("simPhone").value.trim() || "01012345678";
  const text = input.value.trim();
  if (!text) return;

  const box = document.getElementById("chatMessages");
  box.innerHTML += `<div class="msg customer">${escapeHtml(text)}</div>`;
  input.value = "";
  box.scrollTop = box.scrollHeight;

  const res = await fetch("/api/ai/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ phone_number: phone, message: text }),
  });
  const data = await res.json();
  box.innerHTML += `<div class="msg assistant">${escapeHtml(data.reply_ar || "حدث خطأ")}</div>`;
  box.scrollTop = box.scrollHeight;

  document.getElementById("inspIntent").textContent = `${data.intent} (conf: ${data.confidence})`;
  document.getElementById("inspModel").textContent = data.model_used || "—";
  document.getElementById("inspCart").textContent = JSON.stringify(data.cart, null, 2);

  if (data.order_id) {
    showToast(`🎉 تم تثبيت طلب #${data.order_id} عبر Wasla AI!`);
    loadOrders();
    loadMetrics();
  }
});

// WebSocket Connection
function connectWS() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${location.host}/ws/cashier`);
  const dot = document.getElementById("wsDot");
  const conn = document.getElementById("connection");

  ws.onopen = () => {
    dot.className = "dot online";
    conn.textContent = "متصل لحظياً (WS)";
  };
  ws.onclose = () => {
    dot.className = "dot offline";
    conn.textContent = "إعادة الاتصال...";
    setTimeout(connectWS, 2500);
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (["order_created", "order_modified", "customer_alert"].includes(data.event_type)) {
      playChime();
      document.body.classList.add("alert");
      setTimeout(() => document.body.classList.remove("alert"), 1200);
    }
    loadOrders();
    loadMetrics();
  };
}

loadOrders();
loadMetrics();
connectWS();
setInterval(() => { loadOrders(); loadMetrics(); }, 30000);
