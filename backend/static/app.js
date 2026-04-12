/* ═══════════════════════════════════════════════════════════════
   Credit Card Approval — Full Platform JS (Final Build)
   ═══════════════════════════════════════════════════════════════ */

// ── Auth state ───────────────────────────────────────────────
let authToken = localStorage.getItem("authToken");

function loadStoredAuthUser() {
    try {
        const raw = localStorage.getItem("authUser");
        if (!raw || raw === "null") return null;
        const u = JSON.parse(raw);
        return u && typeof u === "object" ? u : null;
    } catch {
        localStorage.removeItem("authUser");
        return null;
    }
}

let authUser = loadStoredAuthUser();
/** "customer" | "staff" — controls auth modal copy (register is always customer). */
let authAudience = "customer";
let currentDraftId = null;
let currentView = "apply";
let notifInterval = null;

const STAFF_ROLES = ["admin", "officer"];
const isStaff = () => authUser && STAFF_ROLES.includes(authUser.role);
const isAdmin = () => authUser && authUser.role === "admin";
const esc = (s) => { const d = document.createElement("div"); d.textContent = s == null ? "" : String(s); return d.innerHTML; };

function authHeaders() {
    const h = { "Content-Type": "application/json" };
    if (authToken) h["Authorization"] = `Bearer ${authToken}`;
    return h;
}

// ── Toasts ───────────────────────────────────────────────────
function toast(msg, type = "info") {
    const c = document.getElementById("toast-container");
    const el = document.createElement("div");
    el.className = `toast ${type}`;
    el.textContent = msg;
    c.appendChild(el);
    setTimeout(() => { el.classList.add("fadeout"); setTimeout(() => el.remove(), 300); }, 3000);
}

function isFetchNetworkError(err) {
    return (
        err &&
        (err.name === "TypeError" || err instanceof TypeError) &&
        String(err.message || "").toLowerCase().includes("fetch")
    );
}

function backendUnreachableHint() {
    if (window.location && window.location.protocol === "file:") {
        return "You opened the page from your filesystem (file://). Start the server and use http://127.0.0.1:8000 instead.";
    }
    return "Backend not reachable. Make sure the server is running (python3 backend/asgi.py) and you’re on http://127.0.0.1:8000.";
}

// ── Status helpers ───────────────────────────────────────────
function statusPill(status) {
    const cls = (status || "").replace(/_/g, "-");
    const label = (status || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    return `<span class="pill ${cls}">${esc(label)}</span>`;
}

function rolePill(role) {
    return `<span class="role-pill ${role}">${role.charAt(0).toUpperCase() + role.slice(1)}</span>`;
}

// ── View switching ───────────────────────────────────────────
function switchView(view) {
    currentView = view;
    document.querySelectorAll("[data-view-panel]").forEach(el => {
        el.classList.toggle("hidden", el.getAttribute("data-view-panel") !== view);
    });
    document.querySelectorAll(".side-nav-btn, .nav-tab").forEach(btn => {
        if (btn.hidden) return;
        btn.classList.toggle("active", btn.dataset.view === view);
    });
    if (view === "dashboard") loadDashboard();
    if (view === "history") loadHistory();
    if (view === "bank") loadBankQueue();
    if (view === "users") loadUsers();
    if (view === "audit") loadAuditLog();
    if (view === "profile") loadProfile();
    if (view === "drafts") loadDraftsList();
    if (view === "whatif") triggerWhatIf();
    if (view === "whatif") loadWhatIfAppOptions();
    if (view === "fairness") loadFairness();
}

function setLandingVisible(visible) {
    const el = document.getElementById("view-landing");
    if (el) el.classList.toggle("hidden", !visible);
}

function setMainShellVisible(visible) {
    const shell = document.getElementById("main-shell");
    if (!shell) return;
    shell.classList.toggle("hidden", !visible);
}

function applyAuthModalCopy() {
    const staff = authAudience === "staff";
    const loginTitle = document.getElementById("login-title");
    const loginSub = document.getElementById("login-subtitle");
    const custSwitch = document.getElementById("login-auth-switch-customer");
    const staffNote = document.getElementById("login-auth-switch-staff");
    if (loginTitle) loginTitle.textContent = staff ? "Staff sign in" : "Welcome back";
    if (loginSub) {
        loginSub.textContent = staff
            ? "Use the email and password issued to your bank account."
            : "Log in to save and track your applications.";
    }
    if (custSwitch) custSwitch.style.display = staff ? "none" : "block";
    if (staffNote) staffNote.style.display = staff ? "block" : "none";
}

function renderSideNav() {
    const title = document.getElementById("side-nav-title");
    const sub = document.getElementById("side-nav-sub");
    const main = document.getElementById("side-nav-main");
    const secondary = document.getElementById("side-nav-secondary");
    if (!title || !sub || !main || !secondary) return;

    if (!authUser) {
        title.textContent = "—";
        sub.textContent = "—";
        main.innerHTML = "";
        secondary.innerHTML = "";
        return;
    }

    const isCust = authUser.role === "customer";
    title.textContent = isCust ? "Customer" : (authUser.role === "admin" ? "Admin" : "Staff");
    sub.textContent = isCust ? "Apply, history, and what-if." : "Review applications and check fairness.";

    const btn = (view, label, pill) => `
        <button type="button" class="side-nav-btn ${currentView === view ? "active" : ""}" data-view="${view}">
            <span>${esc(label)}</span>
            ${pill ? `<span class="side-nav-pill">${esc(pill)}</span>` : ""}
        </button>
    `;

    if (isCust) {
        main.innerHTML = [
            btn("apply", "Apply", "New"),
            btn("history", "History", null),
            btn("whatif", "What If?", null),
        ].join("");
        secondary.innerHTML = [
            btn("drafts", "Drafts", null),
            btn("profile", "Account", null),
        ].join("");
    } else {
        main.innerHTML = [
            btn("bank", "Applications", null),
            btn("fairness", "Fairness", "Bias"),
            btn("audit", "Audit Log", null),
        ].join("");
        secondary.innerHTML = [
            btn("dashboard", "Overview", null),
            (isAdmin() ? btn("users", "Users", null) : ""),
        ].join("");
    }
}

// ── Auth UI ──────────────────────────────────────────────────
function updateAuthUI() {
    const bar = document.getElementById("auth-bar");
    const notifWrap = document.getElementById("notif-wrapper");

    if (!bar) return;

    if (authUser) {
        setLandingVisible(false);
        setMainShellVisible(true);
        bar.innerHTML = `
            <span class="user-info">
                <strong>${esc(authUser.full_name)}</strong>
                ${rolePill(authUser.role)}
            </span>
            <button type="button" class="auth-btn small" onclick="logout()">Log Out</button>
        `;
        renderSideNav();
        if (notifWrap) notifWrap.style.display = "";
        const draftBar = document.getElementById("draft-bar");
        if (draftBar) draftBar.style.display = "";
        startNotifPolling();
        loadDraftOptions();
    } else {
        setLandingVisible(true);
        setMainShellVisible(false);
        authAudience = "customer";
        bar.innerHTML = `
            <button type="button" class="auth-btn" data-open-modal="login">Log In</button>
            <button type="button" class="auth-btn primary" data-open-modal="register">Sign Up</button>
        `;
        renderSideNav();
        if (notifWrap) notifWrap.style.display = "none";
        const draftBar = document.getElementById("draft-bar");
        if (draftBar) draftBar.style.display = "none";
        stopNotifPolling();
    }
}

function openModal(mode, audience) {
    if (mode === "register") authAudience = "customer";
    else if (audience) authAudience = audience;
    else authAudience = "customer";

    if (mode === "register") {
        const roleToggle = document.getElementById("register-role-toggle");
        const codeWrap = document.getElementById("register-admin-code-wrap");
        const codeInput = document.getElementById("register-admin-code");
        if (roleToggle) {
            roleToggle.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
            const cust = roleToggle.querySelector('.toggle-btn[data-value="customer"]');
            if (cust) cust.classList.add("active");
        }
        if (codeWrap) codeWrap.style.display = "none";
        if (codeInput) codeInput.value = "";
    }

    document.getElementById("auth-modal").classList.add("active");
    switchModal(mode);
    applyAuthModalCopy();
}

function closeModal() {
    document.getElementById("auth-modal").classList.remove("active");
    document.querySelectorAll(".auth-error").forEach(e => { e.classList.remove("visible"); e.textContent = ""; });
}

function switchModal(mode) {
    document.getElementById("login-form").style.display = mode === "login" ? "block" : "none";
    document.getElementById("register-form").style.display = mode === "register" ? "block" : "none";
    document.querySelectorAll(".auth-error").forEach(e => { e.classList.remove("visible"); e.textContent = ""; });
    applyAuthModalCopy();
}
function showAuthError(id, msg) { const el = document.getElementById(id); el.textContent = msg; el.classList.add("visible"); }

async function handleLogin() {
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value;
    if (!email || !password) { showAuthError("login-error", "Please fill in all fields."); return; }
    try {
        const resp = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Login failed");
        authToken = data.token;
        authUser = { email: data.email, full_name: data.full_name, role: data.role };
        localStorage.setItem("authToken", authToken);
        localStorage.setItem("authUser", JSON.stringify(authUser));
        closeModal();
        updateAuthUI();
        if (isStaff()) switchView("bank");
        else switchView("apply");
        toast("Welcome back, " + data.full_name, "success");
    } catch (err) { showAuthError("login-error", err.message); }
}

async function handleRegister() {
    const roleToggle = document.getElementById("register-role-toggle");
    const roleBtn = roleToggle ? roleToggle.querySelector(".toggle-btn.active") : null;
    const role = roleBtn ? roleBtn.dataset.value : "customer";
    const enrollment_code = (document.getElementById("register-admin-code")?.value || "").trim() || null;
    const full_name = document.getElementById("register-name").value.trim();
    const email = document.getElementById("register-email").value.trim();
    const password = document.getElementById("register-password").value;
    if (!full_name || !email || !password) { showAuthError("register-error", "Please fill in all fields."); return; }
    if (role === "admin" && !enrollment_code) { showAuthError("register-error", "Admin enrollment code is required."); return; }
    if (password.length < 6) { showAuthError("register-error", "Password must be at least 6 characters."); return; }
    try {
        const resp = await fetch("/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password, full_name, role, enrollment_code }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Registration failed");
        authToken = data.token;
        authUser = { email: data.email, full_name: data.full_name, role: data.role };
        localStorage.setItem("authToken", authToken);
        localStorage.setItem("authUser", JSON.stringify(authUser));
        closeModal();
        updateAuthUI();
        if (isStaff()) switchView("bank");
        else switchView("apply");
        toast("Account created! Welcome, " + data.full_name, "success");
    } catch (err) { showAuthError("register-error", err.message); }
}

function logout() {
    authToken = null; authUser = null;
    localStorage.removeItem("authToken"); localStorage.removeItem("authUser");
    const sel = document.getElementById("wi-load-app-select");
    if (sel) { delete sel.dataset.loaded; sel.innerHTML = '<option value="">No previous applications</option>'; }
    updateAuthUI();
    toast("Logged out", "info");
}

// ── Notifications ───────────────────────────────────────────
function startNotifPolling() {
    pollNotifCount();
    if (notifInterval) clearInterval(notifInterval);
    notifInterval = setInterval(pollNotifCount, 30000);
}
function stopNotifPolling() {
    if (notifInterval) { clearInterval(notifInterval); notifInterval = null; }
}

async function pollNotifCount() {
    if (!authToken) return;
    try {
        const resp = await fetch("/notifications/unread-count", { headers: authHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        const badge = document.getElementById("notif-badge");
        if (data.count > 0) {
            badge.textContent = data.count > 99 ? "99+" : data.count;
            badge.style.display = "flex";
        } else {
            badge.style.display = "none";
        }
    } catch (e) {}
}

function toggleNotifDropdown() {
    const dd = document.getElementById("notif-dropdown");
    const isOpen = dd.classList.contains("open");
    if (isOpen) {
        dd.classList.remove("open");
    } else {
        dd.classList.add("open");
        loadNotifications();
    }
}

// Close dropdown when clicking outside
document.addEventListener("click", (e) => {
    const wrapper = document.getElementById("notif-wrapper");
    if (wrapper && !wrapper.contains(e.target)) {
        document.getElementById("notif-dropdown").classList.remove("open");
    }
});

async function loadNotifications() {
    const list = document.getElementById("notif-list");
    if (!authToken) return;
    try {
        const resp = await fetch("/notifications", { headers: authHeaders() });
        if (!resp.ok) return;
        const notifs = await resp.json();
        if (notifs.length === 0) {
            list.innerHTML = '<div class="empty-state" style="padding:20px;"><p>No notifications yet</p></div>';
            return;
        }
        list.innerHTML = notifs.map(n => {
            const time = timeAgo(n.created_at);
            return `<div class="notif-item ${n.read ? "" : "unread"}">
                <div class="notif-item-msg"><span class="notif-type-badge ${n.type}"></span>${esc(n.message)}</div>
                <div class="notif-item-time">${time}</div>
            </div>`;
        }).join("");
    } catch (e) { list.innerHTML = '<div class="empty-state" style="padding:20px;"><p>Could not load</p></div>'; }
}

async function markAllRead() {
    if (!authToken) return;
    try {
        await fetch("/notifications/read-all", { method: "POST", headers: authHeaders() });
        document.getElementById("notif-badge").style.display = "none";
        loadNotifications();
    } catch (e) {}
}

function timeAgo(dateStr) {
    const now = new Date();
    const d = new Date(dateStr);
    const diff = Math.floor((now - d) / 1000);
    if (diff < 60) return "just now";
    if (diff < 3600) return Math.floor(diff / 60) + "m ago";
    if (diff < 86400) return Math.floor(diff / 3600) + "h ago";
    return Math.floor(diff / 86400) + "d ago";
}

// ── Dashboard ────────────────────────────────────────────────
async function loadDashboard() {
    const container = document.getElementById("dashboard-stats");
    const title = document.getElementById("dashboard-title");
    const sub = document.getElementById("dashboard-sub");
    if (!authToken) return;

    if (isStaff()) {
        title.textContent = "Staff dashboard";
        sub.textContent = "Bank-wide metrics, queue preview, and audit activity.";
        try {
            const resp = await fetch("/admin/stats", { headers: authHeaders() });
            if (!resp.ok) return;
            const s = await resp.json();
            container.innerHTML = `
                <div class="stat-card"><div class="stat-value blue">${s.total_applications}</div><div class="stat-label">Total Applications</div></div>
                <div class="stat-card"><div class="stat-value">${s.total_today}</div><div class="stat-label">Today</div></div>
                <div class="stat-card"><div class="stat-value">${s.total_this_week}</div><div class="stat-label">This Week</div></div>
                <div class="stat-card"><div class="stat-value green">${s.approved}</div><div class="stat-label">Approved</div></div>
                <div class="stat-card"><div class="stat-value red">${s.rejected}</div><div class="stat-label">Rejected</div></div>
                <div class="stat-card"><div class="stat-value amber">${s.under_review}</div><div class="stat-label">Under Review</div></div>
                <div class="stat-card"><div class="stat-value">${s.approval_rate}%</div><div class="stat-label">Approval Rate</div></div>
                <div class="stat-card"><div class="stat-value amber">${s.avg_risk}%</div><div class="stat-label">Avg Risk</div></div>
                <div class="stat-card"><div class="stat-value red">${s.flagged_count}</div><div class="stat-label">Flagged</div></div>
                <div class="stat-card"><div class="stat-value amber">${s.low_confidence_count}</div><div class="stat-label">Low Confidence</div></div>
                <div class="stat-card"><div class="stat-value blue">${s.total_users}</div><div class="stat-label">Registered Users</div></div>
            `;
        } catch (e) { container.innerHTML = '<p class="empty-state">Could not load stats.</p>'; }
    } else {
        title.textContent = "My dashboard";
        sub.textContent = "Your application stats, shortcuts, and recent activity.";
        try {
            const resp = await fetch("/stats/me", { headers: authHeaders() });
            if (!resp.ok) return;
            const s = await resp.json();
            container.innerHTML = `
                <div class="stat-card"><div class="stat-value blue">${s.total_applications}</div><div class="stat-label">Applications</div></div>
                <div class="stat-card"><div class="stat-value green">${s.approved}</div><div class="stat-label">Approved</div></div>
                <div class="stat-card"><div class="stat-value red">${s.rejected}</div><div class="stat-label">Rejected</div></div>
                <div class="stat-card"><div class="stat-value amber">${s.under_review}</div><div class="stat-label">Under Review</div></div>
                <div class="stat-card"><div class="stat-value">${s.approval_rate}%</div><div class="stat-label">Approval Rate</div></div>
                <div class="stat-card"><div class="stat-value blue">${s.best_credit_score != null ? s.best_credit_score : "—"}</div><div class="stat-label">Best Credit Score</div></div>
                <div class="stat-card"><div class="stat-value">${s.latest_decision || "—"}</div><div class="stat-label">Latest Decision</div></div>
            `;
        } catch (e) { container.innerHTML = '<p class="empty-state">Could not load stats.</p>'; }
    }

    await loadDashboardExtras();
}

async function loadDashboardExtras() {
    const qa = document.getElementById("dashboard-quick-actions");
    const actionSub = document.getElementById("dashboard-actions-sub");
    const st = document.getElementById("dashboard-secondary-title");
    const ss = document.getElementById("dashboard-secondary-sub");
    const sb = document.getElementById("dashboard-secondary-body");
    const tt = document.getElementById("dashboard-tertiary-title");
    const ts = document.getElementById("dashboard-tertiary-sub");
    const tb = document.getElementById("dashboard-tertiary-body");
    if (!qa || !authToken) return;

    if (isStaff()) {
        if (actionSub) actionSub.textContent = "Review queues, run tools, manage users (admins).";
        let buttons = `
            <button type="button" class="quick-action-btn primary" onclick="switchView('bank')">Bank queue</button>
            <button type="button" class="quick-action-btn" onclick="switchView('audit')">Audit log</button>
            <button type="button" class="quick-action-btn" onclick="exportCSV()">Export CSV</button>
            <button type="button" class="quick-action-btn" onclick="switchView('apply')">Run evaluation</button>
            <button type="button" class="quick-action-btn" onclick="switchView('whatif')">What-if simulator</button>
            <button type="button" class="quick-action-btn" onclick="switchView('profile')">Profile</button>
        `;
        if (isAdmin()) {
            buttons += `<button type="button" class="quick-action-btn" onclick="switchView('users')">User management</button>`;
        }
        qa.innerHTML = `<div class="quick-actions-grid">${buttons}</div>`;

        if (st) st.textContent = "Latest applications";
        if (ss) ss.textContent = "Newest records from the database — click a row to open details.";
        if (sb) {
            sb.innerHTML = '<p class="dash-loading">Loading…</p>';
            try {
                const resp = await fetch("/admin/applications", { headers: authHeaders() });
                const apps = resp.ok ? await resp.json() : [];
                const slice = apps.slice(0, 6);
                if (slice.length === 0) {
                    sb.innerHTML = '<div class="empty-state mini"><p>No applications yet.</p></div>';
                } else {
                    sb.innerHTML = `<div class="dash-mini-list">${slice.map(a => `
                        <button type="button" class="dash-mini-item" onclick="openAppDetail('${a.id}')">
                            <div class="dash-mini-left">
                                <strong>${esc(a.applicant_name || "—")}</strong>
                                <span class="history-meta">${esc(a.applicant_email || "")}</span>
                            </div>
                            <div class="dash-mini-right">
                                ${statusPill(a.status)}
                                <span class="dash-mini-meta">${a.credit_score} pts · ${a.probability_risky}% risk</span>
                            </div>
                        </button>`).join("")}</div>`;
                }
            } catch {
                sb.innerHTML = '<p class="empty-state">Could not load queue preview.</p>';
            }
        }

        if (tt) tt.textContent = "Recent audit";
        if (ts) ts.textContent = "Staff overrides and flags from the database.";
        if (tb) {
            tb.innerHTML = '<p class="dash-loading">Loading…</p>';
            try {
                const resp = await fetch("/admin/audit", { headers: authHeaders() });
                const logs = resp.ok ? await resp.json() : [];
                const slice = logs.slice(0, 6);
                if (slice.length === 0) {
                    tb.innerHTML = '<div class="empty-state mini"><p>No audit entries yet.</p></div>';
                } else {
                    tb.innerHTML = slice.map(l => `
                        <div class="dash-audit-item">
                            <div><strong>${esc(l.action)}</strong> <span class="history-meta">${esc(l.staff_name)}</span></div>
                            <div class="audit-note">${esc(l.note || "—")}</div>
                            <div class="history-meta">${new Date(l.created_at).toLocaleString()}</div>
                        </div>`).join("");
                }
            } catch {
                tb.innerHTML = '<p class="empty-state">Could not load audit.</p>';
            }
        }
    } else {
        if (actionSub) actionSub.textContent = "Submit applications and track outcomes.";
        qa.innerHTML = `
            <div class="quick-actions-grid">
                <button type="button" class="quick-action-btn primary" onclick="switchView('apply')">New application</button>
                <button type="button" class="quick-action-btn" onclick="switchView('whatif')">What-if simulator</button>
                <button type="button" class="quick-action-btn" onclick="switchView('history')">My applications</button>
                <button type="button" class="quick-action-btn" onclick="switchView('drafts')">Drafts</button>
                <button type="button" class="quick-action-btn" onclick="switchView('profile')">Profile</button>
            </div>`;

        if (st) st.textContent = "Your recent applications";
        if (ss) ss.textContent = "Stored in your account — click for full detail.";
        if (sb) {
            sb.innerHTML = '<p class="dash-loading">Loading…</p>';
            try {
                const resp = await fetch("/applications", { headers: authHeaders() });
                const apps = resp.ok ? await resp.json() : [];
                const slice = apps.slice(0, 6);
                if (slice.length === 0) {
                    sb.innerHTML = '<div class="empty-state mini"><p>No applications yet. Start with <strong>New application</strong>.</p></div>';
                } else {
                    sb.innerHTML = `<div class="dash-mini-list">${slice.map(a => `
                        <button type="button" class="dash-mini-item" onclick="openAppDetail('${a.id}')">
                            <div class="dash-mini-left">
                                ${statusPill(a.status)}
                                <span class="dash-mini-meta">Score ${a.credit_score} · ${a.probability_risky}% risk</span>
                            </div>
                            <div class="dash-mini-right history-meta">${new Date(a.created_at).toLocaleString()}</div>
                        </button>`).join("")}</div>`;
                }
            } catch {
                sb.innerHTML = '<p class="empty-state">Could not load applications.</p>';
            }
        }

        if (tt) tt.textContent = "Notifications";
        if (ts) ts.textContent = "From your account — mark all read from the bell menu.";
        if (tb) {
            tb.innerHTML = '<p class="dash-loading">Loading…</p>';
            try {
                const resp = await fetch("/notifications", { headers: authHeaders() });
                const notifs = resp.ok ? await resp.json() : [];
                const slice = notifs.slice(0, 6);
                if (slice.length === 0) {
                    tb.innerHTML = '<div class="empty-state mini"><p>No notifications yet.</p></div>';
                } else {
                    tb.innerHTML = slice.map(n => `
                        <div class="dash-notif-item ${n.read ? "" : "unread"}">
                            <span class="notif-type-badge ${n.type}"></span>
                            <div><div class="notif-line">${esc(n.message)}</div><div class="history-meta">${timeAgo(n.created_at)}</div></div>
                        </div>`).join("");
                }
            } catch {
                tb.innerHTML = '<p class="empty-state">Could not load notifications.</p>';
            }
        }
    }
}

async function validateSession() {
    if (!authToken) return;
    try {
        const r = await fetch("/auth/me", { headers: authHeaders() });
        if (!r.ok) throw new Error("session");
        const d = await r.json();
        authUser = { email: d.email, full_name: d.full_name, role: d.role };
        localStorage.setItem("authUser", JSON.stringify(authUser));
    } catch {
        authToken = null;
        authUser = null;
        localStorage.removeItem("authToken");
        localStorage.removeItem("authUser");
    }
}

// ── My Applications (customer) ───────────────────────────────
async function loadHistory() {
    const container = document.getElementById("history-list");
    if (!authToken || !container) return;
    try {
        const resp = await fetch("/applications", { headers: authHeaders() });
        if (!resp.ok) return;
        const apps = await resp.json();
        if (apps.length === 0) {
            container.innerHTML = '<div class="empty-state"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>No applications yet. Submit one on the <strong>Apply</strong> tab.</p></div>';
            return;
        }
        let html = "";
        apps.forEach(app => {
            const date = new Date(app.created_at).toLocaleDateString();
            const conf = app.confidence_score != null ? `<span style="font-size:0.78rem;color:var(--text-muted);">Conf: ${app.confidence_score.toFixed(0)}%</span>` : "";
            html += `
                <div class="history-item" onclick="openAppDetail('${app.id}')">
                    <div style="display:flex;align-items:center;gap:10px;">
                        ${statusPill(app.status)}
                        <span style="font-size:0.88rem;font-weight:500;">Score: ${app.credit_score}</span>
                        ${conf}
                        ${app.flagged ? '<span class="pill flag">Flagged</span>' : ""}
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem;font-weight:500;">Risk: ${app.probability_risky}%</div>
                        <div class="history-meta">${date}</div>
                    </div>
                </div>`;
        });
        container.innerHTML = html;
    } catch (e) { container.innerHTML = '<p class="empty-state">Could not load history.</p>'; }
}

// ── Re-apply (pre-fill form) ─────────────────────────────────
function prefillForm(d) {
    const toggles = { gender: d.gender, own_car: d.own_car, own_realty: d.own_realty, work_phone: d.work_phone, phone: d.phone, email: d.email_flag };
    Object.entries(toggles).forEach(([field, val]) => {
        const group = document.querySelector(`#view-apply .toggle-group[data-field="${field}"]`);
        if (!group) return;
        group.querySelectorAll(".toggle-btn").forEach(btn => {
            btn.classList.toggle("active", btn.dataset.value === String(val));
        });
    });
    const fields = { age: d.age, children: d.children, family_members: d.family_members, income: d.income, years_employed: d.years_employed };
    Object.entries(fields).forEach(([id, val]) => { const el = document.getElementById(id); if (el) el.value = val; });
    const selects = { family_status: d.family_status, education: d.education, income_type: d.income_type, occupation: d.occupation, housing: d.housing };
    Object.entries(selects).forEach(([id, val]) => { const el = document.getElementById(id); if (el) el.value = val; });
    closeDetail();
    switchView("apply");
    toast("Form pre-filled from previous application", "info");
}

// ── Application Detail (slide-out) ───────────────────────────
const INCOME_TYPES = {0:"Working",1:"Commercial Associate",2:"Pensioner",3:"State Servant",4:"Student"};
const EDUCATION_TYPES = {0:"Lower Secondary",1:"Secondary / Special",2:"Incomplete Higher",3:"Higher Education",4:"Academic Degree"};
const FAMILY_STATUS_MAP = {0:"Single / Not Married",1:"Married",2:"Separated",3:"Civil Marriage",4:"Widow"};
const HOUSING_TYPES = {0:"With Parents",1:"Rented Apartment",2:"Municipal Apartment",3:"Co-op Apartment",4:"House / Apartment (owned)",5:"Office Apartment"};
const OCCUPATION_TYPES = {0:"Laborers",1:"Core Staff",2:"Sales Staff",3:"Managers",4:"Drivers",5:"High Skill Tech Staff",6:"Accountants",7:"Medicine Staff",8:"Cooking Staff",9:"Security Staff",10:"Cleaning Staff",11:"Private Service Staff",12:"Low-skill Laborers",13:"Waiters / Barmen Staff",14:"Secretaries",15:"Realty Agents",16:"HR Staff",17:"IT Staff",18:"Other / Unknown"};

async function openAppDetail(appId) {
    try {
        const resp = await fetch(`/applications/${appId}`, { headers: authHeaders() });
        if (!resp.ok) { toast("Could not load application", "error"); return; }
        const d = await resp.json();
        const body = document.getElementById("detail-body");

        const gender = d.gender === 0 ? "Male" : "Female";
        const car = d.own_car ? "Yes" : "No";
        const realty = d.own_realty ? "Yes" : "No";
        const confScore = d.confidence_score != null ? d.confidence_score.toFixed(1) + "%" : "N/A";
        const confClass = d.confidence_score >= 80 ? "success" : d.confidence_score >= 65 ? "warning" : "danger";

        let html = `
            <div class="detail-section">
                <h4>Decision</h4>
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    ${statusPill(d.status)}
                    ${d.flagged ? '<span class="pill flag">Flagged</span>' : ""}
                    ${d.confidence_score != null && d.confidence_score < 65 ? '<span class="pill low-conf">Low Confidence</span>' : ""}
                </div>
                <div class="detail-grid">
                    <div class="detail-field"><div class="detail-label">Credit Score</div><div class="detail-val">${d.credit_score}</div></div>
                    <div class="detail-field"><div class="detail-label">Confidence</div><div class="detail-val" style="color:var(--${confClass})">${confScore}</div></div>
                    <div class="detail-field"><div class="detail-label">Risk</div><div class="detail-val" style="color:var(--danger)">${d.probability_risky}%</div></div>
                    <div class="detail-field"><div class="detail-label">Safe</div><div class="detail-val" style="color:var(--success)">${d.probability_safe}%</div></div>
                    <div class="detail-field"><div class="detail-label">Model Raw Risk</div><div class="detail-val">${d.model_probability_risky}%</div></div>
                </div>
            </div>`;

        // Credit breakdown in detail
        if (d.credit_breakdown && d.credit_breakdown.length > 0) {
            html += `<div class="detail-section"><h4>Credit Score Breakdown</h4><div class="credit-breakdown">`;
            d.credit_breakdown.forEach(item => {
                const dir = item.direction;
                const sign = item.points > 0 ? "+" : "";
                html += `<div class="breakdown-item">
                    <div class="breakdown-factor">${esc(item.factor)}</div>
                    <div class="breakdown-points ${dir}">${sign}${item.points}</div>
                </div>`;
            });
            html += `</div></div>`;
        }

        if (d.top_factors && d.top_factors.length > 0) {
            html += `<div class="detail-section"><h4>Why (Top Factors)</h4><div class="factors-list">`;
            d.top_factors.slice(0, 6).forEach((f, i) => {
                const w = Math.max(1, Math.min(100, Number(f.importance) || 0));
                html += `<div class="factor-item" style="padding:8px 0;">
                    <div class="factor-rank">${i + 1}</div>
                    <div class="factor-info">
                        <div class="factor-name">${esc(f.feature)}</div>
                        <div class="factor-value">Value: ${esc(f.value)}</div>
                    </div>
                    <div class="factor-bar-bg"><div class="factor-bar-fill" style="width:${w}%"></div></div>
                </div>`;
            });
            html += `</div></div>`;
        }

        html += `
            <div class="detail-section">
                <h4>Applicant Info</h4>
                ${d.applicant_name ? `<div class="detail-grid"><div class="detail-field"><div class="detail-label">Name</div><div class="detail-val">${esc(d.applicant_name)}</div></div><div class="detail-field"><div class="detail-label">Email</div><div class="detail-val">${esc(d.applicant_email)}</div></div></div>` : ""}
                <div class="detail-grid">
                    <div class="detail-field"><div class="detail-label">Gender</div><div class="detail-val">${gender}</div></div>
                    <div class="detail-field"><div class="detail-label">Age</div><div class="detail-val">${d.age} years</div></div>
                    <div class="detail-field"><div class="detail-label">Family Status</div><div class="detail-val">${FAMILY_STATUS_MAP[d.family_status] || d.family_status}</div></div>
                    <div class="detail-field"><div class="detail-label">Children</div><div class="detail-val">${d.children}</div></div>
                    <div class="detail-field"><div class="detail-label">Family Members</div><div class="detail-val">${d.family_members}</div></div>
                    <div class="detail-field"><div class="detail-label">Education</div><div class="detail-val">${EDUCATION_TYPES[d.education] || d.education}</div></div>
                </div>
            </div>
            <div class="detail-section">
                <h4>Employment & Assets</h4>
                <div class="detail-grid">
                    <div class="detail-field"><div class="detail-label">Income</div><div class="detail-val">$${Number(d.income).toLocaleString()}</div></div>
                    <div class="detail-field"><div class="detail-label">Income Type</div><div class="detail-val">${INCOME_TYPES[d.income_type] || d.income_type}</div></div>
                    <div class="detail-field"><div class="detail-label">Occupation</div><div class="detail-val">${OCCUPATION_TYPES[d.occupation] || d.occupation}</div></div>
                    <div class="detail-field"><div class="detail-label">Years Employed</div><div class="detail-val">${d.years_employed}</div></div>
                    <div class="detail-field"><div class="detail-label">Owns Car</div><div class="detail-val">${car}</div></div>
                    <div class="detail-field"><div class="detail-label">Owns Property</div><div class="detail-val">${realty}</div></div>
                    <div class="detail-field"><div class="detail-label">Housing</div><div class="detail-val">${HOUSING_TYPES[d.housing] || d.housing}</div></div>
                </div>
            </div>
            <div class="detail-section">
                <h4>Submitted</h4>
                <p style="font-size:0.88rem;">${new Date(d.created_at).toLocaleString()}</p>
            </div>
            <div style="margin-top:12px;">
                <button class="auth-btn primary small" onclick='prefillForm(${JSON.stringify(d)})'>Use as Template (Re-apply)</button>
            </div>
        `;

        // Staff actions
        if (isStaff()) {
            html += `
                <div class="detail-actions">
                    <h4 style="width:100%;margin:0 0 8px 0;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);">Staff Actions</h4>
                    <select id="override-status">
                        <option value="manually_approved" ${d.status === "manually_approved" ? "selected" : ""}>Manually Approved</option>
                        <option value="manually_rejected" ${d.status === "manually_rejected" ? "selected" : ""}>Manually Rejected</option>
                        <option value="under_review" ${d.status === "under_review" ? "selected" : ""}>Under Review</option>
                    </select>
                    <textarea id="override-note" placeholder="Override note (required)..." required></textarea>
                    <button class="auth-btn success small" onclick="doOverride('${d.id}')">Save Override</button>
                    <button class="auth-btn ${d.flagged ? "small" : "danger small"}" onclick="doFlag('${d.id}', ${!d.flagged})">${d.flagged ? "Remove Flag" : "Flag Application"}</button>
                </div>
            `;
        }

        body.innerHTML = html;
        document.getElementById("detail-overlay").classList.add("open");
        document.getElementById("detail-panel").classList.add("open");
    } catch (e) { toast("Error loading detail", "error"); }
}

function closeDetail() {
    document.getElementById("detail-overlay").classList.remove("open");
    document.getElementById("detail-panel").classList.remove("open");
}

async function doOverride(appId) {
    const new_status = document.getElementById("override-status").value;
    const note = document.getElementById("override-note").value.trim();
    if (!note) { toast("Override note is required", "error"); return; }
    try {
        const resp = await fetch(`/admin/applications/${appId}/override`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ new_status, note }),
        });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Override failed"); }
        toast("Decision overridden", "success");
        closeDetail();
        loadBankQueue();
    } catch (e) { toast(e.message, "error"); }
}

async function doFlag(appId, flagged) {
    const reason = flagged ? prompt("Flag reason (optional):") || "" : "";
    try {
        const resp = await fetch(`/admin/applications/${appId}/flag`, {
            method: "POST", headers: authHeaders(),
            body: JSON.stringify({ flagged, reason }),
        });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Flag failed"); }
        toast(flagged ? "Application flagged" : "Flag removed", "success");
        closeDetail();
        loadBankQueue();
    } catch (e) { toast(e.message, "error"); }
}

// ── Bank Queue (staff) ───────────────────────────────────────
async function loadBankQueue() {
    const wrap = document.getElementById("bank-table-wrap");
    const msg = document.getElementById("bank-queue-msg");
    if (!wrap || !isStaff() || !authToken) return;
    msg.textContent = "";

    const params = new URLSearchParams();
    const search = document.getElementById("bank-search")?.value;
    const decision = document.getElementById("bank-filter-decision")?.value;
    const flagged = document.getElementById("bank-filter-flagged")?.value;
    const lowConf = document.getElementById("bank-filter-confidence")?.value;
    if (search) params.set("search", search);
    if (decision) params.set("decision", decision);
    if (flagged) params.set("flagged", flagged);
    if (lowConf) params.set("low_confidence", lowConf);

    try {
        const resp = await fetch(`/admin/applications?${params}`, { headers: authHeaders() });
        if (!resp.ok) { msg.textContent = "Could not load applications."; return; }
        const apps = await resp.json();
        if (apps.length === 0) {
            wrap.innerHTML = '<div class="empty-state"><p>No applications match your filters.</p></div>';
            return;
        }
        let html = `<table class="data-table"><thead><tr>
            <th>Applicant</th><th>Status</th><th>Score</th><th>Risk</th><th>Confidence</th><th>Income</th><th>Age</th><th>Flag</th><th>Date</th>
        </tr></thead><tbody>`;
        apps.forEach(a => {
            const date = new Date(a.created_at).toLocaleDateString();
            const isLowConf = a.confidence_score != null && a.confidence_score < 65;
            html += `<tr onclick="openAppDetail('${a.id}')" class="${isLowConf ? "low-confidence" : ""}">
                <td><strong>${esc(a.applicant_name)}</strong><br><span class="history-meta">${esc(a.applicant_email)}</span></td>
                <td>${statusPill(a.status)}</td>
                <td>${a.credit_score}</td>
                <td>${a.probability_risky}%</td>
                <td>${a.confidence_score != null ? a.confidence_score.toFixed(0) + "%" : "—"} ${isLowConf ? '<span class="pill low-conf">LOW</span>' : ""}</td>
                <td>$${Number(a.income).toLocaleString()}</td>
                <td>${a.age}</td>
                <td>${a.flagged ? '<span class="pill flag">Flagged</span>' : "—"}</td>
                <td>${date}</td>
            </tr>`;
        });
        html += "</tbody></table>";
        wrap.innerHTML = html;
    } catch (e) { msg.textContent = "Error loading queue."; }
}

// ── Export CSV ────────────────────────────────────────────────
function exportCSV() {
    if (!authToken) return;
    fetch("/admin/export", { headers: authHeaders() })
        .then(r => r.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "applications_export.csv";
            a.click();
            URL.revokeObjectURL(url);
            toast("CSV exported", "success");
        })
        .catch(() => toast("Export failed", "error"));
}

// ── Fairness / Bias (staff) ─────────────────────────────────
async function loadFairness() {
    const container = document.getElementById("fairness-container");
    const meta = document.getElementById("fairness-meta");
    if (!container || !meta || !isStaff() || !authToken) return;
    container.innerHTML = '<div class="empty-state mini"><p>Loading fairness metrics…</p></div>';
    meta.textContent = "";
    try {
        const resp = await fetch("/admin/fairness", { headers: authHeaders() });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Could not load fairness metrics");

        meta.textContent = `Overall approval rate: ${data.overall_approval_rate}% • Applications analyzed: ${data.total_applications} • Generated: ${new Date(data.generated_at).toLocaleString()}`;

        const card = (m) => {
            const groups = Array.isArray(m.groups) ? m.groups : [];
            return `<div class="fairness-card">
                <h3>${esc(m.label)}</h3>
                <div class="sub">Overall: ${m.overall_approval_rate}%</div>
                <div class="bar-list">
                    ${groups.map(g => {
                        const w = Math.max(0, Math.min(100, Number(g.approval_rate) || 0));
                        const d = Number(g.delta_vs_overall) || 0;
                        const deltaCls = d > 0.05 ? "pos" : d < -0.05 ? "neg" : "";
                        const deltaText = d === 0 ? "" : `${d > 0 ? "+" : ""}${d.toFixed(1)}%`;
                        const n = (g.approved || 0) + (g.rejected || 0);
                        return `<div class="bar-row">
                            <div class="bar-label">${esc(g.label)}</div>
                            <div class="bar-track" title="Approval rate">
                                <div class="bar-fill" style="width:${w}%"></div>
                            </div>
                            <div class="bar-metric">${w.toFixed(1)}% (n=${n})${deltaText ? ` <span class="bar-delta ${deltaCls}">${esc(deltaText)}</span>` : ""}</div>
                        </div>`;
                    }).join("")}
                </div>
            </div>`;
        };

        container.innerHTML = (data.metrics || []).map(card).join("") || '<div class="empty-state"><p>No metrics available.</p></div>';
    } catch (e) {
        container.innerHTML = `<div class="empty-state"><p>${esc(e.message || "Could not load.")}</p></div>`;
    }
}

// ── Users (admin) ────────────────────────────────────────────
async function loadUsers() {
    const wrap = document.getElementById("users-table-wrap");
    if (!wrap || !authToken) return;
    try {
        const resp = await fetch("/admin/users", { headers: authHeaders() });
        if (!resp.ok) return;
        const users = await resp.json();
        if (users.length === 0) {
            wrap.innerHTML = '<div class="empty-state"><p>No users found.</p></div>';
            return;
        }
        let html = `<table class="data-table"><thead><tr class="no-hover">
            <th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Joined</th>${isAdmin() ? "<th>Actions</th>" : ""}
        </tr></thead><tbody>`;
        users.forEach(u => {
            const date = new Date(u.created_at).toLocaleDateString();
            const actions = isAdmin() ? `<td>
                <select onchange="changeRole('${u.id}', this.value)" style="padding:4px 8px;border-radius:6px;border:1px solid var(--border);font-size:0.8rem;">
                    <option value="customer" ${u.role === "customer" ? "selected" : ""}>Customer</option>
                    <option value="officer" ${u.role === "officer" ? "selected" : ""}>Officer</option>
                    <option value="admin" ${u.role === "admin" ? "selected" : ""}>Admin</option>
                </select>
                <button class="auth-btn small ${u.is_active ? "danger" : "success"}" style="margin-left:6px;" onclick="toggleActive('${u.id}', ${!u.is_active})">${u.is_active ? "Deactivate" : "Activate"}</button>
            </td>` : "";
            html += `<tr class="no-hover">
                <td><strong>${esc(u.full_name)}</strong></td>
                <td>${esc(u.email)}</td>
                <td>${rolePill(u.role)}</td>
                <td><span class="pill ${u.is_active ? "active" : "inactive"}">${u.is_active ? "Active" : "Inactive"}</span></td>
                <td>${date}</td>
                ${actions}
            </tr>`;
        });
        html += "</tbody></table>";
        wrap.innerHTML = html;
    } catch (e) { wrap.innerHTML = '<p class="empty-state">Could not load users.</p>'; }
}

async function changeRole(userId, role) {
    try {
        const resp = await fetch(`/admin/users/${userId}/role`, { method: "PUT", headers: authHeaders(), body: JSON.stringify({ role }) });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Failed"); }
        toast("Role updated", "success");
        loadUsers();
    } catch (e) { toast(e.message, "error"); }
}

async function toggleActive(userId, isActive) {
    try {
        const resp = await fetch(`/admin/users/${userId}/active`, { method: "PUT", headers: authHeaders(), body: JSON.stringify({ is_active: isActive }) });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Failed"); }
        toast(isActive ? "User activated" : "User deactivated", "success");
        loadUsers();
    } catch (e) { toast(e.message, "error"); }
}

// ── Audit Log ────────────────────────────────────────────────
async function loadAuditLog() {
    const wrap = document.getElementById("audit-table-wrap");
    if (!wrap || !isStaff() || !authToken) return;
    try {
        const resp = await fetch("/admin/audit", { headers: authHeaders() });
        if (!resp.ok) return;
        const logs = await resp.json();
        if (logs.length === 0) {
            wrap.innerHTML = '<div class="empty-state"><p>No audit entries yet. Overrides and flags will appear here.</p></div>';
            return;
        }
        let html = `<table class="data-table"><thead><tr class="no-hover">
            <th>Action</th><th>Staff</th><th>Old Status</th><th>New Status</th><th>Note</th><th>Date</th>
        </tr></thead><tbody>`;
        logs.forEach(l => {
            const date = new Date(l.created_at).toLocaleString();
            html += `<tr class="no-hover">
                <td><strong>${esc(l.action)}</strong></td>
                <td>${esc(l.staff_name)}<br><span class="history-meta">${esc(l.staff_email)}</span></td>
                <td>${l.old_status ? statusPill(l.old_status) : "—"}</td>
                <td>${l.new_status ? statusPill(l.new_status) : "—"}</td>
                <td style="max-width:200px;word-break:break-word;">${esc(l.note || "—")}</td>
                <td class="history-meta">${date}</td>
            </tr>`;
        });
        html += "</tbody></table>";
        wrap.innerHTML = html;
    } catch (e) { wrap.innerHTML = '<p class="empty-state">Could not load audit log.</p>'; }
}

// ── Profile ──────────────────────────────────────────────────
function loadProfile() {
    if (!authUser) return;
    document.getElementById("profile-email").textContent = authUser.email;
    document.getElementById("profile-role").textContent = authUser.role.charAt(0).toUpperCase() + authUser.role.slice(1);
    document.getElementById("profile-name").value = authUser.full_name;
    fetch("/auth/me", { headers: authHeaders() }).then(r => r.json()).then(d => {
        if (d.created_at) document.getElementById("profile-since").textContent = new Date(d.created_at).toLocaleDateString();
    }).catch(() => {});
}

async function saveName() {
    const full_name = document.getElementById("profile-name").value.trim();
    if (!full_name) { toast("Name cannot be empty", "error"); return; }
    try {
        const resp = await fetch("/auth/profile", { method: "PUT", headers: authHeaders(), body: JSON.stringify({ full_name }) });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Failed"); }
        authUser.full_name = full_name;
        localStorage.setItem("authUser", JSON.stringify(authUser));
        updateAuthUI();
        toast("Name updated", "success");
    } catch (e) { toast(e.message, "error"); }
}

async function changePassword() {
    const current = document.getElementById("pw-current").value;
    const newPw = document.getElementById("pw-new").value;
    if (!current || !newPw) { toast("Fill in both fields", "error"); return; }
    if (newPw.length < 6) { toast("New password must be at least 6 characters", "error"); return; }
    try {
        const resp = await fetch("/auth/password", { method: "PUT", headers: authHeaders(), body: JSON.stringify({ current_password: current, new_password: newPw }) });
        if (!resp.ok) { const d = await resp.json(); throw new Error(d.error || "Failed"); }
        document.getElementById("pw-current").value = "";
        document.getElementById("pw-new").value = "";
        toast("Password updated", "success");
    } catch (e) { toast(e.message, "error"); }
}

// ── Drafts ───────────────────────────────────────────────────
async function loadDraftOptions() {
    if (!authToken) return;
    try {
        const resp = await fetch("/drafts", { headers: authHeaders() });
        if (!resp.ok) return;
        const drafts = await resp.json();
        const sel = document.getElementById("draft-select");
        const currentVal = sel.value;
        sel.innerHTML = '<option value="">Load a draft...</option>';
        drafts.forEach(d => {
            sel.innerHTML += `<option value="${d.id}">${esc(d.name)} (${new Date(d.updated_at).toLocaleDateString()})</option>`;
        });
        if (currentVal) sel.value = currentVal;
    } catch (e) {}
}

function loadSelectedDraft() {
    const sel = document.getElementById("draft-select");
    const draftId = sel.value;
    if (!draftId) { currentDraftId = null; document.getElementById("delete-draft-btn").style.display = "none"; return; }
    currentDraftId = draftId;
    document.getElementById("delete-draft-btn").style.display = "";

    fetch(`/drafts`, { headers: authHeaders() })
        .then(r => r.json())
        .then(drafts => {
            const draft = drafts.find(d => d.id === draftId);
            if (!draft) return;
            const fd = draft.form_data;
            // Apply form data
            const toggles = { gender: fd.gender, own_car: fd.own_car, own_realty: fd.own_realty, work_phone: fd.work_phone, phone: fd.phone, email: fd.email };
            Object.entries(toggles).forEach(([field, val]) => {
                if (val == null) return;
                const group = document.querySelector(`#view-apply .toggle-group[data-field="${field}"]`);
                if (!group) return;
                group.querySelectorAll(".toggle-btn").forEach(btn => {
                    btn.classList.toggle("active", btn.dataset.value === String(val));
                });
            });
            const inputs = { age: fd.age, children: fd.children, family_members: fd.family_members, income: fd.income, years_employed: fd.years_employed };
            Object.entries(inputs).forEach(([id, val]) => { if (val != null) { const el = document.getElementById(id); if (el) el.value = val; } });
            const selects = { family_status: fd.family_status, education: fd.education, income_type: fd.income_type, occupation: fd.occupation, housing: fd.housing };
            Object.entries(selects).forEach(([id, val]) => { if (val != null) { const el = document.getElementById(id); if (el) el.value = val; } });
            toast("Draft loaded", "info");
        })
        .catch(() => toast("Could not load draft", "error"));
}

function getFormData() {
    const toggleValues = {};
    document.querySelectorAll("#view-apply .toggle-group").forEach(group => {
        const field = group.dataset.field;
        const active = group.querySelector(".toggle-btn.active");
        if (active) toggleValues[field] = active.dataset.value;
    });
    return {
        gender: toggleValues.gender, own_car: toggleValues.own_car, own_realty: toggleValues.own_realty,
        work_phone: toggleValues.work_phone, phone: toggleValues.phone, email: toggleValues.email,
        age: document.getElementById("age").value, family_status: document.getElementById("family_status").value,
        children: document.getElementById("children").value, family_members: document.getElementById("family_members").value,
        education: document.getElementById("education").value, income_type: document.getElementById("income_type").value,
        income: document.getElementById("income").value, occupation: document.getElementById("occupation").value,
        years_employed: document.getElementById("years_employed").value, housing: document.getElementById("housing").value,
    };
}

async function saveDraft() {
    if (!authToken) { toast("Log in to save drafts", "error"); return; }
    const form_data = getFormData();

    try {
        if (currentDraftId) {
            let name = "Untitled Draft";
            try {
                const list = await (await fetch("/drafts", { headers: authHeaders() })).json();
                const found = list.find((x) => x.id === currentDraftId);
                if (found) name = found.name;
            } catch {
                /* keep default name */
            }
            const resp = await fetch(`/drafts/${currentDraftId}`, {
                method: "PUT", headers: authHeaders(),
                body: JSON.stringify({ form_data, name }),
            });
            if (!resp.ok) throw new Error("Save failed");
            toast("Draft updated", "success");
        } else {
            const name = prompt("Draft name:", "My Draft") || "Untitled Draft";
            const resp = await fetch("/drafts", {
                method: "POST", headers: authHeaders(),
                body: JSON.stringify({ form_data, name }),
            });
            if (!resp.ok) throw new Error("Save failed");
            const data = await resp.json();
            currentDraftId = data.id;
            document.getElementById("delete-draft-btn").style.display = "";
            toast("Draft saved", "success");
        }
        loadDraftOptions();
    } catch (e) { toast(e.message, "error"); }
}

async function deleteCurrentDraft() {
    if (!currentDraftId || !authToken) return;
    if (!confirm("Delete this draft?")) return;
    try {
        await fetch(`/drafts/${currentDraftId}`, { method: "DELETE", headers: authHeaders() });
        currentDraftId = null;
        document.getElementById("delete-draft-btn").style.display = "none";
        document.getElementById("draft-select").value = "";
        loadDraftOptions();
        toast("Draft deleted", "success");
    } catch (e) { toast("Delete failed", "error"); }
}

// Drafts list view
async function loadDraftsList() {
    const container = document.getElementById("drafts-list");
    if (!authToken || !container) return;
    try {
        const resp = await fetch("/drafts", { headers: authHeaders() });
        if (!resp.ok) return;
        const drafts = await resp.json();
        if (drafts.length === 0) {
            container.innerHTML = '<div class="empty-state"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>No saved drafts. Use the <strong>Save Draft</strong> button on the Apply tab.</p></div>';
            return;
        }
        container.innerHTML = drafts.map(d => `
            <div class="draft-card" onclick="loadDraftAndSwitch('${d.id}')">
                <div class="draft-card-info">
                    <div class="draft-card-name">${esc(d.name)}</div>
                    <div class="draft-card-meta">Last updated: ${new Date(d.updated_at).toLocaleString()}</div>
                </div>
                <div class="draft-card-actions">
                    <button class="auth-btn small danger" onclick="event.stopPropagation(); deleteDraftById('${d.id}')">Delete</button>
                </div>
            </div>
        `).join("");
    } catch (e) { container.innerHTML = '<p class="empty-state">Could not load drafts.</p>'; }
}

function loadDraftAndSwitch(draftId) {
    currentDraftId = draftId;
    document.getElementById("draft-select").value = draftId;
    loadSelectedDraft();
    switchView("apply");
}

async function deleteDraftById(id) {
    if (!confirm("Delete this draft?")) return;
    try {
        await fetch(`/drafts/${id}`, { method: "DELETE", headers: authHeaders() });
        if (currentDraftId === id) { currentDraftId = null; document.getElementById("draft-select").value = ""; }
        loadDraftsList();
        loadDraftOptions();
        toast("Draft deleted", "success");
    } catch (e) { toast("Delete failed", "error"); }
}

function setToggleGroupValue(groupSelector, value) {
    const group = document.querySelector(groupSelector);
    if (!group) return;
    group.querySelectorAll(".toggle-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.value === String(value));
    });
}

// ── What-If Simulator ────────────────────────────────────────
let whatIfTimeout = null;

async function loadWhatIfAppOptions() {
    const sel = document.getElementById("wi-load-app-select");
    if (!sel || !authToken) return;
    if (sel.dataset.loaded === "1") return;
    sel.dataset.loaded = "1";
    try {
        const resp = await fetch("/applications", { headers: authHeaders() });
        const apps = resp.ok ? await resp.json() : [];
        if (!Array.isArray(apps) || apps.length === 0) {
            sel.innerHTML = '<option value="">No previous applications</option>';
            return;
        }
        sel.innerHTML = '<option value="">Select an application…</option>' + apps.slice(0, 50).map(a => {
            const dt = new Date(a.created_at).toLocaleString();
            const label = `${dt} • Score ${a.credit_score} • ${a.status.replace(/_/g, " ")}`;
            return `<option value="${esc(a.id)}">${esc(label)}</option>`;
        }).join("");
    } catch {
        sel.innerHTML = '<option value="">Could not load</option>';
    }
}

async function loadWhatIfFromSelected() {
    const sel = document.getElementById("wi-load-app-select");
    const appId = sel?.value;
    if (!appId) { toast("Select an application to load", "error"); return; }
    try {
        const resp = await fetch(`/applications/${appId}`, { headers: authHeaders() });
        if (!resp.ok) throw new Error("Could not load application");
        const d = await resp.json();
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_gender"]', d.gender);
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_own_car"]', d.own_car);
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_own_realty"]', d.own_realty);
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_work_phone"]', d.work_phone);
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_phone"]', d.phone);
        setToggleGroupValue('#view-whatif .toggle-group[data-field="wi_email"]', d.email_flag);

        const setVal = (id, val) => { const el = document.getElementById(id); if (el != null) el.value = val; };
        setVal("wi_age", d.age);
        setVal("wi_family_status", d.family_status);
        setVal("wi_children", d.children);
        setVal("wi_family_members", d.family_members);
        setVal("wi_education", d.education);
        setVal("wi_income_type", d.income_type);
        setVal("wi_income", d.income);
        setVal("wi_occupation", d.occupation);
        setVal("wi_years_employed", d.years_employed);
        setVal("wi_housing", d.housing);

        toast("Loaded into What-If simulator", "success");
        triggerWhatIf();
    } catch (e) {
        toast(e.message || "Load failed", "error");
    }
}

function getWhatIfPayload() {
    const getToggle = (field) => {
        const group = document.querySelector(`.toggle-group[data-field="${field}"]`);
        if (!group) return 0;
        const active = group.querySelector(".toggle-btn.active");
        return active ? parseInt(active.dataset.value) : 0;
    };
    return {
        gender: getToggle("wi_gender"),
        own_car: getToggle("wi_own_car"),
        own_realty: getToggle("wi_own_realty"),
        work_phone: getToggle("wi_work_phone"),
        phone: getToggle("wi_phone"),
        email: getToggle("wi_email"),
        age: parseFloat(document.getElementById("wi_age")?.value) || 35,
        family_status: parseInt(document.getElementById("wi_family_status")?.value) || 0,
        children: parseInt(document.getElementById("wi_children")?.value) || 0,
        family_members: parseInt(document.getElementById("wi_family_members")?.value) || 1,
        education: parseInt(document.getElementById("wi_education")?.value) || 0,
        income_type: parseInt(document.getElementById("wi_income_type")?.value) || 0,
        income: parseFloat(document.getElementById("wi_income")?.value) || 50000,
        occupation: parseInt(document.getElementById("wi_occupation")?.value) || 0,
        years_employed: parseFloat(document.getElementById("wi_years_employed")?.value) || 1,
        housing: parseInt(document.getElementById("wi_housing")?.value) || 0,
    };
}

function triggerWhatIf() {
    if (whatIfTimeout) clearTimeout(whatIfTimeout);
    const panel = document.getElementById("whatif-results");
    if (panel) panel.classList.add("whatif-loading");
    whatIfTimeout = setTimeout(async () => {
        try {
            const payload = getWhatIfPayload();
            const resp = await fetch("/whatif", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) return;
            const data = await resp.json();
            displayWhatIfResults(data);
        } catch (e) {
            if (isFetchNetworkError(e)) toast(backendUnreachableHint(), "error");
        }
        if (panel) panel.classList.remove("whatif-loading");
    }, 300);
}

function displayWhatIfResults(data) {
    // Decision banner
    const banner = document.getElementById("wi-decision-banner");
    const icon = document.getElementById("wi-decision-icon");
    const text = document.getElementById("wi-decision-text");
    const strength = document.getElementById("wi-decision-strength");
    banner.className = "decision-banner";
    if (data.decision === "Approved") { banner.classList.add("approved"); icon.textContent = "\u2713"; }
    else if (data.probability_risky > 40 && data.probability_risky < 60) { banner.classList.add("borderline"); icon.textContent = "\u2014"; }
    else { banner.classList.add("rejected"); icon.textContent = "\u2717"; }
    text.textContent = data.decision;
    strength.textContent = data.strength_description;

    // Confidence UI removed (keep data in the API, just don't render it)

    // Credit score
    const scoreNum = document.getElementById("wi-credit-score-number");
    const scoreRing = document.getElementById("wi-credit-score-ring");
    scoreNum.textContent = data.credit_score;
    scoreRing.className = "credit-score-ring " + creditScoreClass(data.credit_score);

    // Risk gauge
    document.getElementById("wi-gauge-marker").style.left = data.probability_risky + "%";
    document.getElementById("wi-prob-safe").textContent = data.probability_safe + "%";
    document.getElementById("wi-prob-risky").textContent = data.probability_risky + "%";

    // Credit breakdown
    const breakdownEl = document.getElementById("wi-credit-breakdown");
    if (breakdownEl && data.credit_breakdown && data.credit_breakdown.length > 0) {
        const maxPts = Math.max(...data.credit_breakdown.map(b => Math.abs(b.points)), 1);
        breakdownEl.innerHTML = data.credit_breakdown.map(b => {
            const pct = Math.min(Math.abs(b.points) / maxPts * 50, 50);
            const sign = b.points > 0 ? "+" : "";
            const dir = b.direction;
            const barStyle = dir === "positive"
                ? `right:50%;width:${pct}%`
                : dir === "negative"
                ? `left:50%;width:${pct}%`
                : `left:50%;width:0%`;
            return `<div class="breakdown-item">
                <div class="breakdown-factor">${esc(b.factor)}</div>
                <div class="breakdown-bar-wrap">
                    <div class="breakdown-bar-center"></div>
                    <div class="breakdown-bar ${dir}" style="${barStyle}"></div>
                </div>
                <div class="breakdown-points ${dir}">${sign}${b.points}</div>
            </div>`;
        }).join("");
    }

    // Top factors
    const factorsEl = document.getElementById("wi-factors-list");
    if (factorsEl && data.top_factors && data.top_factors.length > 0) {
        factorsEl.innerHTML = "";
        data.top_factors.forEach((f, i) => {
            factorsEl.innerHTML += `<div class="factor-item"><div class="factor-rank">${i + 1}</div><div class="factor-info"><div class="factor-name">${esc(f.feature)}</div><div class="factor-value">Your value: ${esc(f.value)}</div></div><div class="factor-bar-bg"><div class="factor-bar-fill" style="width:${f.importance}%"></div></div></div>`;
        });
    }

    // Risk tips
    const tipsEl = document.getElementById("wi-tips-section");
    const tipsListEl = document.getElementById("wi-tips-list");
    if (tipsEl && tipsListEl) {
        if (data.risk_tips && data.risk_tips.length > 0 && data.decision === "Rejected") {
            tipsEl.style.display = "block";
            tipsListEl.innerHTML = data.risk_tips.map(t => `<div class="tip-item"><span class="tip-icon">*</span><span>${esc(t)}</span></div>`).join("");
        } else {
            tipsEl.style.display = "none";
        }
    }
}

function creditScoreClass(score) {
    if (score >= 750) return "excellent";
    if (score >= 650) return "good";
    if (score >= 550) return "fair";
    return "poor";
}

// ── Display Results (Apply tab) ──────────────────────────────
function displayResults(data) {
    const placeholder = document.getElementById("results-placeholder");
    const content = document.getElementById("results-content");
    placeholder.style.display = "none";
    content.style.display = "block";
    content.style.animation = "none";
    content.offsetHeight;
    content.style.animation = "";

    // Decision banner
    const banner = document.getElementById("decision-banner");
    const icon = document.getElementById("decision-icon");
    const text = document.getElementById("decision-text");
    const strength = document.getElementById("decision-strength");
    banner.className = "decision-banner";
    if (data.decision === "Approved") { banner.classList.add("approved"); icon.textContent = "\u2713"; }
    else if (data.probability_risky > 40 && data.probability_risky < 60) { banner.classList.add("borderline"); icon.textContent = "\u2014"; }
    else { banner.classList.add("rejected"); icon.textContent = "\u2717"; }
    text.textContent = data.decision;
    strength.textContent = data.strength.description;

    // Confidence UI removed (keep data in the API, just don't render it)

    // Credit score
    const scoreNum = document.getElementById("credit-score-number");
    const scoreRing = document.getElementById("credit-score-ring");
    scoreNum.textContent = data.credit_score;
    scoreRing.className = "credit-score-ring " + creditScoreClass(data.credit_score);

    // Credit breakdown
    const breakdownEl = document.getElementById("credit-breakdown");
    if (data.credit_breakdown && data.credit_breakdown.length > 0) {
        const maxPts = Math.max(...data.credit_breakdown.map(b => Math.abs(b.points)), 1);
        breakdownEl.innerHTML = data.credit_breakdown.map(b => {
            const pct = Math.min(Math.abs(b.points) / maxPts * 50, 50);
            const sign = b.points > 0 ? "+" : "";
            const dir = b.direction;
            const barStyle = dir === "positive"
                ? `right:50%;width:${pct}%`
                : dir === "negative"
                ? `left:50%;width:${pct}%`
                : `left:50%;width:0%`;
            return `<div class="breakdown-item">
                <div class="breakdown-factor">${esc(b.factor)}</div>
                <div class="breakdown-bar-wrap">
                    <div class="breakdown-bar-center"></div>
                    <div class="breakdown-bar ${dir}" style="${barStyle}"></div>
                </div>
                <div class="breakdown-points ${dir}">${sign}${b.points}</div>
            </div>`;
        }).join("");
    } else {
        breakdownEl.innerHTML = "";
    }

    // Risk gauge
    document.getElementById("gauge-marker").style.left = data.probability_risky + "%";
    document.getElementById("prob-safe").textContent = data.probability_safe + "%";
    document.getElementById("prob-risky").textContent = data.probability_risky + "%";

    // Top factors
    const list = document.getElementById("factors-list");
    list.innerHTML = "";
    data.top_factors.forEach((f, i) => {
        const item = document.createElement("div");
        item.className = "factor-item";
        item.innerHTML = `<div class="factor-rank">${i + 1}</div><div class="factor-info"><div class="factor-name">${esc(f.feature)}</div><div class="factor-value">Your value: ${esc(f.value)}</div></div><div class="factor-bar-bg"><div class="factor-bar-fill" style="width:${f.importance}%"></div></div>`;
        list.appendChild(item);
    });

    // Risk tips
    showRiskTips(data);

    if (window.innerWidth <= 900) document.getElementById("results-panel").scrollIntoView({ behavior: "smooth" });
}

function showRiskTips(data) {
    const section = document.getElementById("tips-section");
    const list = document.getElementById("tips-list");

    if (data.risk_tips && data.risk_tips.length > 0) {
        section.style.display = "block";
        list.innerHTML = data.risk_tips.map(t => `<div class="tip-item"><span class="tip-icon">*</span><span>${esc(t)}</span></div>`).join("");
    } else if (data.decision === "Rejected") {
        section.style.display = "block";
        const tips = [
            "A higher annual income significantly improves approval odds.",
            "Longer employment history (5+ years) reduces risk scores.",
            "Owning property or a vehicle adds stability to your profile.",
        ];
        list.innerHTML = tips.map(t => `<div class="tip-item"><span class="tip-icon">*</span><span>${esc(t)}</span></div>`).join("");
    } else {
        section.style.display = "none";
    }
}

// ── Main app (form handling + init) ─────────────────────────

/** Bind functions used from HTML onclick / onchange so handlers always resolve. */
function exposeGlobals() {
    window.switchView = switchView;
    window.openModal = openModal;
    window.closeModal = closeModal;
    window.switchModal = switchModal;
    window.handleLogin = handleLogin;
    window.handleRegister = handleRegister;
    window.logout = logout;
    window.openAppDetail = openAppDetail;
    window.closeDetail = closeDetail;
    window.toggleNotifDropdown = toggleNotifDropdown;
    window.markAllRead = markAllRead;
    window.exportCSV = exportCSV;
    window.loadBankQueue = loadBankQueue;
    window.loadFairness = loadFairness;
    window.prefillForm = prefillForm;
    window.loadWhatIfFromSelected = loadWhatIfFromSelected;
    window.doOverride = doOverride;
    window.doFlag = doFlag;
    window.changeRole = changeRole;
    window.toggleActive = toggleActive;
    window.saveName = saveName;
    window.changePassword = changePassword;
    window.saveDraft = saveDraft;
    window.deleteCurrentDraft = deleteCurrentDraft;
    window.loadSelectedDraft = loadSelectedDraft;
    window.deleteDraftById = deleteDraftById;
}

/** Run as soon as this file loads — before DOMContentLoaded — so inline onclick always resolves. */
exposeGlobals();

/** Delegated clicks: works even if a browser/extension interferes with inline handlers. */
function bindAuthOpenDelegation() {
    document.body.addEventListener("click", (e) => {
        const btn = e.target.closest("[data-open-modal]");
        if (!btn) return;
        const mode = btn.getAttribute("data-open-modal");
        if (!mode) return;
        const audience = btn.getAttribute("data-auth-audience");
        openModal(mode, audience || undefined);
    });
}

function bindNavDelegation() {
    document.body.addEventListener("click", (e) => {
        const btn = e.target.closest(".side-nav-btn, .nav-tab");
        if (!btn) return;
        const view = btn.getAttribute("data-view");
        if (!view) return;
        switchView(view);
    });
}

document.addEventListener("DOMContentLoaded", () => {
    if (window.location && window.location.protocol === "file:") {
        toast("This app must be opened via the backend server (not file://). Run: python3 backend/asgi.py then open http://127.0.0.1:8000.", "error");
    }
    bindAuthOpenDelegation();
    bindNavDelegation();

    // Register role toggle (Customer vs Admin) + enrollment code field
    const roleToggle = document.getElementById("register-role-toggle");
    const codeWrap = document.getElementById("register-admin-code-wrap");
    if (roleToggle) {
        roleToggle.querySelectorAll(".toggle-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                roleToggle.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const isAdminRole = btn.dataset.value === "admin";
                if (codeWrap) codeWrap.style.display = isAdminRole ? "block" : "none";
            });
        });
    }

    const authModal = document.getElementById("auth-modal");
    if (authModal) {
        authModal.addEventListener("click", (e) => {
            if (e.target === authModal) closeModal();
        });
    }

    const form = document.getElementById("approval-form");

    // Toggle button groups (Apply tab)
    document.querySelectorAll("#view-apply .toggle-group").forEach(group => {
        group.querySelectorAll(".toggle-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });
    });

    // What-if toggle groups
    document.querySelectorAll("#view-whatif .toggle-group").forEach(group => {
        group.querySelectorAll(".toggle-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                triggerWhatIf();
            });
        });
    });

    // What-if input change listeners
    document.querySelectorAll("#view-whatif input.whatif-input, #view-whatif select.whatif-input").forEach(el => {
        el.addEventListener("input", triggerWhatIf);
        el.addEventListener("change", triggerWhatIf);
    });

    // Auto-sync children -> family members
    const childrenInput = document.getElementById("children");
    const familyInput = document.getElementById("family_members");
    if (childrenInput && familyInput) {
        childrenInput.addEventListener("input", () => {
            const kids = parseInt(childrenInput.value) || 0;
            const current = parseInt(familyInput.value) || 1;
            if (current < kids + 1) familyInput.value = kids + 1;
        });
    }

    // Form submission
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const toggleValues = {};
        let missingToggle = false;
        document.querySelectorAll("#view-apply .toggle-group").forEach(group => {
            const field = group.dataset.field;
            const active = group.querySelector(".toggle-btn.active");
            if (active) {
                toggleValues[field] = active.dataset.value;
                group.closest(".form-group").classList.remove("error");
            } else {
                group.closest(".form-group").classList.add("error");
                missingToggle = true;
            }
        });
        if (missingToggle) { showError("Please select an option for all toggle buttons."); return; }

        const requiredSelects = form.querySelectorAll("select[required]");
        let missingSelect = false;
        requiredSelects.forEach(sel => {
            if (!sel.value) { sel.closest(".form-group").classList.add("error"); missingSelect = true; }
            else { sel.closest(".form-group").classList.remove("error"); }
        });
        if (missingSelect) { showError("Please fill in all required fields."); return; }

        hideError();

        const payload = {
            gender: toggleValues.gender, own_car: toggleValues.own_car, own_realty: toggleValues.own_realty,
            work_phone: toggleValues.work_phone, phone: toggleValues.phone, email: toggleValues.email,
            age: document.getElementById("age").value, family_status: document.getElementById("family_status").value,
            children: document.getElementById("children").value, family_members: document.getElementById("family_members").value,
            education: document.getElementById("education").value, income_type: document.getElementById("income_type").value,
            income: document.getElementById("income").value, occupation: document.getElementById("occupation").value,
            years_employed: document.getElementById("years_employed").value, housing: document.getElementById("housing").value,
        };

        const btn = document.getElementById("submit-btn");
        btn.classList.add("loading");
        btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Evaluating...';

        try {
            const resp = await fetch("/predict", { method: "POST", headers: authHeaders(), body: JSON.stringify(payload) });
            if (!resp.ok) {
                const raw = await resp.text();
                let message = `Server error (HTTP ${resp.status})`;
                try {
                    const parsed = JSON.parse(raw);
                    message = parsed.error || parsed.detail || message;
                } catch {
                    const trimmed = (raw || "").replace(/\s+/g, " ").trim();
                    if (trimmed && !trimmed.startsWith("<!DOCTYPE") && !trimmed.startsWith("<html")) {
                        message = trimmed;
                    }
                }
                throw new Error(message);
            }
            const result = await resp.json();
            displayResults(result);
            if (authToken) {
                toast("Application saved to your history", "success");
                pollNotifCount();
            }
        } catch (err) {
            if (isFetchNetworkError(err)) {
                showError(backendUnreachableHint());
            } else {
                showError("Error: " + err.message);
            }
        } finally {
            btn.classList.remove("loading");
            btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Evaluate Application';
        }
    });

    function showError(msg) {
        let el = document.querySelector("#view-apply .error-msg");
        if (!el) { el = document.createElement("div"); el.className = "error-msg"; form.insertBefore(el, form.querySelector(".submit-btn")); }
        el.textContent = msg; el.classList.add("visible");
    }
    function hideError() { const el = document.querySelector("#view-apply .error-msg"); if (el) el.classList.remove("visible"); }

    void (async () => {
        try {
            await validateSession();
            updateAuthUI();
            if (authUser) {
                if (isStaff()) switchView("bank");
                else switchView("apply");
            }
        } catch (err) {
            console.error("Session / auth UI init failed:", err);
            try {
                updateAuthUI();
            } catch (e2) {
                console.error("updateAuthUI failed:", e2);
            }
        }
    })();
});
