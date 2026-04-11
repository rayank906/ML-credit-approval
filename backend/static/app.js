// ── Auth State ────────────────────────────────────────────────
let authToken = localStorage.getItem("authToken");
let authUser = JSON.parse(localStorage.getItem("authUser") || "null");

function updateAuthUI() {
    const bar = document.getElementById("auth-bar");
    if (authUser) {
        bar.innerHTML = `
            <span class="user-info">Signed in as <strong>${authUser.full_name}</strong></span>
            <button class="auth-btn small" onclick="logout()">Log Out</button>
        `;
        loadHistory();
    } else {
        bar.innerHTML = `
            <button class="auth-btn" onclick="openModal('login')">Log In</button>
            <button class="auth-btn primary" onclick="openModal('register')">Sign Up</button>
        `;
        const hist = document.getElementById("history-section");
        if (hist) hist.remove();
    }
}

function openModal(mode) {
    document.getElementById("auth-modal").classList.add("active");
    switchModal(mode);
}

function closeModal() {
    document.getElementById("auth-modal").classList.remove("active");
    document.querySelectorAll(".auth-error").forEach(e => { e.classList.remove("visible"); e.textContent = ""; });
}

function switchModal(mode) {
    document.getElementById("login-form").style.display = mode === "login" ? "block" : "none";
    document.getElementById("register-form").style.display = mode === "register" ? "block" : "none";
    document.querySelectorAll(".auth-error").forEach(e => { e.classList.remove("visible"); e.textContent = ""; });
}

function showAuthError(formId, msg) {
    const el = document.getElementById(formId);
    el.textContent = msg;
    el.classList.add("visible");
}

async function handleLogin() {
    const email = document.getElementById("login-email").value.trim();
    const password = document.getElementById("login-password").value;
    if (!email || !password) { showAuthError("login-error", "Please fill in all fields."); return; }

    try {
        const resp = await fetch("/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Login failed");

        authToken = data.token;
        authUser = { email: data.email, full_name: data.full_name, role: data.role };
        localStorage.setItem("authToken", authToken);
        localStorage.setItem("authUser", JSON.stringify(authUser));
        closeModal();
        updateAuthUI();
    } catch (err) {
        showAuthError("login-error", err.message);
    }
}

async function handleRegister() {
    const full_name = document.getElementById("register-name").value.trim();
    const email = document.getElementById("register-email").value.trim();
    const password = document.getElementById("register-password").value;
    if (!full_name || !email || !password) { showAuthError("register-error", "Please fill in all fields."); return; }
    if (password.length < 6) { showAuthError("register-error", "Password must be at least 6 characters."); return; }

    try {
        const resp = await fetch("/auth/register", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password, full_name }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || "Registration failed");

        authToken = data.token;
        authUser = { email: data.email, full_name: data.full_name, role: data.role };
        localStorage.setItem("authToken", authToken);
        localStorage.setItem("authUser", JSON.stringify(authUser));
        closeModal();
        updateAuthUI();
    } catch (err) {
        showAuthError("register-error", err.message);
    }
}

function logout() {
    authToken = null;
    authUser = null;
    localStorage.removeItem("authToken");
    localStorage.removeItem("authUser");
    updateAuthUI();
}

// ── Application History ──────────────────────────────────────
async function loadHistory() {
    if (!authToken) return;
    try {
        const resp = await fetch("/applications", {
            headers: { "Authorization": `Bearer ${authToken}` },
        });
        if (!resp.ok) return;
        const apps = await resp.json();

        let section = document.getElementById("history-section");
        if (!section) {
            section = document.createElement("div");
            section.id = "history-section";
            section.className = "history-section";
            document.getElementById("results-panel").appendChild(section);
        }

        if (apps.length === 0) {
            section.innerHTML = `<h3>Your Applications</h3><p style="font-size:0.85rem;color:var(--text-muted);">No applications yet. Submit one above!</p>`;
            return;
        }

        let html = `<h3>Your Applications (${apps.length})</h3>`;
        apps.forEach(app => {
            const date = new Date(app.created_at).toLocaleDateString();
            const cls = app.decision === "Approved" ? "approved" : "rejected";
            html += `
                <div class="history-item">
                    <div>
                        <span class="history-decision ${cls}">${app.decision}</span>
                        <span class="history-meta" style="margin-left:8px;">Score: ${app.credit_score}</span>
                    </div>
                    <span class="history-meta">${date}</span>
                </div>`;
        });
        section.innerHTML = html;
    } catch (e) {
        // silently fail
    }
}

// ── Main App ─────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    updateAuthUI();

    const form = document.getElementById("approval-form");

    // ── Toggle button groups ─────────────────────────────────
    document.querySelectorAll(".toggle-group").forEach((group) => {
        group.querySelectorAll(".toggle-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                group.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });
    });

    // ── Auto-sync children -> family members ──────────────────
    const childrenInput = document.getElementById("children");
    const familyInput = document.getElementById("family_members");
    childrenInput.addEventListener("input", () => {
        const kids = parseInt(childrenInput.value) || 0;
        const current = parseInt(familyInput.value) || 1;
        if (current < kids + 1) {
            familyInput.value = kids + 1;
        }
    });

    // ── Form submission ──────────────────────────────────────
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Collect toggle values
        const toggleValues = {};
        let missingToggle = false;
        document.querySelectorAll(".toggle-group").forEach((group) => {
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

        if (missingToggle) {
            showError("Please select an option for all toggle buttons.");
            return;
        }

        // Validate required fields
        const requiredSelects = form.querySelectorAll("select[required]");
        let missingSelect = false;
        requiredSelects.forEach((sel) => {
            if (!sel.value) {
                sel.closest(".form-group").classList.add("error");
                missingSelect = true;
            } else {
                sel.closest(".form-group").classList.remove("error");
            }
        });

        if (missingSelect) {
            showError("Please fill in all required fields.");
            return;
        }

        hideError();

        const payload = {
            gender: toggleValues.gender,
            own_car: toggleValues.own_car,
            own_realty: toggleValues.own_realty,
            work_phone: toggleValues.work_phone,
            phone: toggleValues.phone,
            email: toggleValues.email,
            age: document.getElementById("age").value,
            family_status: document.getElementById("family_status").value,
            children: document.getElementById("children").value,
            family_members: document.getElementById("family_members").value,
            education: document.getElementById("education").value,
            income_type: document.getElementById("income_type").value,
            income: document.getElementById("income").value,
            occupation: document.getElementById("occupation").value,
            years_employed: document.getElementById("years_employed").value,
            housing: document.getElementById("housing").value,
        };

        const btn = document.getElementById("submit-btn");
        btn.classList.add("loading");
        btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Evaluating...`;

        try {
            const headers = { "Content-Type": "application/json" };
            if (authToken) {
                headers["Authorization"] = `Bearer ${authToken}`;
            }

            const resp = await fetch("/predict", {
                method: "POST",
                headers,
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || "Server error");
            }

            const result = await resp.json();
            displayResults(result);

            // Refresh history if logged in
            if (authToken) loadHistory();
        } catch (err) {
            showError("Error: " + err.message);
        } finally {
            btn.classList.remove("loading");
            btn.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Evaluate Application`;
        }
    });

    function displayResults(data) {
        const placeholder = document.getElementById("results-placeholder");
        const content = document.getElementById("results-content");
        placeholder.style.display = "none";
        content.style.display = "block";
        content.style.animation = "none";
        content.offsetHeight; // reflow
        content.style.animation = "";

        // Decision banner
        const banner = document.getElementById("decision-banner");
        const icon = document.getElementById("decision-icon");
        const text = document.getElementById("decision-text");
        const strength = document.getElementById("decision-strength");

        banner.className = "decision-banner";
        if (data.decision === "Approved") {
            banner.classList.add("approved");
            icon.textContent = "\u2713";
        } else if (data.probability_risky > 40 && data.probability_risky < 60) {
            banner.classList.add("borderline");
            icon.textContent = "\u2014";
        } else {
            banner.classList.add("rejected");
            icon.textContent = "\u2717";
        }

        text.textContent = data.decision;
        strength.textContent = data.strength.description;

        // Gauge
        const marker = document.getElementById("gauge-marker");
        marker.style.left = data.probability_risky + "%";

        document.getElementById("prob-safe").textContent = data.probability_safe + "%";
        document.getElementById("prob-risky").textContent = data.probability_risky + "%";

        // Factors
        const list = document.getElementById("factors-list");
        list.innerHTML = "";
        data.top_factors.forEach((f, i) => {
            const item = document.createElement("div");
            item.className = "factor-item";
            item.innerHTML = `
                <div class="factor-rank">${i + 1}</div>
                <div class="factor-info">
                    <div class="factor-name">${f.feature}</div>
                    <div class="factor-value">Your value: ${f.value}</div>
                </div>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:${f.importance}%"></div>
                </div>
            `;
            list.appendChild(item);
        });

        // Scroll to results on mobile
        if (window.innerWidth <= 900) {
            document.getElementById("results-panel").scrollIntoView({ behavior: "smooth" });
        }
    }

    function showError(msg) {
        let el = document.querySelector(".error-msg");
        if (!el) {
            el = document.createElement("div");
            el.className = "error-msg";
            form.insertBefore(el, form.querySelector(".submit-btn"));
        }
        el.textContent = msg;
        el.classList.add("visible");
    }

    function hideError() {
        const el = document.querySelector(".error-msg");
        if (el) el.classList.remove("visible");
    }
});
