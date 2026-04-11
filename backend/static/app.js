document.addEventListener("DOMContentLoaded", () => {
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

    // ── Auto-sync children → family members ──────────────────
    const childrenInput = document.getElementById("children");
    const familyInput = document.getElementById("family_members");
    childrenInput.addEventListener("input", () => {
        const kids = parseInt(childrenInput.value) || 0;
        const current = parseInt(familyInput.value) || 1;
        // At minimum: applicant + children
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
            const resp = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || "Server error");
            }

            const result = await resp.json();
            displayResults(result);
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
        // Re-trigger animation
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
