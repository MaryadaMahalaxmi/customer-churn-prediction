/**
 * app.js - Client-side interactivity for Telco Customer Churn Predictor
 */

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (el) {
        el.value = value;
    }
}

/**
 * High Risk Customer Demo Preset:
 * - 2 months tenure, month-to-month, fiber optic, electronic check
 * - High charges, no security/support add-ons
 */
function fillHighRisk() {
    setInputValue('gender', 'Female');
    setInputValue('SeniorCitizen', '0');
    setInputValue('Partner', 'No');
    setInputValue('Dependents', 'No');
    setInputValue('tenure', '2');
    setInputValue('PhoneService', 'Yes');
    setInputValue('MultipleLines', 'No');
    setInputValue('InternetService', 'Fiber optic');
    setInputValue('OnlineSecurity', 'No');
    setInputValue('OnlineBackup', 'No');
    setInputValue('DeviceProtection', 'No');
    setInputValue('TechSupport', 'No');
    setInputValue('StreamingTV', 'No');
    setInputValue('StreamingMovies', 'No');
    setInputValue('Contract', 'Month-to-month');
    setInputValue('PaperlessBilling', 'Yes');
    setInputValue('PaymentMethod', 'Electronic check');
    setInputValue('MonthlyCharges', '70.70');
    setInputValue('TotalCharges', '151.65');

    highlightFormChange();
}

/**
 * Loyal Customer Demo Preset:
 * - 64 months tenure, two-year contract, automatic payment
 * - Security & tech support, low/moderate cost
 */
function fillLoyal() {
    setInputValue('gender', 'Male');
    setInputValue('SeniorCitizen', '0');
    setInputValue('Partner', 'Yes');
    setInputValue('Dependents', 'Yes');
    setInputValue('tenure', '64');
    setInputValue('PhoneService', 'Yes');
    setInputValue('MultipleLines', 'Yes');
    setInputValue('InternetService', 'DSL');
    setInputValue('OnlineSecurity', 'Yes');
    setInputValue('OnlineBackup', 'Yes');
    setInputValue('DeviceProtection', 'Yes');
    setInputValue('TechSupport', 'Yes');
    setInputValue('StreamingTV', 'No');
    setInputValue('StreamingMovies', 'Yes');
    setInputValue('Contract', 'Two year');
    setInputValue('PaperlessBilling', 'No');
    setInputValue('PaymentMethod', 'Bank transfer (automatic)');
    setInputValue('MonthlyCharges', '65.20');
    setInputValue('TotalCharges', '4172.80');

    highlightFormChange();
}

/**
 * Moderate Risk Demo Preset:
 * - 18 months tenure, one-year contract, credit card auto
 */
function fillMediumRisk() {
    setInputValue('gender', 'Female');
    setInputValue('SeniorCitizen', '0');
    setInputValue('Partner', 'Yes');
    setInputValue('Dependents', 'No');
    setInputValue('tenure', '20');
    setInputValue('PhoneService', 'Yes');
    setInputValue('MultipleLines', 'No');
    setInputValue('InternetService', 'DSL');
    setInputValue('OnlineSecurity', 'Yes');
    setInputValue('OnlineBackup', 'No');
    setInputValue('DeviceProtection', 'No');
    setInputValue('TechSupport', 'No');
    setInputValue('StreamingTV', 'No');
    setInputValue('StreamingMovies', 'No');
    setInputValue('Contract', 'One year');
    setInputValue('PaperlessBilling', 'Yes');
    setInputValue('PaymentMethod', 'Credit card (automatic)');
    setInputValue('MonthlyCharges', '54.50');
    setInputValue('TotalCharges', '1090.00');

    highlightFormChange();
}

function highlightFormChange() {
    const cards = document.querySelectorAll('.form-card');
    cards.forEach(card => {
        card.style.transition = 'box-shadow 0.3s ease';
        card.style.boxShadow = '0 0 0 3px rgba(79, 70, 229, 0.3)';
        setTimeout(() => {
            card.style.boxShadow = '';
        }, 500);
    });
}

// Auto-compute total charges if user updates monthly charges or tenure
document.addEventListener('DOMContentLoaded', () => {
    const tenureInput = document.getElementById('tenure');
    const monthlyInput = document.getElementById('MonthlyCharges');
    const totalInput = document.getElementById('TotalCharges');
    const form = document.getElementById('churnForm');
    const submitBtn = document.getElementById('submitBtn');

    function updateEstimatedTotal() {
        if (tenureInput && monthlyInput && totalInput) {
            const tenure = parseFloat(tenureInput.value) || 0;
            const monthly = parseFloat(monthlyInput.value) || 0;
            // Only auto-update if current total matches previous multiple or is empty
            if (!totalInput.value || totalInput.dataset.autocalc === "true") {
                totalInput.value = (tenure * monthly).toFixed(2);
                totalInput.dataset.autocalc = "true";
            }
        }
    }

    if (tenureInput) tenureInput.addEventListener('input', updateEstimatedTotal);
    if (monthlyInput) monthlyInput.addEventListener('input', updateEstimatedTotal);
    if (totalInput) {
        totalInput.addEventListener('input', () => {
            totalInput.dataset.autocalc = "false";
        });
    }

    if (form && submitBtn) {
        form.addEventListener('submit', () => {
            submitBtn.disabled = true;
            submitBtn.innerText = "⏳ Analyzing Customer Churn Risk...";
        });
    }
});
