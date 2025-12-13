// ===== DOM Elements =====
const form = document.getElementById('profileForm');
const formSection = document.getElementById('formSection');
const resultSection = document.getElementById('resultSection');
const resultSummary = document.getElementById('resultSummary');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const recurrenceTypeSection = document.getElementById('recurrenceTypeSection');

// ===== Form Fields for Progress Tracking =====
const requiredFields = ['age', 'bmi', 'recurrence'];

// ===== Event Listeners =====
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

function setupEventListeners() {
    // Track form progress
    form.addEventListener('input', updateProgress);
    form.addEventListener('change', updateProgress);

    // Handle recurrence conditional field
    document.querySelectorAll('input[name="recurrence"]').forEach(radio => {
        radio.addEventListener('change', handleRecurrenceChange);
    });

    // Handle form submission
    form.addEventListener('submit', handleSubmit);
}

// ===== Progress Tracking =====
function updateProgress() {
    let completed = 0;
    let total = requiredFields.length;

    // Check age
    if (document.getElementById('age').value) completed++;
    
    // Check BMI
    if (document.getElementById('bmi').value) completed++;
    
    // Check recurrence
    if (document.querySelector('input[name="recurrence"]:checked')) completed++;

    // If recurrence is yes, add recurrence type to requirements
    const recurrenceValue = document.querySelector('input[name="recurrence"]:checked')?.value;
    if (recurrenceValue === 'yes') {
        total++;
        const recurrenceTypes = document.querySelectorAll('input[name="recurrenceType"]:checked');
        if (recurrenceTypes.length > 0) completed++;
    }

    const percentage = Math.round((completed / total) * 100);
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${percentage}% completado`;
}

// ===== Conditional Fields =====
function handleRecurrenceChange(e) {
    if (e.target.value === 'yes') {
        recurrenceTypeSection.classList.add('visible');
    } else {
        recurrenceTypeSection.classList.remove('visible');
        // Clear recurrence type selections
        document.querySelectorAll('input[name="recurrenceType"]').forEach(cb => {
            cb.checked = false;
        });
    }
    updateProgress();
}

// ===== Form Submission =====
async function handleSubmit(e) {
    e.preventDefault();

    // Collect form data
    const formData = collectFormData();

    // Validate
    if (!validateForm(formData)) return;

    try {
        // Send to server
        const response = await fetch('/api/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        
        if (result.success) {
            displayResults(formData);
        }
    } catch (error) {
        console.error('Error:', error);
        // Still show results even if server fails
        displayResults(formData);
    }
}

function collectFormData() {
    const age = document.getElementById('age').value;
    const bmi = document.getElementById('bmi').value;
    const recurrence = document.querySelector('input[name="recurrence"]:checked')?.value;
    
    const recurrenceTypes = [];
    document.querySelectorAll('input[name="recurrenceType"]:checked').forEach(cb => {
        recurrenceTypes.push(cb.value);
    });

    return {
        age: parseInt(age),
        bmi: parseFloat(bmi),
        recurrence: recurrence === 'yes',
        recurrenceTypes: recurrenceTypes,
        timestamp: new Date().toISOString()
    };
}

function validateForm(data) {
    if (!data.age || data.age < 18 || data.age > 120) {
        alert('Por favor, introduzca una edad válida (18-120 años)');
        return false;
    }

    if (!data.bmi || data.bmi < 10 || data.bmi > 60) {
        alert('Por favor, introduzca un IMC válido (10-60)');
        return false;
    }

    if (data.recurrence === undefined) {
        alert('Por favor, indique si ha tenido recidiva');
        return false;
    }

    if (data.recurrence && data.recurrenceTypes.length === 0) {
        alert('Por favor, seleccione al menos un tipo de recidiva');
        return false;
    }

    return true;
}

// ===== Display Results =====
function displayResults(data) {
    // Build results HTML
    let html = `
        <div class="result-item">
            <span class="result-label">Edad</span>
            <span class="result-value">${data.age} años</span>
        </div>
        <div class="result-item">
            <span class="result-label">IMC</span>
            <span class="result-value">${data.bmi} ${getBMICategory(data.bmi)}</span>
        </div>
        <div class="result-item">
            <span class="result-label">Recidiva</span>
            <span class="result-value">${data.recurrence ? 'Sí' : 'No'}</span>
        </div>
    `;

    if (data.recurrence && data.recurrenceTypes.length > 0) {
        const typeLabels = {
            'ganglionar': '🔵 Ganglionar',
            'pulmonar': '🫁 Pulmonar',
            'peritoneal': '🟣 Peritoneal'
        };
        const types = data.recurrenceTypes.map(t => typeLabels[t]).join(', ');
        html += `
            <div class="result-item">
                <span class="result-label">Tipo(s) de recidiva</span>
                <span class="result-value">${types}</span>
            </div>
        `;
    }

    resultSummary.innerHTML = html;

    // Show results, hide form
    formSection.style.display = 'none';
    resultSection.classList.add('visible');

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function getBMICategory(bmi) {
    if (bmi < 18.5) return '(Bajo peso)';
    if (bmi < 25) return '(Normal)';
    if (bmi < 30) return '(Sobrepeso)';
    return '(Obesidad)';
}

// ===== Reset Form =====
function resetForm() {
    form.reset();
    recurrenceTypeSection.classList.remove('visible');
    resultSection.classList.remove('visible');
    formSection.style.display = 'block';
    updateProgress();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
