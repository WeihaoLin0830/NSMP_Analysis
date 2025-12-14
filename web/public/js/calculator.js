/**
 * Patient Profile Calculator
 * Handles form submission, data processing, and result display
 * With automatic imputation of missing values using medians
 */

// =============================================================================
// Configuration
// =============================================================================

// Median values for imputation (from dataset)
const MEDIANS = {
    imc: 30.20,
    valor_de_ca125: 3.23,
    tamano_tumoral: 1.39,
    recep_est_porcent: 0.90,
    rece_de_Ppor: 0.80
};

// Numeric columns that have associated _miss flags
const NUM_COLS_WITH_MISS_FLAG = ['imc', 'tamano_tumoral', 'valor_de_ca125'];

// All expected columns
const EXPECTED_COLUMNS = {
    numeric: ['imc', 'valor_de_ca125', 'tamano_tumoral', 'recep_est_porcent', 'rece_de_Ppor', 'edad_en_cirugia'],
    categorical: ['asa', 'histo_defin', 'grado_histologi', 'FIGO2023', 'afectacion_linf', 'metasta_distan', 'AP_centinela_pelvico', 'AP_ganPelv', 'AP_glanPaor', 'beta_cateninap'],
    flags: ['histo_defin__from_pre', 'grado_histologi__from_pre', 'FIGO2023__from_pre', 'imc_miss', 'tamano_tumoral_miss', 'valor_de_ca125_miss']
};

// Cluster info for display
const CLUSTER_INFO = {
    1: {
        name: "Cluster 1 - Bajo Riesgo",
        color: "#27ae60",
        icon: "✅",
        description: "Tu perfil se asocia con un grupo de menor riesgo de recurrencia.",
        recommendations: [
            "Seguimiento regular cada 6 meses",
            "Mantener estilo de vida saludable",
            "Buen pronóstico general"
        ]
    },
    2: {
        name: "Cluster 2 - Seguimiento Requerido",
        color: "#f39c12",
        icon: "⚠️",
        description: "Tu perfil requiere un seguimiento más cercano por el equipo médico.",
        recommendations: [
            "Seguimiento cada 3-4 meses",
            "Adherencia al tratamiento recomendado",
            "Comunicación regular con el equipo médico"
        ]
    }
};

const RISK_GROUP_INFO = {
    bajo: { name: "Riesgo Bajo", color: "#27ae60", icon: "🟢" },
    medio: { name: "Riesgo Medio", color: "#f39c12", icon: "🟡" },
    alto: { name: "Riesgo Alto", color: "#e74c3c", icon: "🔴" }
};

// =============================================================================
// DOM Elements
// =============================================================================

const form = document.getElementById('profileForm');
const formSection = document.getElementById('formSection');
const loadingSection = document.getElementById('loadingSection');
const resultSection = document.getElementById('resultSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// =============================================================================
// Event Listeners
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateProgress();
});

function setupEventListeners() {
    // Track form progress
    form.addEventListener('input', updateProgress);
    form.addEventListener('change', updateProgress);

    // Handle form submission
    form.addEventListener('submit', handleSubmit);
}

// =============================================================================
// Progress Tracking
// =============================================================================

function updateProgress() {
    let completed = 0;
    let total = 1; // edad_en_cirugia is required

    // Check required field (edad_en_cirugia)
    const edad = document.getElementById('edad_en_cirugia').value;
    if (edad) completed++;

    // Count optional but filled fields (for user feedback)
    EXPECTED_COLUMNS.numeric.forEach(field => {
        if (field !== 'edad_en_cirugia') {
            total++;
            const input = document.getElementById(field);
            if (input && input.value.trim()) completed++;
        }
    });

    EXPECTED_COLUMNS.categorical.forEach(field => {
        total++;
        const input = document.getElementById(field);
        if (input && input.value) completed++;
    });

    const percentage = Math.round((completed / total) * 100);
    progressFill.style.width = `${percentage}%`;
    progressText.textContent = `${percentage}% completado`;
}

// =============================================================================
// Data Collection & Processing
// =============================================================================

function collectFormData() {
    const data = {};
    const missingFlags = {};

    // Collect numeric values
    EXPECTED_COLUMNS.numeric.forEach(field => {
        const input = document.getElementById(field);
        const rawValue = input ? input.value.trim() : '';
        
        // Check if value is a valid number
        const numValue = parseFloat(rawValue);
        
        if (rawValue === '' || isNaN(numValue)) {
            // Value is missing - use median if available
            if (MEDIANS[field] !== undefined) {
                data[field] = MEDIANS[field];
            } else if (field === 'edad_en_cirugia') {
                // This is required, should not happen
                data[field] = 0;
            } else {
                data[field] = 0;
            }
            
            // Set missing flag if this column has one
            if (NUM_COLS_WITH_MISS_FLAG.includes(field)) {
                missingFlags[`${field}_miss`] = 1;
            }
        } else {
            // Valid numeric value
            data[field] = numValue;
            
            // Set missing flag to 0 if this column has one
            if (NUM_COLS_WITH_MISS_FLAG.includes(field)) {
                missingFlags[`${field}_miss`] = 0;
            }
        }
    });

    // Collect categorical values
    EXPECTED_COLUMNS.categorical.forEach(field => {
        const input = document.getElementById(field);
        const value = input ? input.value : '';
        
        if (value === '' || value === null) {
            // Missing categorical - use "Missing" as per the model
            data[field] = 'Missing';
        } else {
            data[field] = value;
        }
    });

    // Collect flag values (checkboxes)
    ['histo_defin__from_pre', 'grado_histologi__from_pre', 'FIGO2023__from_pre'].forEach(field => {
        const input = document.getElementById(field);
        data[field] = input && input.checked ? 1 : 0;
    });

    // Add missing flags
    data.imc_miss = missingFlags.imc_miss ?? 0;
    data.tamano_tumoral_miss = missingFlags.tamano_tumoral_miss ?? 0;
    data.valor_de_ca125_miss = missingFlags.valor_de_ca125_miss ?? 0;

    return data;
}

function validateForm(data) {
    // Only edad_en_cirugia is strictly required
    if (!data.edad_en_cirugia || data.edad_en_cirugia <= 0) {
        alert('Por favor, introduzca la edad en cirugía.');
        return false;
    }

    if (data.edad_en_cirugia < 18 || data.edad_en_cirugia > 100) {
        alert('La edad debe estar entre 18 y 100 años.');
        return false;
    }

    return true;
}

// =============================================================================
// Form Submission
// =============================================================================

async function handleSubmit(e) {
    e.preventDefault();

    // Collect and process form data
    const formData = collectFormData();
    
    console.log('Collected form data:', formData);

    // Validate
    if (!validateForm(formData)) return;

    // Show loading
    formSection.style.display = 'none';
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';

    try {
        // Send to backend for prediction
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        
        console.log('Prediction result:', result);
        
        if (result.success) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'Error en la predicción');
        }
    } catch (error) {
        console.error('Error:', error);
        loadingSection.style.display = 'none';
        formSection.style.display = 'block';
        alert('Error al procesar la solicitud: ' + error.message);
    }
}

// =============================================================================
// Display Results
// =============================================================================

function displayResults(result) {
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';

    const prediction = result.prediction;
    const clusterInfo = CLUSTER_INFO[prediction.cluster_pred] || CLUSTER_INFO[1];
    const riskInfo = RISK_GROUP_INFO[prediction.risk_group] || RISK_GROUP_INFO.medio;

    // Build results HTML
    let html = `
        <!-- Result Header -->
        <div class="result-header">
            <div class="result-icon">${clusterInfo.icon}</div>
            <h2>Análisis de Perfil Completado</h2>
            <p style="color: var(--text-secondary);">Resultados basados en el modelo de clustering y supervivencia</p>
            <div class="cluster-badge cluster-${prediction.cluster_pred}" style="background: ${clusterInfo.color}; color: white;">
                ${clusterInfo.name}
            </div>
        </div>

        <!-- Survival Probabilities -->
        <h3 style="margin: 2rem 0 1rem; color: var(--primary);">📊 Probabilidad de Supervivencia</h3>
        <div class="survival-grid">
            <div class="survival-card">
                <h4>1 Año</h4>
                <div class="value ${getSurvivalClass(prediction.S_1y)}">${(prediction.S_1y * 100).toFixed(1)}%</div>
                <div class="label">S(1y)</div>
            </div>
            <div class="survival-card">
                <h4>3 Años</h4>
                <div class="value ${getSurvivalClass(prediction.S_3y)}">${(prediction.S_3y * 100).toFixed(1)}%</div>
                <div class="label">S(3y)</div>
            </div>
            <div class="survival-card">
                <h4>5 Años</h4>
                <div class="value ${getSurvivalClass(prediction.S_5y)}">${(prediction.S_5y * 100).toFixed(1)}%</div>
                <div class="label">S(5y)</div>
            </div>
            <div class="survival-card">
                <h4>Grupo de Riesgo</h4>
                <div class="value" style="font-size: 1.5rem; color: ${riskInfo.color};">
                    ${riskInfo.icon} ${riskInfo.name}
                </div>
                <div class="label">Riesgo a 3 años: ${(prediction.risk_3y * 100).toFixed(1)}%</div>
            </div>
        </div>
    `;

    // Subcluster info if available
    if (prediction.cluster_pred === 1 && prediction.subcluster_pred) {
        html += `
            <div class="info-card" style="margin-bottom: 1.5rem;">
                <h3>🔬 Subcluster</h3>
                <p>Subgrupo asignado: <strong>${prediction.subcluster_pred}</strong></p>
                <p>Confianza: ${prediction.p_subcluster11 !== null ? (prediction.p_subcluster11 * 100).toFixed(1) + '%' : 'N/A'}</p>
            </div>
        `;
    }

    // Info cards
    html += `
        <div class="info-grid">
            <div class="info-card">
                <h3>📋 Descripción</h3>
                <p>${clusterInfo.description}</p>
            </div>
            <div class="info-card">
                <h3>💡 Recomendaciones</h3>
                <ul>
                    ${clusterInfo.recommendations.map(r => `<li>${r}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;

    // Warnings if any
    const warnings = prediction.warnings || {};
    const activeWarnings = Object.entries(warnings).filter(([k, v]) => v);
    
    if (activeWarnings.length > 0) {
        html += `
            <div class="warnings-section">
                <h4>⚠️ Advertencias</h4>
                <ul>
                    ${activeWarnings.map(([k, v]) => `<li>${getWarningMessage(k)}</li>`).join('')}
                </ul>
            </div>
        `;
    }

    // Actions
    html += `
        <div class="result-actions">
            <button class="btn btn-secondary" onclick="resetForm()">
                <span>🔄</span> Nueva Evaluación
            </button>
            <button class="btn btn-primary" onclick="window.print()">
                <span>🖨️</span> Imprimir Resultados
            </button>
        </div>
    `;

    resultSection.innerHTML = html;
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

function getSurvivalClass(value) {
    if (value >= 0.8) return 'high';
    if (value >= 0.5) return 'medium';
    return 'low';
}

function getWarningMessage(key) {
    const messages = {
        'many_missing_flag': 'Se detectaron varios valores faltantes en los datos ingresados.',
        'low_confidence_cluster': 'La asignación de cluster tiene baja confianza.',
        'low_confidence_flag': 'La asignación de subcluster tiene baja confianza.',
        'unknown_category_flag': 'Se detectaron categorías no reconocidas en los datos.'
    };
    return messages[key] || key;
}

// =============================================================================
// Reset Form
// =============================================================================

function resetForm() {
    form.reset();
    formSection.style.display = 'block';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    updateProgress();
    formSection.scrollIntoView({ behavior: 'smooth' });
}

// Make resetForm available globally
window.resetForm = resetForm;
