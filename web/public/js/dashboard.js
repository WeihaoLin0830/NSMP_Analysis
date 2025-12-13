// ===== Cluster Profiles Data =====
const clusterProfiles = {
    1: {
        name: "Bajo Riesgo",
        color: "#a8d5a2",
        characteristics: [
            { label: "Edad media", value: "52 años" },
            { label: "IMC medio", value: "24.3" },
            { label: "Recidiva", value: "12% de casos" },
            { label: "Tipo histológico predominante", value: "Endometrioide G1" },
            { label: "Estadio FIGO", value: "IA (85%)" }
        ],
        symptoms: [
            "Sangrado postmenopáusico leve",
            "Detección temprana por screening",
            "Sin síntomas en 30% de casos"
        ],
        behaviors: [
            "Buena respuesta a tratamiento conservador",
            "Baja tasa de metástasis",
            "Remisión completa frecuente"
        ],
        treatments: [
            "Cirugía conservadora",
            "Seguimiento activo",
            "Braquiterapia opcional"
        ]
    },
    2: {
        name: "Riesgo Moderado",
        color: "#f4d03f",
        characteristics: [
            { label: "Edad media", value: "58 años" },
            { label: "IMC medio", value: "28.1" },
            { label: "Recidiva", value: "24% de casos" },
            { label: "Tipo histológico predominante", value: "Endometrioide G2" },
            { label: "Estadio FIGO", value: "IB-II (70%)" }
        ],
        symptoms: [
            "Sangrado irregular abundante",
            "Dolor pélvico ocasional",
            "Fatiga moderada"
        ],
        behaviors: [
            "Respuesta variable al tratamiento",
            "Posible afectación linfovascular",
            "Requiere seguimiento estrecho"
        ],
        treatments: [
            "Histerectomía total + anexectomía",
            "Linfadenectomía pélvica",
            "Radioterapia adyuvante"
        ]
    },
    3: {
        name: "Alto Riesgo",
        color: "#e67e22",
        characteristics: [
            { label: "Edad media", value: "64 años" },
            { label: "IMC medio", value: "32.5" },
            { label: "Recidiva", value: "45% de casos" },
            { label: "Tipo histológico predominante", value: "Endometrioide G3 / Seroso" },
            { label: "Estadio FIGO", value: "II-IIIA (65%)" }
        ],
        symptoms: [
            "Sangrado abundante persistente",
            "Dolor pélvico frecuente",
            "Pérdida de peso involuntaria",
            "Masa palpable en algunos casos"
        ],
        behaviors: [
            "Alto riesgo de recidiva local",
            "Afectación ganglionar frecuente",
            "Requiere tratamiento agresivo"
        ],
        treatments: [
            "Cirugía radical",
            "Quimioterapia adyuvante (Carboplatino/Paclitaxel)",
            "Radioterapia pélvica + braquiterapia"
        ]
    },
    4: {
        name: "Riesgo Elevado",
        color: "#e74c3c",
        characteristics: [
            { label: "Edad media", value: "68 años" },
            { label: "IMC medio", value: "29.8" },
            { label: "Recidiva", value: "67% de casos" },
            { label: "Tipo histológico predominante", value: "Seroso / Células claras" },
            { label: "Estadio FIGO", value: "IIIB-IV (80%)" }
        ],
        symptoms: [
            "Síntomas sistémicos marcados",
            "Dolor abdominal difuso",
            "Ascitis en casos avanzados",
            "Síntomas respiratorios (metástasis)"
        ],
        behaviors: [
            "Metástasis a distancia frecuente",
            "Resistencia a tratamientos convencionales",
            "Progresión rápida"
        ],
        treatments: [
            "Quimioterapia neoadyuvante",
            "Cirugía citorreductora si factible",
            "Terapia combinada QT+RT",
            "Considerar ensayos clínicos"
        ]
    },
    5: {
        name: "Casos Atípicos",
        color: "#9b59b6",
        characteristics: [
            { label: "Edad media", value: "55 años" },
            { label: "IMC medio", value: "26.4" },
            { label: "Recidiva", value: "35% de casos" },
            { label: "Tipo histológico predominante", value: "Mixto / No clasificable" },
            { label: "Estadio FIGO", value: "Variable" }
        ],
        symptoms: [
            "Presentación atípica",
            "Síntomas no específicos",
            "Diagnóstico tardío frecuente"
        ],
        behaviors: [
            "Comportamiento impredecible",
            "Requiere estudio molecular detallado",
            "Respuesta variable a tratamientos estándar"
        ],
        treatments: [
            "Evaluación multidisciplinar",
            "Tratamiento individualizado",
            "Considerar inmunoterapia",
            "Seguimiento personalizado"
        ]
    }
};

// ===== Initialize Dashboard =====
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initTabs();
    loadClusterProfile(1);
});

// ===== Charts Initialization =====
function initCharts() {
    // Cluster Distribution Chart
    const ctxDistribution = document.getElementById('clusterDistribution').getContext('2d');
    new Chart(ctxDistribution, {
        type: 'doughnut',
        data: {
            labels: ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'],
            datasets: [{
                data: [42, 38, 28, 19, 12],
                backgroundColor: [
                    '#a8d5a2',
                    '#f4d03f',
                    '#e67e22',
                    '#e74c3c',
                    '#9b59b6'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        font: { size: 12 }
                    }
                }
            }
        }
    });

    // Remission Rate Chart
    const ctxRemission = document.getElementById('remissionRate').getContext('2d');
    new Chart(ctxRemission, {
        type: 'bar',
        data: {
            labels: ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'],
            datasets: [{
                label: 'Tasa de Remisión (%)',
                data: [87, 71, 52, 34, 65],
                backgroundColor: [
                    'rgba(168, 213, 162, 0.8)',
                    'rgba(244, 208, 63, 0.8)',
                    'rgba(230, 126, 34, 0.8)',
                    'rgba(231, 76, 60, 0.8)',
                    'rgba(155, 89, 182, 0.8)'
                ],
                borderColor: [
                    '#a8d5a2',
                    '#f4d03f',
                    '#e67e22',
                    '#e74c3c',
                    '#9b59b6'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: value => value + '%'
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Survival Chart
    const ctxSurvival = document.getElementById('survivalChart').getContext('2d');
    new Chart(ctxSurvival, {
        type: 'line',
        data: {
            labels: ['0', '6', '12', '18', '24', '30', '36', '42', '48', '54', '60'],
            datasets: [
                {
                    label: 'Cluster 1',
                    data: [100, 98, 96, 94, 92, 90, 89, 88, 87, 87, 87],
                    borderColor: '#a8d5a2',
                    backgroundColor: 'rgba(168, 213, 162, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Cluster 2',
                    data: [100, 95, 90, 85, 80, 77, 74, 72, 71, 71, 71],
                    borderColor: '#f4d03f',
                    backgroundColor: 'rgba(244, 208, 63, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Cluster 3',
                    data: [100, 90, 80, 70, 62, 57, 54, 52, 52, 52, 52],
                    borderColor: '#e67e22',
                    backgroundColor: 'rgba(230, 126, 34, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Cluster 4',
                    data: [100, 82, 68, 55, 45, 40, 36, 34, 34, 34, 34],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Cluster 5',
                    data: [100, 92, 84, 76, 70, 67, 65, 65, 65, 65, 65],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    fill: true,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Meses desde diagnóstico'
                    }
                },
                y: {
                    beginAtZero: false,
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Supervivencia (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Curvas de Supervivencia Kaplan-Meier (Simuladas)'
                }
            }
        }
    });
}

// ===== Tabs Functionality =====
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadClusterProfile(parseInt(btn.dataset.cluster));
        });
    });
}

// ===== Load Cluster Profile =====
function loadClusterProfile(clusterId) {
    const profile = clusterProfiles[clusterId];
    const container = document.getElementById('profileContent');
    
    container.innerHTML = `
        <div class="profile-grid">
            <div class="profile-card">
                <h3 style="border-left: 4px solid ${profile.color}; padding-left: 12px;">
                    📋 Características
                </h3>
                <div class="characteristics-list">
                    ${profile.characteristics.map(c => `
                        <div class="char-item">
                            <span class="char-label">${c.label}</span>
                            <span class="char-value">${c.value}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="profile-card">
                <h3 style="border-left: 4px solid ${profile.color}; padding-left: 12px;">
                    🩺 Síntomas
                </h3>
                <ul class="info-list symptoms">
                    ${profile.symptoms.map(s => `<li>${s}</li>`).join('')}
                </ul>
            </div>
            
            <div class="profile-card">
                <h3 style="border-left: 4px solid ${profile.color}; padding-left: 12px;">
                    📈 Comportamiento
                </h3>
                <ul class="info-list behaviors">
                    ${profile.behaviors.map(b => `<li>${b}</li>`).join('')}
                </ul>
            </div>
            
            <div class="profile-card">
                <h3 style="border-left: 4px solid ${profile.color}; padding-left: 12px;">
                    💊 Tratamientos
                </h3>
                <ul class="info-list treatments">
                    ${profile.treatments.map(t => `<li>${t}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
}
