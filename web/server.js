const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// ===== Simulated Cluster Data (based on clustering analysis) =====
const clusterCentroids = [
    { id: 1, age: 52, bmi: 24.3, recurrence: 0, recurrenceType: 0 },  // Bajo riesgo
    { id: 2, age: 58, bmi: 28.1, recurrence: 0.24, recurrenceType: 1 }, // Riesgo moderado
    { id: 3, age: 64, bmi: 32.5, recurrence: 0.45, recurrenceType: 2 }, // Alto riesgo
    { id: 4, age: 68, bmi: 29.8, recurrence: 0.67, recurrenceType: 3 }, // Riesgo elevado
    { id: 5, age: 55, bmi: 26.4, recurrence: 0.35, recurrenceType: 1 }  // Casos atípicos
];

const clusterInfo = {
    1: {
        name: "Bajo Riesgo",
        color: "#a8d5a2",
        remissionRate: 87,
        description: "Tu perfil se asocia con el grupo de menor riesgo. Los pacientes en este grupo generalmente presentan buena evolución.",
        characteristics: [
            "Alta probabilidad de remisión completa",
            "Respuesta favorable a tratamientos conservadores",
            "Baja tasa de recidiva"
        ],
        recommendations: [
            "Seguimiento regular cada 6 meses",
            "Mantener estilo de vida saludable",
            "Control de peso y actividad física moderada"
        ],
        treatments: [
            "Cirugía conservadora como primera opción",
            "Braquiterapia adyuvante en casos seleccionados",
            "Seguimiento activo con controles periódicos"
        ],
        prognosis: "Favorable - La mayoría de pacientes logran remisión completa"
    },
    2: {
        name: "Riesgo Moderado",
        color: "#f4d03f",
        remissionRate: 71,
        description: "Tu perfil indica un riesgo moderado. Con el tratamiento adecuado, la mayoría de pacientes tienen buena evolución.",
        characteristics: [
            "Respuesta variable al tratamiento inicial",
            "Posible afectación linfovascular",
            "Requiere seguimiento más estrecho"
        ],
        recommendations: [
            "Seguimiento cada 4 meses",
            "Control estricto del peso corporal",
            "Comunicación regular con el equipo médico"
        ],
        treatments: [
            "Histerectomía total con anexectomía bilateral",
            "Linfadenectomía pélvica selectiva",
            "Radioterapia pélvica adyuvante"
        ],
        prognosis: "Moderado - Buenas posibilidades con tratamiento completo"
    },
    3: {
        name: "Alto Riesgo",
        color: "#e67e22",
        remissionRate: 52,
        description: "Tu perfil sugiere un riesgo elevado. Es importante un abordaje terapéutico intensivo y seguimiento cercano.",
        characteristics: [
            "Mayor probabilidad de recidiva local",
            "Posible afectación ganglionar",
            "Requiere tratamiento multimodal"
        ],
        recommendations: [
            "Seguimiento cada 3 meses",
            "Adherencia estricta al tratamiento",
            "Apoyo psicológico recomendado",
            "Participación en grupos de apoyo"
        ],
        treatments: [
            "Cirugía radical con linfadenectomía completa",
            "Quimioterapia adyuvante (Carboplatino/Paclitaxel)",
            "Radioterapia pélvica con braquiterapia"
        ],
        prognosis: "Reservado - Requiere tratamiento agresivo y seguimiento intensivo"
    },
    4: {
        name: "Riesgo Elevado",
        color: "#e74c3c",
        remissionRate: 34,
        description: "Tu perfil indica un riesgo significativo. El equipo médico diseñará un plan de tratamiento personalizado e intensivo.",
        characteristics: [
            "Alto riesgo de metástasis a distancia",
            "Posible resistencia a tratamientos convencionales",
            "Necesidad de enfoque multidisciplinar"
        ],
        recommendations: [
            "Seguimiento mensual",
            "Apoyo integral (médico, psicológico, nutricional)",
            "Considerar participación en ensayos clínicos",
            "Red de apoyo familiar importante"
        ],
        treatments: [
            "Quimioterapia neoadyuvante",
            "Cirugía citorreductora si es factible",
            "Terapia combinada QT + RT",
            "Evaluación para inmunoterapia o terapias dirigidas"
        ],
        prognosis: "Complejo - Tratamiento intensivo con enfoque personalizado"
    },
    5: {
        name: "Perfil Atípico",
        color: "#9b59b6",
        remissionRate: 65,
        description: "Tu perfil presenta características particulares que requieren una evaluación individualizada.",
        characteristics: [
            "Comportamiento clínico variable",
            "Puede beneficiarse de estudios moleculares adicionales",
            "Respuesta a tratamiento impredecible"
        ],
        recommendations: [
            "Evaluación por comité multidisciplinar",
            "Posibles estudios genéticos adicionales",
            "Plan de seguimiento personalizado"
        ],
        treatments: [
            "Tratamiento individualizado según evaluación",
            "Posible inclusión en protocolos de investigación",
            "Considerar terapias dirigidas según perfil molecular"
        ],
        prognosis: "Variable - Depende de la respuesta individual al tratamiento"
    }
};

// ===== KNN Classification Function =====
function classifyPatient(patientData) {
    // Normalize patient data
    const recurrenceValue = patientData.recurrence ? 1 : 0;
    const recurrenceTypeValue = patientData.recurrenceTypes.length > 0 
        ? (patientData.recurrenceTypes.includes('pulmonar') ? 3 
           : patientData.recurrenceTypes.includes('peritoneal') ? 2 
           : 1)
        : 0;
    
    // Calculate distances to each centroid (simplified Euclidean)
    const distances = clusterCentroids.map(centroid => {
        const ageDiff = (patientData.age - centroid.age) / 20; // Normalized
        const bmiDiff = (patientData.bmi - centroid.bmi) / 10;
        const recDiff = recurrenceValue - centroid.recurrence;
        const typeDiff = (recurrenceTypeValue - centroid.recurrenceType) / 3;
        
        const distance = Math.sqrt(
            ageDiff * ageDiff + 
            bmiDiff * bmiDiff + 
            recDiff * recDiff * 2 + // Weight recurrence more
            typeDiff * typeDiff
        );
        
        return { clusterId: centroid.id, distance };
    });
    
    // Find nearest cluster
    distances.sort((a, b) => a.distance - b.distance);
    return distances[0].clusterId;
}

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'role-select.html'));
});

app.get('/calculator', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'calculator.html'));
});

app.get('/dashboard', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
});

// API endpoint for patient classification
app.post('/api/classify', (req, res) => {
    const patientData = req.body;
    console.log('Patient data received:', patientData);
    
    // Perform KNN classification
    const clusterId = classifyPatient(patientData);
    const cluster = clusterInfo[clusterId];
    
    console.log(`Patient classified to Cluster ${clusterId}: ${cluster.name}`);
    
    res.json({ 
        success: true, 
        clusterId: clusterId,
        clusterInfo: cluster,
        patientData: patientData
    });
});

// API endpoint to get cluster info (for doctors)
app.get('/api/clusters', (req, res) => {
    res.json({
        success: true,
        clusters: clusterInfo
    });
});

// ===== RAG Chatbot Proxy =====
const RAG_API_URL = process.env.RAG_API_URL || 'http://localhost:8000';

// Proxy chat requests to RAG API
app.post('/api/chat', async (req, res) => {
    try {
        const response = await fetch(`${RAG_API_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        
        if (!response.ok) {
            throw new Error(`RAG API error: ${response.status}`);
        }
        
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Chat API error:', error);
        res.status(500).json({ 
            error: 'Error communicating with chatbot',
            message: error.message 
        });
    }
});

// Get suggested questions
app.get('/api/chat/suggestions/:role', async (req, res) => {
    try {
        const response = await fetch(`${RAG_API_URL}/suggestions/${req.params.role}`);
        
        if (!response.ok) {
            throw new Error(`RAG API error: ${response.status}`);
        }
        
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('Suggestions API error:', error);
        // Return default suggestions if RAG is unavailable
        res.json({ 
            questions: [
                "¿Qué es el perfil molecular NSMP?",
                "¿Cuáles son los tratamientos disponibles?",
                "¿Qué significa mi clasificación de riesgo?",
                "¿Cuál es el seguimiento recomendado?",
                "¿Cuáles son los factores pronósticos?"
            ],
            role: req.params.role
        });
    }
});

// Check RAG system status
app.get('/api/chat/status', async (req, res) => {
    try {
        const response = await fetch(`${RAG_API_URL}/health`);
        const data = await response.json();
        res.json({ available: true, ...data });
    } catch (error) {
        res.json({ available: false, message: 'RAG system not available' });
    }
});

app.listen(PORT, () => {
    console.log(`🏥 Server running at http://localhost:${PORT}`);
    console.log('📋 Role selection: http://localhost:${PORT}/role-select');
    console.log(`🤖 RAG API expected at: ${RAG_API_URL}`);
    console.log('Press Ctrl+C to stop');
});
