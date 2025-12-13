const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/calculator', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'calculator.html'));
});

// API endpoint for form submission
app.post('/api/profile', (req, res) => {
    const profileData = req.body;
    console.log('Profile received:', profileData);
    res.json({ 
        success: true, 
        message: 'Perfil recibido correctamente',
        data: profileData 
    });
});

app.listen(PORT, () => {
    console.log(`🏥 Server running at http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop');
});
