@echo off
echo ========================================
echo NSMP Cancer RAG System - Startup Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado. Por favor instala Python 3.9+
    pause
    exit /b 1
)

REM Navigate to rag_system directory
cd /d "%~dp0"

REM Check if data directory exists (system initialized)
if not exist "data\faiss_index.bin" (
    echo [1/3] Primera ejecucion - Inicializando sistema RAG...
    echo Esto puede tardar unos minutos...
    echo.
    python initialize.py
    if errorlevel 1 (
        echo ERROR: Fallo la inicializacion
        pause
        exit /b 1
    )
) else (
    echo [1/3] Sistema RAG ya inicializado
)

echo.
echo [2/3] Iniciando API RAG en puerto 8000...
start "RAG API" cmd /c "python api.py"

REM Wait for API to start
timeout /t 3 /nobreak >nul

echo.
echo [3/3] Iniciando servidor web en puerto 3000...
cd ..\web
start "Web Server" cmd /c "node server.js"

echo.
echo ========================================
echo Sistema iniciado correctamente!
echo ========================================
echo.
echo API RAG: http://localhost:8000
echo Web App: http://localhost:3000
echo Documentacion API: http://localhost:8000/docs
echo.
echo Presiona cualquier tecla para abrir la aplicacion...
pause >nul

start http://localhost:3000
