@echo off
echo ========================================
echo HSN RAG System - Run Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "hsn_env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call hsn_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo SUCCESS: Virtual environment activated
echo.

REM Check if required files exist
if not exist "output\vectors\hsn_vector_config.json" (
    echo ERROR: Vector store not found. Please run setup.bat first.
    pause
    exit /b 1
)

if not exist "output\models\hsn_knowledge_graph.pkl" (
    echo ERROR: Knowledge graph not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Run Streamlit app
echo Starting HSN RAG System web interface...
echo.
echo The web interface will open in your default browser.
echo You can also access it at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

streamlit run app.py

REM Deactivate environment when done
call hsn_env\Scripts\deactivate.bat

echo.
echo HSN RAG System stopped.
pause