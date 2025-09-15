@echo off
echo ========================================
echo HSN RAG System Setup Script
echo ========================================
echo.

REM Create virtual environment
echo 1. Creating virtual environment...
python -m venv hsn_env
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo SUCCESS: Virtual environment created
echo.

REM Activate virtual environment
echo 2. Activating virtual environment...
call hsn_env\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo SUCCESS: Virtual environment activated
echo.

REM Upgrade pip
echo 3. Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo WARNING: Failed to upgrade pip, continuing...
)
echo.

REM Install requirements
echo 4. Installing requirements...
where uv >nul 2>nul
if %errorlevel% equ 0 (
    echo INFO: Using uv package manager for faster installation...
    uv pip install -r ../requirements.txt
    if %errorlevel% neq 0 (
        echo WARNING: uv failed, falling back to pip...
        pip install -r ../requirements.txt
        if %errorlevel% neq 0 (
            echo ERROR: Failed to install requirements with both uv and pip
            pause
            exit /b 1
        )
    )
) else (
    echo INFO: uv not found, using pip...
    pip install -r ../requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)
echo SUCCESS: Requirements installed
echo.

REM Check if data extraction is needed
echo 5. Checking data extraction...
if exist "..\output\data\extraction_complete.csv" (
    echo INFO: Data extraction already completed, skipping...
) else (
    echo INFO: Running data extraction...
    python ..\src\data_processing\test_extraction.py
    if %errorlevel% neq 0 (
        echo ERROR: Data extraction failed
        pause
        exit /b 1
    )
    echo SUCCESS: Data extraction completed
)
echo.

REM Check if data enhancement is needed
echo 6. Checking data enhancement...
if exist "..\output\data\sample_enhanced_data.csv" (
    echo INFO: Data enhancement already completed, skipping...
) else (
    echo INFO: Running data enhancement...
    python ..\src\data_processing\test_enhancement.py
    if %errorlevel% neq 0 (
        echo ERROR: Data enhancement failed
        pause
        exit /b 1
    )
    echo SUCCESS: Data enhancement completed
)
echo.

REM Check if vector store needs to be built
echo 7. Checking vector store...
if exist "..\output\vectors\hsn_vector_config.json" (
    echo INFO: Vector store already exists, skipping...
) else (
    echo INFO: Building vector store...
    python ..\src\rag_system\vector_store.py
    if %errorlevel% neq 0 (
        echo ERROR: Vector store creation failed
        pause
        exit /b 1
    )
    echo SUCCESS: Vector store created
)
echo.

REM Check if graph schema needs to be built
echo 8. Checking graph schema...
if exist "..\output\data\graph_schema.json" (
    echo INFO: Graph schema already exists, skipping design phase...
) else (
    echo INFO: Building graph schema, nodes, and relationships...
    python ..\src\knowledge_graph\graph_design.py
    if %errorlevel% neq 0 (
        echo ERROR: Graph schema creation failed
        pause
        exit /b 1
    )
    echo SUCCESS: Graph schema, nodes, and relationships created
)

REM Check if knowledge graph needs to be built
echo 9. Checking knowledge graph...
if exist "..\output\models\hsn_knowledge_graph.pkl" (
    echo INFO: Knowledge graph already exists, skipping...
) else (
    echo INFO: Building knowledge graph from schema...
    python ..\src\knowledge_graph\graph_implementation.py
    if %errorlevel% neq 0 (
        echo ERROR: Knowledge graph creation failed
        pause
        exit /b 1
    )
    echo SUCCESS: Knowledge graph created
)
echo.

REM Test system initialization
echo 10. Testing system initialization...
python -c "
from src.rag_system.hsn_rag_system import HSN_RAG_System
system = HSN_RAG_System()
if system.initialize_system():
    print('SUCCESS: System initialization test passed')
    metrics = system.get_system_metrics()
    print(f'  Total queries: {metrics[\"total_queries\"]}')
    print(f'  Success rate: {metrics.get(\"success_rate\", 0):.1%}')
else:
    print('ERROR: System initialization test failed')
    exit(1)
"
if %errorlevel% neq 0 (
    echo ERROR: System initialization test failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Your HSN RAG System is now ready!
echo.
echo To run the system:
echo 1. Activate the environment: call hsn_env\Scripts\activate.bat
echo 2. Run the web app: streamlit run ../app.py
echo    OR run CLI: python ../main.py
echo.
echo For inference and testing, use run.bat
echo.
pause