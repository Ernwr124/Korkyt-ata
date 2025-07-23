@echo off
chcp 65001 > nul
title Установка бота Коркыт Ата

echo.
echo #######################################################
echo #       Установка бота Коркыт Ата - начало            #
echo #######################################################
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo !!! ОШИБКА: Python не установлен или не добавлен в PATH
    echo Пожалуйста, установите Python 3.8+ с сайта python.org
    echo И убедитесь, что выбрали опцию "Add Python to PATH"
    pause
    exit /b 1
)

REM Проверка версии Python
for /f "tokens=2 delims= " %%A in ('python --version 2^>^&1') do set "python_version=%%A"
for /f "tokens=1,2 delims=." %%A in ("%python_version%") do (
    if %%A LSS 3 (
        echo.
        echo !!! ОШИБКА: Требуется Python 3.8 или выше (у вас версия %python_version%)
        pause
        exit /b 1
    )
    if %%A EQU 3 if %%B LSS 8 (
        echo.
        echo !!! ОШИБКА: Требуется Python 3.8 или выше (у вас версия %python_version%)
        pause
        exit /b 1
    )
)

echo.
echo === Установка Python библиотек ===
pip install --upgrade pip --quiet
pip install flask faiss-cpu sentence-transformers numpy --quiet
if %errorlevel% neq 0 (
    echo.
    echo !!! ОШИБКА при установке Python библиотек
    echo Попробуйте установить вручную: 
    echo pip install flask faiss-cpu sentence-transformers numpy
    pause
    exit /b 1
)

echo.
echo === Проверка установки Ollama ===
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Ollama не установлен. Пытаюсь установить автоматически...
    
    echo Скачивание Ollama...
    curl -L -o OllamaSetup.exe https://ollama.ai/download/OllamaSetup.exe
    if exist OllamaSetup.exe (
        echo Установка Ollama...
        start /wait "" OllamaSetup.exe /S
        timeout /t 30 > nul
        del OllamaSetup.exe
        echo Добавление Ollama в PATH...
        setx PATH "%PATH%;%ProgramFiles%\Ollama" /m
        echo Ожидание запуска службы Ollama...
        timeout /t 10 > nul
    ) else (
        echo.
        echo !!! Не удалось скачать Ollama
        echo Пожалуйста, установите вручную с https://ollama.ai/download
        pause
    )
)

echo.
echo === Установка модели didustin/kazllm:8b ===
ollama pull didustin/kazllm:8b
if %errorlevel% neq 0 (
    echo.
    echo !!! ОШИБКА при загрузке модели
    echo Попробуйте вручную: ollama pull didustin/kazllm:8b
    pause
)

echo.
echo === Создание структуры папок ===
if not exist data mkdir data
if not exist logs mkdir logs

echo.
echo #######################################################
echo #       Установка успешно завершена!                  #
echo #                                                    #
echo # 1. Поместите файл full_text.txt в:                 #
echo #    - папку с ботом ИЛИ                             #
echo #    - папку data/                                   #
echo #                                                    #
echo # 2. Запустите бота командой:                        #
echo #    python Korkyt.py                                #
echo #                                                    #
echo # 3. Откройте в браузере:                            #
echo #    http://localhost:5000                           #
echo #                                                    #
echo # Логи работы будут сохраняться в папке logs/        #
echo #######################################################
echo.

timeout /t 30