import os
import faiss
import subprocess
from sentence_transformers import SentenceTransformer
import numpy as np
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# Конфигурация
system_prompt = (
    "[Қорқыт Ата Рөлі]"
    "Сен - Қорқыт Атасың. Жауабың мынандай болуы тиіс:"
    "1. Қазақтың дәстүрлі даналық стилінде (3-5 сөйлем)"
    "2. Контекстке сілтеме жасау (егер қажет болса)"
    "3. Философиялық тереңдік пен нақылдық"
)

def find_text_file():
    # Ищем файл в нескольких возможных местах
    possible_paths = [
        "full_text.txt",  # Текущая директория
        "data/full_text.txt",  # Поддиректория data
        os.path.expanduser("~/Downloads/full_text.txt"),  # Папка Загрузки пользователя
        os.path.join(os.path.dirname(__file__), "full_text.txt"),  # Рядом со скриптом
        os.path.join(os.path.dirname(__file__), "data", "full_text.txt")  # В data рядом со скриптом
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_and_chunk(file_path, chunk_size=500):
    chunks = []
    if not file_path:
        return chunks
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                chunks.append(chunk)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {str(e)}")
    return chunks

def build_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    if not chunks:
        raise ValueError("Нет текстовых данных для индексации")
        
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks, model

def search(query, index, chunks, model, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]

def generate_with_qwen3(context, question):
    prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    try:
        result = subprocess.run(
            ["ollama", "run", "didustin/kazllm:8b"],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.stdout.decode('utf-8')
    except Exception as e:
        print(f"Ошибка при генерации ответа: {str(e)}")
        return "Жауап құрастыру мүмкін емес. Қайталап көріңіз."

# Инициализация
print("📚 Подготовка бота Коркыт Ата...")
text_file = find_text_file()

if text_file:
    print(f"🔍 Найден файл с данными: {text_file}")
    chunks = load_and_chunk(text_file)
    if chunks:
        try:
            index, chunk_store, model = build_faiss_index(chunks)
            print("🤖 Индекс успешно построен! Бот готов к работе.")
            bot_ready = True
        except Exception as e:
            print(f"⚠️ Ошибка при построении индекса: {str(e)}")
            bot_ready = False
    else:
        print("⚠️ Не удалось загрузить данные из файла")
        bot_ready = False
else:
    print("⚠️ Файл full_text.txt не найден. Бот будет работать без базы знаний.")
    bot_ready = False

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html lang="kk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Коркыт аты - Қазақ мәдениеті бот</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Noto Sans', sans-serif;
            background-color: #f8fafc;
        }
        .typing-animation::after {
            content: '...';
            animation: typing 1.5s infinite;
        }
        @keyframes typing {
            0% { content: '.'; }
            33% { content: '..'; }
            66% { content: '...'; }
        }
        .bot-message {
            white-space: pre-wrap; /* Сохраняет форматирование текста */
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen flex flex-col">
    <div class="container mx-auto px-4 py-8 flex-grow flex flex-col max-w-3xl">
        <div class="text-center mb-8">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Gorkut_Ata_Statue_at_Independence_Monument_%2827506806397%29.jpg/1024px-Gorkut_Ata_Statue_at_Independence_Monument_%2827506806397%29.jpg" 
                 alt="Коркыт ата" 
                 class="w-32 h-32 rounded-full mx-auto mb-4 border-4 border-yellow-500 shadow-lg object-cover">
            <h1 class="text-3xl font-bold text-gray-800">Коркыт аты</h1>
            <p class="text-gray-600 mt-2">Қазақ мәдениеті мен философиясы боты</p>
        </div>
        
        <div id="chat-container" class="flex-grow bg-white rounded-lg shadow-md p-4 mb-4 overflow-y-auto space-y-4">
            <div class="text-center text-gray-500 py-8">
                Сәлеметсіз бе! Мен сізге қазақ мәдениеті мен философиясы туралы ақпарат бере аламын.
            </div>
        </div>
        
        <div class="bg-white rounded-lg shadow-md p-4">
            <form id="chat-form" class="flex space-x-2">
                <input 
                    type="text" 
                    id="user-input" 
                    placeholder="Сұрақ қойыңыз..." 
                    class="flex-grow px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500"
                    autocomplete="off"
                >
                <button 
                    type="submit" 
                    class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold px-6 py-2 rounded-lg transition duration-200"
                >
                    Жіберу
                </button>
            </form>
        </div>
    </div>
    
    <script>
        document.getElementById('chat-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const input = document.getElementById('user-input');
            const question = input.value.trim();
            
            if (!question) return;
            
            input.value = '';
            addMessage(question, 'user');
            const loadingId = addLoadingMessage();
            
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
            .then(data => {
                removeLoadingMessage(loadingId);
                addMessage(data.answer, 'bot');
            })
            .catch(error => {
                removeLoadingMessage(loadingId);
                addMessage('Қате орын алды. Қайталап көріңіз.', 'bot');
                console.error('Error:', error);
            });
        });
        
        function addMessage(text, sender) {
            const chatContainer = document.getElementById('chat-container');
            const messageDiv = document.createElement('div');
            
            if (sender === 'user') {
                messageDiv.className = 'bg-yellow-100 text-gray-800 p-3 rounded-lg ml-12 self-end';
                messageDiv.innerHTML = `<p>${text}</p>`;
            } else {
                messageDiv.className = 'bg-gray-100 text-gray-800 p-3 rounded-lg mr-12';
                messageDiv.innerHTML = `
                    <p class="font-semibold">Коркыт ата:</p>
                    <p class="bot-message">${text}</p>
                `;
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function addLoadingMessage() {
            const chatContainer = document.getElementById('chat-container');
            const loadingDiv = document.createElement('div');
            const loadingId = 'loading-' + Date.now();
            
            loadingDiv.id = loadingId;
            loadingDiv.className = 'bg-gray-100 text-gray-800 p-3 rounded-lg mr-12';
            loadingDiv.innerHTML = '<p class="font-semibold">Коркыт ата:</p><p class="typing-animation">Жауап дайындалуда</p>';
            
            chatContainer.appendChild(loadingDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return loadingId;
        }
        
        function removeLoadingMessage(id) {
            const loadingElement = document.getElementById(id);
            if (loadingElement) {
                loadingElement.remove();
            }
        }
    </script>
</body>
</html>
'''

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data['question']
    
    if not query or query.lower() in ['exit', 'quit']:
        return jsonify({'answer': 'Сау болыңыз! Көріскенше!'})
    
    try:
        time.sleep(1)  # Имитация задержки
        
        if not bot_ready:
            return jsonify({
                'answer': 'Кешіріңіз, бот толық дайын емес. Текст файлы табылмады немесе индекстеу мүмкін емес.\n\n' +
                         'Бірақ мен сіздің сұрақтарыңызға жауап бере аламын жалпы білімге сүйене отырып.'
            })
        
        top_chunks = search(query, index, chunk_store, model)
        context = "\n---\n".join(top_chunks)
        answer = generate_with_qwen3(context, query)
        
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'answer': f'Қате орын алды: {str(e)}\n\nҚайталап көріңіз.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)