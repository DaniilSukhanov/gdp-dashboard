import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import os

# Путь для сохранения модели (будет создана при первом запуске)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Кэшируем загрузку модели, чтобы не перекачивать при каждом обновлении страницы
@st.cache_resource
def load_model():
    # Скачиваем GGUF-файл с Hugging Face (если ещё не скачан)
    model_path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",  # 4‑битная квантизация
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    # Загружаем модель в память
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,           # длина контекста
        n_threads=2,          # используем 2 ядра CPU
        n_gpu_layers=0,       # 0 = только CPU
        verbose=False,
    )
    return llm

# Загружаем модель (один раз)
llm = load_model()

# Интерфейс чата
st.set_page_config(page_title="Мини-чат", page_icon="🤖")
st.title("🤖 Мини-чат с Qwen 0.5B")

# Инициализация истории сообщений
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Привет! Я маленькая LLM. Задай мне вопрос."}]

# Отображаем все сообщения
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Поле ввода
if prompt := st.chat_input("Ваш вопрос..."):
    # Добавляем сообщение пользователя
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Генерируем ответ
    with st.spinner("Думаю..."):
        # Формируем промпт для instruct-модели
        # Qwen2.5 использует шаблон: <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        full_prompt = ""
        for m in st.session_state.messages:
            if m["role"] == "user":
                full_prompt += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
            elif m["role"] == "assistant":
                full_prompt += f"<|im_start|>assistant\n{m['content']}<|im_end|>\n"
        full_prompt += "<|im_start|>assistant\n"

        # Генерация
        response = llm(
            full_prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|im_end|>"],
        )
        answer = response["choices"][0]["text"].strip()

    # Добавляем ответ ассистента
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})