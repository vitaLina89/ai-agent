# Настройка Telegram-бота для RAG-системы

## Описание

Telegram-бот позволяет пользователям отправлять запросы через Telegram и получать ответы на основе векторной базы знаний с применением техник промптинга (Few-shot и Chain-of-Thought).

**Имя бота:** [@MyStarWarsBot](https://t.me/MyStarWarsBot)

## Установка

### 1. Установка зависимостей

```bash
cd Task4
pip install -r requirements.txt
```

### 2. Создание Telegram-бота

1. Откройте Telegram и найдите [@BotFather](https://t.me/BotFather)
2. Отправьте команду `/newbot`
3. Следуйте инструкциям:
   - Введите имя бота (например: "RAG Knowledge Bot")
   - Введите username бота (должен заканчиваться на `bot`, например: `rag_knowledge_bot`)
4. BotFather предоставит вам **токен бота** (выглядит как: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 3. Настройка переменных окружения

Создайте файл `.env` в папке `Task4/` или установите переменные окружения:

```bash
# Обязательно: токен Telegram-бота
export TELEGRAM_BOT_TOKEN="ваш-токен-от-BotFather"

# Опционально: настройки RAG-бота
export LLM_PROVIDER="local"  # local, openai, yandexgpt
export LLM_MODEL="llama3.2"  # llama3.2, gpt-3.5-turbo, и т.д.
export USE_FEW_SHOT="true"    # true или false
export USE_CHAIN_OF_THOUGHT="true"  # true или false
export TOP_K="3"              # Количество релевантных чанков
```

**Или создайте файл `.env`:**
```bash
# .env
TELEGRAM_BOT_TOKEN=ваш-токен-от-BotFather
LLM_PROVIDER=local
LLM_MODEL=llama3.2
USE_FEW_SHOT=true
USE_CHAIN_OF_THOUGHT=true
TOP_K=3
```

**Важно:** Добавьте `.env` в `.gitignore`, чтобы не коммитить токен!

### 4. Для локальной LLM (Ollama)

Если используете локальную модель через Ollama:

```bash
# Убедитесь, что Ollama запущен
brew services start ollama  # или ollama serve

# Убедитесь, что модель загружена
ollama pull llama3.2
```

### 5. Для OpenAI

Если используете OpenAI:

```bash
export OPENAI_API_KEY="ваш-api-ключ"
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-3.5-turbo"
```

## Запуск бота

### Базовый запуск

```bash
cd Task4
python telegram_bot.py
```

### С виртуальным окружением (если используете из Task3)

```bash
cd Task3
source venv/bin/activate
cd ../Task4
python telegram_bot.py
```

### Запуск в фоне (Linux/Mac)

```bash
nohup python telegram_bot.py > bot.log 2>&1 &
```

### Запуск через systemd (Linux)

Создайте файл `/etc/systemd/system/rag-telegram-bot.service`:

```ini
[Unit]
Description=RAG Telegram Bot
After=network.target

[Service]
Type=simple
User=ваш-пользователь
WorkingDirectory=/путь/к/Task4
Environment="TELEGRAM_BOT_TOKEN=ваш-токен"
Environment="LLM_PROVIDER=local"
Environment="LLM_MODEL=llama3.2"
ExecStart=/usr/bin/python3 telegram_bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Затем:
```bash
sudo systemctl enable rag-telegram-bot
sudo systemctl start rag-telegram-bot
sudo systemctl status rag-telegram-bot
```

## Использование

### Команды бота

- `/start` - Начать работу с ботом
- `/help` - Показать справку
- `/settings` - Показать текущие настройки
- `/status` - Проверить статус системы

### Отправка запросов

Просто отправьте боту вопрос текстом:

```
Кто такой Kael Starwind?
```

```
Что такое Synth Flux?
```

```
Расскажи о Void Core
```

Бот ответит на основе базы знаний и укажет источники информации.

## Примеры использования

### Пример 1: Простой запрос

**Пользователь:** `Кто такой Kael Starwind?`

**Бот:**
```
Kael Starwind - это главный герой истории, который был героем Галактической гражданской войны. Он обладал способностями к Synth Flux и играл важную роль в борьбе против Cosmic Dominion.

📚 Источники: Luke_Skywalker
🔍 Найдено фрагментов: 3
```

### Пример 2: Запрос с CoT

**Пользователь:** `Что такое Void Core?`

**Бот:**
```
## Шаги рассуждения:

1. Сначала определю, какая информация нужна для ответа на вопрос.
2. Затем найду релевантные фрагменты в предоставленном контексте.
3. Проанализирую найденную информацию и сформулирую ответ.

Void Core - это гигантская космическая станция, вооруженная планетарным оружием...

📚 Источники: Death_Star, Battle_of_Yavin
🔍 Найдено фрагментов: 3
```

## Настройка

### Изменение провайдера LLM

```bash
# Локальная модель (Ollama)
export LLM_PROVIDER="local"
export LLM_MODEL="llama3.2"

# OpenAI
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-3.5-turbo"
export OPENAI_API_KEY="ваш-ключ"
```

### Включение/выключение техник промптинга

```bash
# Включить Few-shot
export USE_FEW_SHOT="true"

# Выключить Chain-of-Thought
export USE_CHAIN_OF_THOUGHT="false"
```

### Изменение количества чанков

```bash
export TOP_K="5"  # Больше контекста для ответов
```

## Устранение неполадок

### Бот не отвечает

1. Проверьте, что токен установлен: `echo $TELEGRAM_BOT_TOKEN`
2. Проверьте логи бота
3. Убедитесь, что бот запущен: `ps aux | grep telegram_bot.py`

### Ошибка инициализации RAG-бота

1. Убедитесь, что векторная база существует: `ls ../Task3/chroma_db/`
2. Проверьте, что модель эмбеддингов доступна
3. Для локальной LLM: убедитесь, что Ollama запущен

### Ошибка подключения к Ollama

```bash
# Проверьте, что Ollama запущен
curl http://localhost:11434/api/tags

# Если не запущен:
brew services start ollama
# или
ollama serve
```

### Ошибка с OpenAI API

1. Проверьте API ключ: `echo $OPENAI_API_KEY`
2. Убедитесь, что на счету есть кредиты
3. Проверьте лимиты API

## Безопасность

1. **Никогда не коммитьте токен в Git!**
   - Добавьте `.env` в `.gitignore`
   - Используйте переменные окружения

2. **Ограничение доступа:**
   - Можно добавить проверку user_id для ограничения доступа
   - Использовать whitelist пользователей

3. **Rate limiting:**
   - Telegram API имеет ограничения на количество запросов
   - Бот автоматически обрабатывает это

## Мониторинг

### Логи

Бот пишет логи в консоль. Для сохранения в файл:

```bash
python telegram_bot.py > bot.log 2>&1
```

### Статистика

Используйте команду `/status` для проверки состояния системы.

## Расширение функциональности

Можно добавить:
- История диалогов
- Кэширование ответов
- Аналитика запросов
- Мультиязычная поддержка
- Голосовые сообщения
- Изображения и документы

---

**Готово!** Теперь у вас есть работающий Telegram-бот для RAG-системы! 🚀
