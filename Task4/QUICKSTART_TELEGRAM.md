# Быстрый запуск Telegram-бота

## 1. Создайте бота в Telegram

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Введите имя и username бота
4. Скопируйте токен (выглядит как `123456789:ABCdef...`)

## 2. Установите зависимости

```bash
cd Task4
pip install -r requirements.txt
```

## 3. Настройте переменные окружения

```bash
# Установите токен бота
export TELEGRAM_BOT_TOKEN="ваш-токен-от-BotFather"

# Настройки RAG (опционально)
export LLM_PROVIDER="local"
export LLM_MODEL="llama3.2"
```

Или создайте файл `.env`:
```
TELEGRAM_BOT_TOKEN=ваш-токен
LLM_PROVIDER=local
LLM_MODEL=llama3.2
```

## 4. Запустите бота

```bash
# С виртуальным окружением (если используете)
cd Task3
source venv/bin/activate
cd ../Task4

# Запуск
python telegram_bot.py
```

## 5. Используйте бота

1. Найдите вашего бота в Telegram по username
2. Отправьте `/start`
3. Задайте вопрос, например: "Кто такой Kael Starwind?"

## Команды бота

- `/start` - Начать работу
- `/help` - Справка
- `/settings` - Настройки
- `/status` - Статус системы

Готово! 🚀
