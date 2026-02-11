#!/bin/bash
# Скрипт для запуска Telegram-бота

cd "$(dirname "$0")"
cd ../Task3
source venv/bin/activate
cd ../Task4

export TELEGRAM_BOT_TOKEN="8144660959:AAEuXu-DE9xrGH__I8n-rDXTlMawLhhnb_4"
export LLM_PROVIDER="local"
export LLM_MODEL="llama3.2"
export USE_FEW_SHOT="true"
export USE_CHAIN_OF_THOUGHT="true"
export TOP_K="3"

echo "🚀 Запуск Telegram-бота..."
echo "📝 Токен установлен"
echo "🤖 Провайдер: $LLM_PROVIDER"
echo "📊 Модель: $LLM_MODEL"
echo ""

python telegram_bot.py
