"""
Telegram-бот для RAG-системы с техниками промптинга.

Позволяет пользователям отправлять запросы через Telegram и получать ответы
на основе векторной базы знаний.
"""

import os
import logging
import asyncio
from typing import Optional
from pathlib import Path

# Попытка загрузить переменные из .env файла
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Загружены переменные из {env_path}")
except ImportError:
    pass  # python-dotenv не обязателен

try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes,
        ConversationHandler
    )
except ImportError:
    print("Для работы Telegram-бота установите: pip install python-telegram-bot")
    raise

from rag_bot import RAGBot, RAGConfig

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальная переменная для RAG-бота
rag_bot: Optional[RAGBot] = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    welcome_message = """
🤖 Привет! Я RAG-бот с техниками промптинга.

Я могу отвечать на вопросы на основе базы знаний о вселенной Cosmic Dominion.

📝 **Как использовать:**
Просто отправь мне вопрос текстом, и я найду ответ в базе знаний!

🔧 **Доступные команды:**
/start - Начать работу
/help - Показать справку
/settings - Настройки бота
/status - Статус системы

**Примеры вопросов:**
• Кто такой Kael Starwind?
• Что такое Synth Flux?
• Расскажи о Void Core
• Кто такой Xarn Velgor?

Я знаю о вселенной **Cosmic Dominion** - галактической цивилизации с уникальными технологиями, персонажами и событиями.
"""
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    help_text = """
📚 **Справка по использованию бота:**

**Основные команды:**
/start - Начать работу с ботом
/help - Показать эту справку
/settings - Настройки (Few-shot, Chain-of-Thought)
/status - Проверить статус системы

**Как задать вопрос:**
Просто напиши свой вопрос обычным текстом. Например:
• "Кто такой Kael Starwind?"
• "Что такое Synth Flux?"
• "Расскажи о битве на Endor"

**Техники промптинга:**
• Few-shot: Использует примеры из базы знаний
• Chain-of-Thought: Пошаговое рассуждение перед ответом

**Источники:**
Бот использует векторную базу знаний и указывает источники информации.
"""
    await update.message.reply_text(help_text)


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /settings - показывает текущие настройки."""
    if rag_bot is None:
        await update.message.reply_text("❌ Бот не инициализирован. Проверьте конфигурацию.")
        return
    
    config = rag_bot.config
    settings_text = f"""
⚙️ **Текущие настройки:**

**LLM провайдер:** {config.llm_provider}
**Модель:** {config.llm_model}
**Few-shot:** {'✅ Включен' if config.use_few_shot else '❌ Выключен'}
**Chain-of-Thought:** {'✅ Включен' if config.use_chain_of_thought else '❌ Выключен'}
**Количество чанков:** {config.top_k}

**Защита от промпт-инъекций:**
• Pre-prompt: {'✅ Включена' if config.enable_pre_prompt_protection else '❌ Выключена'}
• Post-filter: {'✅ Включен' if config.enable_post_filter else '❌ Выключен'}
• Content-cleaning: {'✅ Включена' if config.enable_content_cleaning else '❌ Выключена'}

**Примечание:** Настройки можно изменить в коде или через переменные окружения.
"""
    await update.message.reply_text(settings_text)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /status - проверяет статус системы."""
    if rag_bot is None:
        await update.message.reply_text("❌ Бот не инициализирован.")
        return
    
    try:
        # Проверяем доступность векторной базы
        test_query = "test"
        docs = rag_bot.retrieve_context(test_query)
        
        status_text = f"""
✅ **Статус системы:**

**Векторная база:** ✅ Доступна
**Найдено чанков:** {len(docs)} (тестовый запрос)
**LLM провайдер:** {rag_bot.config.llm_provider}
**Модель:** {rag_bot.config.llm_model}

**Техники промптинга:**
• Few-shot: {'✅' if rag_bot.config.use_few_shot else '❌'}
• Chain-of-Thought: {'✅' if rag_bot.config.use_chain_of_thought else '❌'}

Система готова к работе! 🚀
"""
        await update.message.reply_text(status_text)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка проверки статуса: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений - основной обработчик запросов."""
    if rag_bot is None:
        await update.message.reply_text("❌ Бот не инициализирован. Проверьте конфигурацию.")
        return
    
    user_message = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "пользователь"
    
    logger.info(f"Запрос от пользователя {username} ({user_id}): {user_message}")
    
    # Отправляем индикатор печати
    await update.message.reply_chat_action(action="typing")
    
    try:
        # Обрабатываем запрос через RAG-бота
        # Используем стандартный порог релевантности (0.8)
        result = rag_bot.query(user_message, verbose=False)
        
        answer = result.get("answer", "Извините, не удалось сформировать ответ.")
        sources = result.get("sources", [])
        chunks_count = result.get("retrieved_chunks", 0)
        
        # Формируем ответное сообщение (без Markdown форматирования для избежания ошибок)
        response_text = f"{answer}\n\n"
        
        if sources:
            unique_sources = list(set(sources))
            response_text += f"📚 Источники: {', '.join(unique_sources)}\n"
        
        if chunks_count > 0:
            response_text += f"🔍 Найдено фрагментов: {chunks_count}"
        
        # Отправляем ответ без Markdown форматирования (Telegram ограничивает длину сообщения 4096 символами)
        if len(response_text) > 4000:
            # Если ответ слишком длинный, разбиваем на части
            await update.message.reply_text(response_text[:4000])
            if len(response_text) > 4000:
                await update.message.reply_text(response_text[4000:])
        else:
            # Отправляем без форматирования, чтобы избежать ошибок парсинга
            await update.message.reply_text(response_text)
        
        logger.info(f"Ответ отправлен пользователю {username}")
        
    except Exception as e:
        error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
        logger.error(f"Ошибка для пользователя {username}: {e}", exc_info=True)
        await update.message.reply_text(error_message)


def initialize_rag_bot() -> RAGBot:
    """Инициализирует RAG-бота с настройками из переменных окружения."""
    # Читаем настройки из переменных окружения
    llm_provider = os.getenv("LLM_PROVIDER", "local")
    llm_model = os.getenv("LLM_MODEL", "llama3.2")
    use_few_shot = os.getenv("USE_FEW_SHOT", "true").lower() == "true"
    use_cot = os.getenv("USE_CHAIN_OF_THOUGHT", "true").lower() == "true"
    top_k = int(os.getenv("TOP_K", "3"))
    
    # Параметры защиты (по умолчанию включены для безопасности)
    enable_pre_prompt = os.getenv("ENABLE_PRE_PROMPT_PROTECTION", "true").lower() == "true"
    enable_post_filter = os.getenv("ENABLE_POST_FILTER", "true").lower() == "true"
    enable_content_cleaning = os.getenv("ENABLE_CONTENT_CLEANING", "true").lower() == "true"
    
    config = RAGConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_few_shot=use_few_shot,
        use_chain_of_thought=use_cot,
        top_k=top_k,
        enable_pre_prompt_protection=enable_pre_prompt,
        enable_post_filter=enable_post_filter,
        enable_content_cleaning=enable_content_cleaning
    )
    
    logger.info(f"Инициализация RAG-бота с настройками:")
    logger.info(f"  Провайдер: {llm_provider}, Модель: {llm_model}")
    logger.info(f"  Few-shot: {use_few_shot}, CoT: {use_cot}")
    logger.info(f"  Защита: Pre-prompt={enable_pre_prompt}, Post-filter={enable_post_filter}, Content-cleaning={enable_content_cleaning}")
    
    return RAGBot(config)


def main() -> None:
    """Основная функция для запуска Telegram-бота."""
    # Получаем токен бота из переменных окружения
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN не установлен!")
        logger.error("Установите переменную окружения: export TELEGRAM_BOT_TOKEN='ваш-токен'")
        logger.error("Или создайте файл .env с TELEGRAM_BOT_TOKEN=ваш-токен")
        return
    
    # Инициализируем RAG-бота
    global rag_bot
    try:
        rag_bot = initialize_rag_bot()
        logger.info("RAG-бот успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации RAG-бота: {e}", exc_info=True)
        return
    
    # Создаем приложение Telegram
    application = Application.builder().token(bot_token).build()
    
    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("status", status))
    
    # Регистрируем обработчик текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запускаем бота
    logger.info("Запуск Telegram-бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
