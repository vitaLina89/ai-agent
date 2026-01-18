"""
RAG-бот с техниками промптинга (Few-shot и Chain-of-Thought).

Использует векторную базу знаний из Task3 для поиска релевантных фрагментов
и формирует ответы с применением продвинутых техник промптинга.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Добавляем родительскую директорию в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document


@dataclass
class RAGConfig:
    """Конфигурация RAG-бота."""
    # Пути
    chroma_db_path: str = "../Task3/chroma_db"
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Параметры поиска
    top_k: int = 3  # Количество релевантных чанков для извлечения
    
    # Параметры LLM
    llm_provider: str = "openai"  # openai, yandexgpt, local
    llm_model: str = "gpt-3.5-turbo"  # Для local: "llama2", "mistral", "llama3.2" и т.д.
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Техники промптинга
    use_few_shot: bool = True
    use_chain_of_thought: bool = True


class LLMProvider:
    """Базовый класс для провайдеров LLM."""
    
    def generate(self, prompt: str, config: RAGConfig) -> str:
        """Генерирует ответ на основе промпта."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """Провайдер для OpenAI API."""
    
    def __init__(self):
        try:
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Для использования OpenAI установите: pip install openai")
        except Exception as e:
            raise ValueError(f"Ошибка инициализации OpenAI: {e}")
    
    def generate(self, prompt: str, config: RAGConfig) -> str:
        """Генерирует ответ через OpenAI API."""
        try:
            # Определяем системный промпт в зависимости от использования CoT
            system_content = "Ты полезный ассистент, который отвечает на вопросы на основе предоставленного контекста."
            if config.use_chain_of_thought:
                system_content = "Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги рассуждения перед ответом."
            
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"


class YandexGPTProvider(LLMProvider):
    """Провайдер для YandexGPT API."""
    
    def __init__(self):
        try:
            import yandexcloud
            from yandex.cloud.ai.foundation_models.v1.foundation_models_pb2 import CompletionRequest
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import FoundationModelsServiceStub
            
            # YandexGPT требует Yandex Cloud SDK
            self.api_key = os.getenv("YANDEX_API_KEY")
            self.folder_id = os.getenv("YANDEX_FOLDER_ID")
            if not self.api_key or not self.folder_id:
                raise ValueError("Установите YANDEX_API_KEY и YANDEX_FOLDER_ID")
        except ImportError:
            raise ImportError("Для использования YandexGPT установите Yandex Cloud SDK")
    
    def generate(self, prompt: str, config: RAGConfig) -> str:
        """Генерирует ответ через YandexGPT API."""
        # Упрощенная реализация - в реальности нужна полная настройка SDK
        return "YandexGPT провайдер требует дополнительной настройки SDK. Используйте OpenAI для демонстрации."


class LocalLLMProvider(LLMProvider):
    """Провайдер для локальной LLM через Ollama."""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self._check_ollama()
    
    def _check_ollama(self):
        """Проверяет доступность Ollama."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                print(f"✅ Ollama доступен. Доступные модели: {', '.join(model_names) if model_names else 'нет моделей'}")
                
                # Проверяем, есть ли нужная модель
                if not any(self.model_name in name for name in model_names):
                    print(f"⚠️  Модель '{self.model_name}' не найдена. Доступные модели: {', '.join(model_names)}")
                    print(f"💡 Установите модель: ollama pull {self.model_name}")
            else:
                print("⚠️  Ollama запущен, но не отвечает корректно")
        except requests.exceptions.ConnectionError:
            print("❌ Ollama не запущен. Установите Ollama с https://ollama.ai/")
            print("💡 После установки запустите модель: ollama pull llama2")
        except ImportError:
            print("⚠️  Для работы с Ollama установите requests: pip install requests")
        except Exception as e:
            print(f"⚠️  Не удалось проверить Ollama: {str(e)}")
    
    def generate(self, prompt: str, config: RAGConfig) -> str:
        """Генерирует ответ через Ollama API (chat endpoint)."""
        try:
            import requests
            
            model = config.llm_model if hasattr(config, 'llm_model') else self.model_name
            
            # Определяем системный промпт в зависимости от использования CoT
            system_prompt = "Ты полезный ассистент, который отвечает на вопросы на основе предоставленного контекста. ВАЖНО: Всегда отвечай полностью на русском языке. Используй только русский язык в ответах, не смешивай с английским."
            
            # Если используется CoT, добавляем инструкцию о пошаговом рассуждении
            if hasattr(config, 'use_chain_of_thought') and config.use_chain_of_thought:
                system_prompt = "Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги рассуждения перед ответом. ВАЖНО: Всегда отвечай полностью на русском языке. Используй только русский язык в ответах, не смешивай с английским."
            
            # Используем chat API для лучшего форматирования
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens
                    }
                },
                timeout=120  # 2 минуты для генерации
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                content = message.get("content", "")
                
                if not content:
                    return "Ошибка: пустой ответ от модели"
                
                # Постобработка: пытаемся очистить ответ от английских слов
                # (в будущем можно добавить более продвинутую обработку)
                cleaned_content = content.strip()
                
                return cleaned_content
            else:
                error_msg = response.text
                return f"Ошибка Ollama API ({response.status_code}): {error_msg}"
                
        except requests.exceptions.ConnectionError:
            return "Ошибка: Ollama не запущен. Убедитесь, что Ollama работает: ollama serve"
        except requests.exceptions.Timeout:
            return "Ошибка: Превышено время ожидания ответа от Ollama. Попробуйте уменьшить max_tokens или использовать меньшую модель."
        except ImportError:
            return "Ошибка: Установите requests: pip install requests"
        except Exception as e:
            return f"Ошибка локальной LLM: {str(e)}"


class RAGBot:
    """RAG-бот с техниками промптинга."""
    
    def __init__(self, config: RAGConfig = None):
        """Инициализация RAG-бота."""
        self.config = config or RAGConfig()
        
        # Загрузка модели эмбеддингов (та же, что в Task3)
        print(f"Загрузка модели эмбеддингов: {self.config.embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Загрузка векторной базы из Task3
        print(f"Загрузка векторной базы из {self.config.chroma_db_path}...")
        self.vectorstore = Chroma(
            persist_directory=self.config.chroma_db_path,
            embedding_function=self.embeddings
        )
        print("Векторная база загружена успешно!")
        
        # Инициализация LLM провайдера
        self.llm = self._init_llm()
        
        # Few-shot примеры будут извлекаться из базы при необходимости
        self.few_shot_examples = None
    
    def _init_llm(self) -> LLMProvider:
        """Инициализирует провайдер LLM."""
        provider = self.config.llm_provider.lower()
        
        if provider == "openai":
            return OpenAIProvider()
        elif provider == "yandexgpt":
            return YandexGPTProvider()
        elif provider == "local":
            # Используем модель из конфига, по умолчанию llama2
            model_name = self.config.llm_model if self.config.llm_model != "gpt-3.5-turbo" else "llama2"
            return LocalLLMProvider(model_name=model_name)
        else:
            raise ValueError(f"Неизвестный провайдер LLM: {provider}")
    
    def _extract_few_shot_examples_from_db(self, num_examples: int = 2) -> List[Dict[str, str]]:
        """
        Извлекает Few-shot примеры из векторной базы знаний.
        
        Примеры включают реальные вопросы о сущностях из базы и соответствующие
        фрагменты текста из базы знаний.
        
        Args:
            num_examples: Количество примеров для извлечения
            
        Returns:
            Список словарей с ключами: query, context, answer
        """
        # Определяем примеры вопросов на основе реальных сущностей из базы
        example_queries = [
            "Кто такой Kael Starwind?",
            "Что такое Synth Flux?",
            "Расскажи о Void Core",
            "Кто такой Xarn Velgor?",
            "Что такое Millennium Falcon?"
        ]
        
        examples = []
        
        for query in example_queries[:num_examples]:
            # Извлекаем релевантные фрагменты из базы для каждого вопроса
            # Используем более мягкий порог для примеров
            docs = self.retrieve_context(query, min_score=0.9)
            
            if not docs:
                continue
            
            # Форматируем контекст из найденных фрагментов
            context_parts = []
            for doc in docs[:2]:  # Берем первые 2 наиболее релевантных чанка
                source = doc.metadata.get('file_name', 'Unknown')
                content = doc.page_content.strip()[:300]  # Первые 300 символов
                context_parts.append(f"[Из {source}]: {content}")
            
            context = "\n".join(context_parts)
            
            # Формируем ответ на основе контекста
            # Используем начало первого релевантного чанка как основу для ответа
            if docs:
                first_doc_content = docs[0].page_content.strip()
                # Формируем краткий ответ на основе контекста
                # Для вопросов типа "Кто такой X" используем первые предложения
                answer = self._generate_answer_from_context(query, docs)
            else:
                answer = ""
            
            examples.append({
                "query": query,
                "context": context,
                "answer": answer
            })
        
        return examples
    
    def _generate_answer_from_context(self, query: str, documents: List[Document]) -> str:
        """
        Генерирует краткий ответ для Few-shot примера на основе контекста.
        
        Это упрощенная версия, которая формирует ответ на основе найденных
        фрагментов без вызова LLM (для быстрого формирования примеров).
        Извлекает осмысленные фрагменты текста из базы знаний.
        """
        if not documents:
            return ""
        
        # Объединяем содержимое всех релевантных документов
        all_content = []
        for doc in documents[:2]:  # Берем до 2 документов
            content = doc.page_content.strip()
            # Берем первые несколько предложений, которые выглядят осмысленно
            sentences = [s.strip() for s in content.split('.') if s.strip() and len(s.strip()) > 20]
            all_content.extend(sentences[:3])  # Первые 3 осмысленных предложения
        
        if not all_content:
            # Если не получилось разбить на предложения, берем первые 200 символов
            first_doc = documents[0]
            content = first_doc.page_content.strip()
            return content[:200] + ("..." if len(content) > 200 else "")
        
        # Формируем ответ из осмысленных предложений
        answer = '. '.join(all_content[:3]).strip()  # До 3 предложений
        if answer and not answer.endswith('.'):
            answer += '.'
        
        # Добавляем обобщение в зависимости от типа вопроса
        query_lower = query.lower()
        
        if "кто такой" in query_lower:
            if answer and "персонаж" not in answer.lower():
                answer = f"{answer} Это важный персонаж, который играет ключевую роль в истории."
        
        return answer
    
    def retrieve_context(self, query: str, min_score: float = 0.5) -> List[Document]:
        """
        Извлекает релевантные фрагменты из векторной базы.
        
        Args:
            query: Текст запроса
            min_score: Минимальный порог релевантности (cosine similarity)
                      Для ChromaDB: меньше score = лучше (расстояние)
                      Обычно хорошие результаты имеют score < 0.8
        
        Returns:
            Список релевантных документов
        """
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=self.config.top_k
        )
        
        # Фильтруем по релевантности
        # В ChromaDB score - это расстояние (меньше = лучше)
        # Обычно релевантные результаты имеют score < 0.8-0.9
        relevant_docs = []
        for doc, score in results:
            # Если score слишком большой (низкая релевантность), пропускаем
            # Для cosine distance: хорошие результаты обычно < 0.8
            if score < min_score:
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def format_context(self, documents: List[Document]) -> str:
        """Форматирует найденные документы в контекст для промпта."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('file_name', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"[Фрагмент {i} из {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def build_prompt_with_few_shot(self, query: str, context: str) -> str:
        """Строит промпт с Few-shot примерами, извлеченными из базы знаний."""
        prompt_parts = []
        
        prompt_parts.append("Ты помощник, который отвечает на вопросы на основе предоставленного контекста.")
        prompt_parts.append("КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО на русском языке. Не используй английские слова в ответе. Если в контексте есть английские термины, переведи их на русский или объясни на русском языке.")
        prompt_parts.append("\nВАЖНО: Отвечай ТОЛЬКО на основе предоставленного контекста. Если в контексте нет релевантной информации для ответа на вопрос, скажи: 'Извините, я не нашел релевантной информации по вашему вопросу в базе знаний.'")
        prompt_parts.append("\n## Примеры правильных ответов (из базы знаний):\n")
        
        # Извлекаем Few-shot примеры из базы знаний
        few_shot_examples = self._extract_few_shot_examples_from_db(num_examples=2)
        
        # Добавляем Few-shot примеры из базы
        for example in few_shot_examples:
            prompt_parts.append(f"Вопрос: {example['query']}")
            prompt_parts.append(f"Контекст: {example['context']}")
            prompt_parts.append(f"Ответ: {example['answer']}")
            prompt_parts.append("")
        
        prompt_parts.append("## Текущий вопрос:\n")
        prompt_parts.append(f"Контекст из базы знаний:\n{context}\n\n")
        prompt_parts.append(f"Вопрос: {query}")
        prompt_parts.append("\nОтвет (ТОЛЬКО на русском языке, без английских слов, как в примерах выше):")
        
        return "\n".join(prompt_parts)
    
    def build_prompt_with_cot(self, query: str, context: str) -> str:
        """Строит промпт с Chain-of-Thought техникой с примерами пошагового рассуждения."""
        prompt_parts = []
        
        prompt_parts.append("КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО на русском языке. Не используй английские слова в ответе. Если в контексте есть английские термины, переведи их на русский или объясни на русском языке.")
        prompt_parts.append("\nВАЖНО: Отвечай ТОЛЬКО на основе предоставленного контекста. Если в контексте нет релевантной информации для ответа на вопрос, скажи: 'Извините, я не нашел релевантной информации по вашему вопросу в базе знаний.'")
        prompt_parts.append("\nТы помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги рассуждения перед ответом.")
        prompt_parts.append("\n## Пример пошагового рассуждения:\n")
        prompt_parts.append("Вопрос: Какая технология используется в HyperRelay?")
        prompt_parts.append("Контекст: [Из документа] HyperRelay питается от ядра VoidCore...")
        prompt_parts.append("\nПроцесс рассуждения:")
        prompt_parts.append("1. Сначала найду, какая технология используется в HyperRelay.")
        prompt_parts.append("2. В документе указано, что HyperRelay питается от ядра VoidCore.")
        prompt_parts.append("3. Следовательно, ответ — VoidCore.")
        prompt_parts.append("\n## Текущий вопрос:\n")
        prompt_parts.append(f"Контекст из базы знаний:\n{context}\n\n")
        prompt_parts.append(f"Вопрос: {query}\n")
        prompt_parts.append("## Процесс рассуждения (опиши свои шаги):\n")
        prompt_parts.append("1. Сначала определю, какая информация нужна для ответа на вопрос.")
        prompt_parts.append("2. Затем найду релевантные фрагменты в предоставленном контексте.")
        prompt_parts.append("3. Проанализирую найденную информацию и сформулирую ответ.")
        prompt_parts.append("4. Следовательно, мой ответ:\n")
        
        return "\n".join(prompt_parts)
    
    def build_combined_prompt(self, query: str, context: str) -> str:
        """Строит промпт с комбинацией Few-shot и Chain-of-Thought."""
        prompt_parts = []
        
        prompt_parts.append("Ты помощник, который отвечает на вопросы на основе предоставленного контекста.")
        prompt_parts.append("КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО на русском языке. Не используй английские слова в ответе. Если в контексте есть английские термины, переведи их на русский или объясни на русском языке.")
        prompt_parts.append("\nВАЖНО: Отвечай ТОЛЬКО на основе предоставленного контекста. Если в контексте нет релевантной информации для ответа на вопрос, скажи: 'Извините, я не нашел релевантной информации по вашему вопросу в базе знаний.'")
        
        # Few-shot часть (извлекаем из базы)
        if self.config.use_few_shot:
            prompt_parts.append("\n## Примеры правильных ответов (из базы знаний):\n")
            few_shot_examples = self._extract_few_shot_examples_from_db(num_examples=1)
            for example in few_shot_examples:
                prompt_parts.append(f"Вопрос: {example['query']}")
                prompt_parts.append(f"Контекст: {example['context'][:200]}...")
                prompt_parts.append(f"Ответ: {example['answer']}")
                prompt_parts.append("")
        
        # Chain-of-Thought часть
        if self.config.use_chain_of_thought:
            prompt_parts.append("\nТы помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги рассуждения перед ответом.")
            prompt_parts.append("\n## Пример пошагового рассуждения:\n")
            prompt_parts.append("Вопрос: Какая технология используется в HyperRelay?")
            prompt_parts.append("Контекст: [Из документа] HyperRelay питается от ядра VoidCore...")
            prompt_parts.append("\nПроцесс рассуждения:")
            prompt_parts.append("1. Сначала найду, какая технология используется в HyperRelay.")
            prompt_parts.append("2. В документе указано, что HyperRelay питается от ядра VoidCore.")
            prompt_parts.append("3. Следовательно, ответ — VoidCore.")
            prompt_parts.append("")
        
        prompt_parts.append("## Текущий вопрос:\n")
        prompt_parts.append(f"Контекст из базы знаний:\n{context}\n\n")
        prompt_parts.append(f"Вопрос: {query}\n")
        
        if self.config.use_chain_of_thought:
            prompt_parts.append("## Процесс рассуждения (опиши свои шаги):\n")
            prompt_parts.append("1. Сначала определю, какая информация нужна для ответа на вопрос.")
            prompt_parts.append("2. Затем найду релевантные фрагменты в предоставленном контексте.")
            prompt_parts.append("3. Проанализирую найденную информацию и сформулирую ответ.")
            prompt_parts.append("4. Следовательно, мой ответ:\n")
        else:
            prompt_parts.append("Ответ (на основе контекста):")
        
        return "\n".join(prompt_parts)
    
    def build_prompt(self, query: str, context: str) -> str:
        """Строит финальный промпт с учетом настроек."""
        if self.config.use_few_shot and self.config.use_chain_of_thought:
            return self.build_combined_prompt(query, context)
        elif self.config.use_few_shot:
            return self.build_prompt_with_few_shot(query, context)
        elif self.config.use_chain_of_thought:
            return self.build_prompt_with_cot(query, context)
        else:
            # Базовый промпт без техник
            return f"""Ты помощник, который отвечает на вопросы на основе предоставленного контекста.
ВАЖНО: Всегда отвечай полностью на русском языке. Не используй английский язык в ответе.

КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО на основе предоставленного контекста. Если в контексте нет релевантной информации для ответа на вопрос, скажи: 'Извините, я не нашел релевантной информации по вашему вопросу в базе знаний.'

Контекст из базы знаний:
{context}

Вопрос: {query}

Ответ (ТОЛЬКО на русском языке, без английских слов, ТОЛЬКО на основе контекста выше):"""
    
    def query(self, question: str, verbose: bool = False, min_relevance_score: float = 0.8) -> Dict[str, Any]:
        """
        Обрабатывает вопрос пользователя и возвращает ответ.
        
        Args:
            question: Текст вопроса
            verbose: Если True, возвращает дополнительную информацию
            min_relevance_score: Максимальный допустимый score для релевантности
                                (меньше = лучше, для ChromaDB обычно < 0.8-0.9)
            
        Returns:
            Словарь с ответом и метаданными
        """
        # 1. Поиск релевантных фрагментов с проверкой релевантности
        documents = self.retrieve_context(question, min_score=min_relevance_score)
        
        if not documents:
            return {
                "answer": "Извините, я не нашел релевантной информации по вашему вопросу в базе знаний. Попробуйте задать вопрос о персонажах, планетах, технологиях или событиях из вселенной Cosmic Dominion.",
                "sources": [],
                "retrieved_chunks": 0
            }
        
        # 2. Дополнительная проверка: если документы найдены, но их очень мало или они не релевантны
        # Проверяем, что у нас есть достаточно релевантной информации
        if len(documents) < 1:
            return {
                "answer": "Извините, я не нашел релевантной информации по вашему вопросу в базе знаний. Попробуйте задать вопрос о персонажах, планетах, технологиях или событиях из вселенной Cosmic Dominion.",
                "sources": [],
                "retrieved_chunks": 0
            }
        
        # 2. Форматирование контекста
        context = self.format_context(documents)
        
        # 3. Построение промпта с техниками промптинга
        prompt = self.build_prompt(question, context)
        
        if verbose:
            print("\n" + "="*60)
            print("ПРОМПТ:")
            print("="*60)
            print(prompt)
            print("="*60 + "\n")
        
        # 4. Генерация ответа через LLM
        answer = self.llm.generate(prompt, self.config)
        
        # 5. Формирование результата
        result = {
            "answer": answer,
            "sources": [doc.metadata.get('file_name', 'Unknown') for doc in documents],
            "retrieved_chunks": len(documents)
        }
        
        if verbose:
            result["prompt"] = prompt
            result["context"] = context
        
        return result


def main():
    """Основная функция для интерактивного использования бота."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG-бот с техниками промптинга")
    parser.add_argument("--provider", choices=["openai", "yandexgpt", "local"], 
                       default="openai", help="Провайдер LLM")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Модель LLM")
    parser.add_argument("--no-few-shot", action="store_true", help="Отключить Few-shot")
    parser.add_argument("--no-cot", action="store_true", help="Отключить Chain-of-Thought")
    parser.add_argument("--top-k", type=int, default=3, help="Количество релевантных чанков")
    parser.add_argument("--verbose", action="store_true", help="Показать промпт")
    parser.add_argument("--query", type=str, help="Выполнить один запрос и выйти")
    
    args = parser.parse_args()
    
    # Создание конфигурации
    config = RAGConfig(
        llm_provider=args.provider,
        llm_model=args.model,
        use_few_shot=not args.no_few_shot,
        use_chain_of_thought=not args.no_cot,
        top_k=args.top_k
    )
    
    print("="*60)
    print("Инициализация RAG-бота...")
    print(f"Провайдер LLM: {config.llm_provider}")
    print(f"Модель: {config.llm_model}")
    print(f"Few-shot: {'Включен' if config.use_few_shot else 'Выключен'}")
    print(f"Chain-of-Thought: {'Включен' if config.use_chain_of_thought else 'Выключен'}")
    print("="*60 + "\n")
    
    try:
        bot = RAGBot(config)
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        if config.llm_provider == "openai":
            print("\nУбедитесь, что установлена переменная окружения OPENAI_API_KEY")
        return 1
    
    # Режим одного запроса
    if args.query:
        result = bot.query(args.query, verbose=args.verbose)
        print(f"\nВопрос: {args.query}")
        print(f"\nОтвет: {result['answer']}")
        print(f"\nИсточники: {', '.join(result['sources'])}")
        return 0
    
    # Интерактивный режим
    print("\nRAG-бот готов к работе!")
    print("Введите вопрос или 'exit' для выхода.\n")
    
    while True:
        try:
            question = input("Вопрос: ").strip()
            
            if question.lower() in ['exit', 'quit', 'выход']:
                break
            
            if not question:
                continue
            
            result = bot.query(question, verbose=args.verbose)
            
            print(f"\nОтвет: {result['answer']}")
            print(f"\nИсточники: {', '.join(result['sources'])}")
            print(f"Найдено фрагментов: {result['retrieved_chunks']}\n")
            
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
