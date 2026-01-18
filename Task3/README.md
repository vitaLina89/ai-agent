# Задание 3: Создание векторного индекса базы знаний

## Описание

Это задание преобразует подготовленную базу знаний из Task2 в векторный индекс для семантического поиска. Индекс используется для поиска релевантных фрагментов текста по пользовательским запросам.

## Структура проекта

```
Task3/
├── build_index.py       # Скрипт для создания векторного индекса
├── requirements.txt     # Зависимости Python
├── chroma_db/           # Папка с векторным индексом ChromaDB (создается автоматически)
└── README.md            # Документация
```

## Выбранная модель эмбеддингов

### Модель: `sentence-transformers/all-mpnet-base-v2`

- **Название модели:** all-mpnet-base-v2
- **Размер эмбеддингов:** 768
- **Репозиторий:** https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- **API:** Sentence-Transformers (Hugging Face)
- **Обоснование выбора:** 
  - Высокое качество семантического поиска (9/10 по сравнению с OpenAI embeddings)
  - Локальное выполнение (конфиденциальность данных)
  - Бесплатное использование (нет платы за токен)
  - Хорошая производительность на CPU и GPU
  - Рекомендована в Task1 как оптимальный выбор для гибридного решения

### Альтернативные модели

В зависимости от требований можно использовать:
- `all-MiniLM-L6-v2` (384 размерности) - быстрее, меньше память
- `paraphrase-multilingual-mpnet-base-v2` - поддержка многоязычности
- `text-embedding-ada-002` (OpenAI) - облачная модель, 1536 размерности

## Процесс создания индекса

### 1. Загрузка документов

Загружаются все текстовые файлы из `Task2/knowledge_base/` (34+ документа о вселенной Star Wars с замененными терминами).

### 2. Разбиение на чанки

Используется `RecursiveCharacterTextSplitter` из LangChain с параметрами:
- **Размер чанка:** 1000 символов (~250-300 слов или ~500-700 токенов)
- **Перекрытие:** 200 символов (для сохранения контекста между чанками)
- **Разделители:** `["\n\n", "\n", ". ", " ", ""]` (иерархическая разбивка)

Каждый чанк сохраняет метаданные:
- `source` - путь к исходному файлу
- `file_name` - имя файла без расширения
- `title` - заголовок документа (имя файла с пробелами)
- `chunk_id` - уникальный идентификатор чанка

### 3. Генерация эмбеддингов

Для каждого чанка генерируется 768-мерный вектор с помощью модели `all-mpnet-base-v2`.

### 4. Создание векторного индекса

Векторы и метаданные сохраняются в ChromaDB:
- **Векторная БД:** ChromaDB
- **Алгоритм поиска:** HNSW (Hierarchical Navigable Small World)
- **Расстояние:** косинусная близость (cosine similarity)
- **Персистентность:** автоматическое сохранение на диск

## Использование

### Установка зависимостей

```bash
cd Task3
pip install -r requirements.txt
```

### Создание индекса

```bash
python build_index.py
```

Скрипт выполнит:
1. Загрузку всех документов из `../Task2/knowledge_base/`
2. Разбиение на чанки
3. Генерацию эмбеддингов
4. Создание и сохранение индекса в `chroma_db/`
5. Тестовый поиск по 3 примерам запросов

### Использование индекса в коде

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Загрузка индекса
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Поиск по запросу
query = "Who is Kael Starwind?"
results = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Source: {doc.metadata['file_name']}")
    print(f"Text: {doc.page_content[:200]}...")
```

## Результаты

### Статистика индекса

**Результаты выполнения:**

- **Модель эмбеддингов:** `sentence-transformers/all-mpnet-base-v2` (768 размерностей)
- **База знаний:** 34 текстовых файла из `Task2/knowledge_base/` (Star Wars с замененными терминами)
- **Количество документов:** 34
- **Количество чанков:** 9,206
- **Время создания индекса:** 362.02 минут (~6 часов на CPU)
- **Размер индекса:** ~126 MB
- **Векторная БД:** ChromaDB (SQLite бэкенд)

### Пример запросов и результатов

Скрипт автоматически тестирует индекс на следующих запросах:

#### 1. Запрос: "Who is Kael Starwind?"

**Результат 1** (score: 0.7827):
- **Источник:** Luke_Skywalker
- **Chunk ID:** 6267
- **Текст:** "Though unsure if it was real or just a memory, Skywalker heard the familiar and comforting voice of Aris Thorne, urging him to let go, and so he did. Staring at a binary sunset on the horizon much li..."

**Результат 2** (score: 0.8221):
- **Источник:** Luke_Skywalker
- **Chunk ID:** 6317
- **Текст:** "Shortly after the revelation that Xarn Velgor was his father, Kael Starwind struggled with using Synth Flux and could not reach out to Kenobi. When consumed by anger and fear over what he had learned ..."

**Результат 3** (score: 0.8486):
- **Источник:** Luke_Skywalker
- **Chunk ID:** 6157
- **Текст:** "Kael Starwind visited many places in search of Flux Wardens lore, including Elphrona. Skywalker's research into the Flux Wardens was a long and difficult task that took him years. He was aided by L..."

#### 2. Запрос: "What is Synth Flux?"

**Результат 1** (score: 0.7251):
- **Источник:** The_Force
- **Chunk ID:** 8675
- **Текст:** "Pablo Hidalgo stated that the term \"Flux-sensitive\" is akin to someone being talented or gifted in the field. leitmotif of Synth Flux exists. In The Synthetic Wars, a deep rumble was typically u..."

**Результат 2** (score: 0.7323):
- **Источник:** The_Force
- **Chunk ID:** 8614
- **Текст:** "which was beyond the power of any man-made machine. Every lifeform in the universe had a place in Synth Flux, even simple bugs. Synth Flux existed in two forms: the Living Synth Flux and the Co..."

**Результат 3** (score: 0.7331):
- **Источник:** The_Force
- **Chunk ID:** 8613
- **Текст:** "A Synth Flux of others Non-canon Appearances Non-canon appearances Sources Non-canon sources Notes and references External links Description If it isn't magic, then what is it? Synth Flux is..."

#### 3. Запрос: "Tell me about the Void Core"

**Результат 1** (score: 0.6866):
- **Источник:** Battle_of_Endor
- **Chunk ID:** 1302
- **Текст:** "The Void Core Stratagem Star Wars Insider Star Wars: The Rise and Fall of the Cosmic Dominion Please update the article to include missing information, and remove this template when finished. Toda..."

**Результат 2** (score: 0.7821):
- **Источник:** Death_Star
- **Chunk ID:** 4428
- **Текст:** "# Death_Star For other uses, see Void Core We call it the Void Core. There is no better name, and the day is coming soon when it will be unleashed. ―Scientist Galen Erso Void Core was a gargantu..."

**Результат 3** (score: 0.7859):
- **Источник:** Battle_of_Yavin
- **Chunk ID:** 1530
- **Текст:** "Void Core and was seen as one of the first major victories over the Cosmic Dominion Contents Prelude The battle The approach Trench run Destruction of the Void Core Aftermath Liberation Fron..."

**Примечание:** Score показывает расстояние (меньше = лучше релевантность). Все результаты релевантны запросам и содержат нужную информацию.

## База знаний

### Источник

База знаний создана в Task2 на основе статей из Star Wars Fandom Wiki, где все ключевые термины заменены на вымышленные названия.

### Формат

- **Тип:** Текстовые файлы (.txt)
- **Количество:** 34+ документов
- **Категории:**
  - Персонажи (14 документов): Kael Starwind, Xarn Velgor, Princess Zara и др.
  - Планеты (8 документов): Vexara Prime, Metropolis Prime, Frosthold и др.
  - Технологии (6 документов): Void Core, Plasma Blade, Stardust Runner и др.
  - События (6 документов): Battle of Helios, Protocol Omega и др.

### Замененные термины

Примеры замен:
- `Luke Skywalker` → `Kael Starwind`
- `The Force` → `Synth Flux`
- `Death Star` → `Void Core`
- `Darth Vader` → `Xarn Velgor`

Полный список замен находится в `Task2/terms_map.json`.

## Ограничения и рекомендации

### Производительность

- **CPU:** Индексация может занять 2-5 минут
- **GPU:** При наличии GPU можно ускорить в 5-10 раз (измените `device='cuda'` в `build_index.py`)
- **Память:** Требуется ~2-4 GB RAM для модели и индекса

### Размер индекса

- Размер индекса на диске: ~50-200 MB (зависит от количества чанков)
- Время поиска: < 100 мс для 1000+ чанков

### Масштабирование

ChromaDB с SQLite бэкендом подходит для:
- До 100K-1M документов
- Небольшие и средние команды (< 200 пользователей)

Для больших объемов рекомендуется:
- Использовать ClickHouse бэкенд ChromaDB
- Или перейти на Qdrant/FAISS с дополнительной инфраструктурой

## Проверка качества

Скрипт автоматически выполняет тестовый поиск для проверки качества индекса. Рекомендуется:

1. Убедиться, что запросы возвращают релевантные результаты
2. Проверить, что найденные чанки соответствуют теме запроса
3. Оценить score (меньше = лучше релевантность)

### Дополнительные тесты

Вы можете добавить свои тестовые запросы в скрипт или использовать интерактивный поиск:

```python
# Интерактивный поиск
while True:
    query = input("Введите запрос (или 'exit' для выхода): ")
    if query.lower() == 'exit':
        break
    
    results = vectorstore.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Source: {doc.metadata['file_name']}")
        print(f"   Text: {doc.page_content[:300]}...")
```

## Зависимости

Основные библиотеки:
- `langchain` - фреймворк для работы с документами и чанкингом
- `langchain-community` - интеграция с ChromaDB и HuggingFace
- `sentence-transformers` - модель эмбеддингов
- `chromadb` - векторная база данных
- `torch` - PyTorch для работы модели

Полный список: `requirements.txt`

## Следующие шаги

После создания индекса можно:
1. Интегрировать индекс в RAG-бот (Task 4+)
2. Настроить фильтрацию по метаданным
3. Оптимизировать параметры чанкинга для вашей задачи
4. Добавить поддержку обновления индекса при изменении документов

## Примечания

- Индекс автоматически сохраняется в `chroma_db/` при первом запуске
- При повторном запуске скрипт создаст новый индекс (перезапишет старый)
- Модель `all-mpnet-base-v2` загружается автоматически при первом использовании (~400 MB)
- Для ускорения можно использовать GPU (измените `device='cuda'` в `create_embeddings_model()`)
