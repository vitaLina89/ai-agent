# Задание 2: Подготовка базы знаний

## Описание

Это задание создает уникальную базу знаний на основе вселенной Star Wars с заменой всех ключевых терминов на вымышленные названия. Это гарантирует, что LLM не сможет использовать информацию из своей обучающей выборки и будет полагаться только на загруженную базу данных.

## Структура проекта

```
Task2/
├── download_pages.py      # Скрипт для скачивания HTML страниц
├── process_and_replace.py # Скрипт для очистки HTML и замены терминов
├── terms_map.json         # Словарь замен (исходное → вымышленное)
├── requirements.txt       # Зависимости Python
├── raw_pages/            # Временная папка с сырыми HTML (создается автоматически)
└── knowledge_base/       # Финальная база знаний (30+ документов)
```

## Использование

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Скачивание страниц

```bash
python download_pages.py
```

Это скачает 30+ HTML страниц из Star Wars Fandom Wiki в папку `raw_pages/`.

### 3. Обработка и замена терминов

```bash
python process_and_replace.py
```

Это извлечет текст из HTML, заменит все термины согласно `terms_map.json` и сохранит чистые текстовые документы в `knowledge_base/`.

## Словарь замен (terms_map.json)

### Исходная вселенная
Была выбрана вселенная **Star Wars** (starwars.fandom.com) как хорошо известная LLM вселенная с богатой мифологией.

### Принцип замены

Все ключевые термины заменены на вымышленные, сохраняя:
- Логику и структуру мира
- Стиль повествования
- Связи между сущностями

### Примеры замен

**Персонажи:**
- `Darth Vader` → `Xarn Velgor`
- `Luke Skywalker` → `Kael Starwind`
- `Yoda` → `Master Qylos`

**Планеты:**
- `Tatooine` → `Vexara Prime`
- `Coruscant` → `Metropolis Prime`
- `Hoth` → `Frosthold`

**Технологии:**
- `Death Star` → `Void Core`
- `Lightsaber` → `Plasma Blade`
- `Millennium Falcon` → `Stardust Runner`

**Концепции:**
- `The Force` → `Synth Flux`
- `Jedi` → `Flux Wardens`
- `Sith` → `Void Lords`
- `Dark Side` → `Void Aspect`

Полный список замен находится в `terms_map.json`.

## Результат

После выполнения скриптов в папке `knowledge_base/` будет находиться 30+ текстовых документов, каждый из которых:

1. Содержит информацию об одной сущности (персонаж, планета, технология, событие)
2. Очищен от HTML разметки
3. Имеет все оригинальные термины заменены на вымышленные
4. Сохраняет логику и смысл исходного текста

## Почему это работает для RAG

1. **Невозможно угадать**: LLM не знает термины вроде "Xarn Velgor" или "Synth Flux" из обучающей выборки
2. **Реалистичный сценарий**: Имитирует корпоративную базу знаний с уникальной терминологией
3. **Проверка работы RAG**: Модель может ответить только используя загруженные документы

## Список документов

База знаний включает:

- **Персонажи** (14): Xarn Velgor (Darth Vader), Kael Starwind (Luke Skywalker), Princess Zara (Leia), Torrin Black (Han Solo), и другие
- **Планеты** (8): Vexara Prime (Tatooine), Metropolis Prime (Coruscant), Frosthold (Hoth), и другие
- **Технологии** (6): Void Core (Death Star), Plasma Blade (Lightsaber), Stardust Runner (Millennium Falcon), и другие
- **События** (6): Battle of Helios (Yavin), Protocol Omega (Order 66), и другие

**Всего: 34+ документа**
