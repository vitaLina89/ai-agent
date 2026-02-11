# Примеры запросов к векторному индексу

Этот файл содержит примеры запросов к векторному индексу и найденные результаты.

## Информация об индексе

- **Модель эмбеддингов:** `sentence-transformers/all-mpnet-base-v2` (768 размерностей)
- **База знаний:** 34 текстовых файла из `Task2/knowledge_base/`
- **Количество чанков:** 9,206
- **Время создания индекса:** 362.02 минут (~6 часов)

## Пример 1: Запрос о персонаже

**Запрос:** `"Who is Kael Starwind?"`

**Найденные чанки:**

### Результат 1 (score: 0.7827)
- **Источник:** Luke_Skywalker.txt
- **Chunk ID:** 6267
- **Текст:** "Though unsure if it was real or just a memory, Skywalker heard the familiar and comforting voice of Aris Thorne, urging him to let go, and so he did. Staring at a binary sunset on the horizon much li..."

### Результат 2 (score: 0.8221)
- **Источник:** Luke_Skywalker.txt
- **Chunk ID:** 6317
- **Текст:** "Shortly after the revelation that Xarn Velgor was his father, Kael Starwind struggled with using Synth Flux and could not reach out to Kenobi. When consumed by anger and fear over what he had learned ..."

### Результат 3 (score: 0.8486)
- **Источник:** Luke_Skywalker.txt
- **Chunk ID:** 6157
- **Текст:** "Kael Starwind visited many places in search of Flux Wardens lore, including Elphrona. Skywalker's research into the Flux Wardens was a long and difficult task that took him years. He was aided by L..."

**Анализ:** Все три результата из правильного файла `Luke_Skywalker.txt`, который содержит информацию о Kael Starwind (вымышленное имя Luke Skywalker). Чанки релевантны запросу.

---

## Пример 2: Запрос о концепции

**Запрос:** `"What is Synth Flux?"`

**Найденные чанки:**

### Результат 1 (score: 0.7251)
- **Источник:** The_Force.txt
- **Chunk ID:** 8675
- **Текст:** "Pablo Hidalgo stated that the term \"Flux-sensitive\" is akin to someone being talented or gifted in the field. leitmotif of Synth Flux exists. In The Synthetic Wars, a deep rumble was typically u..."

### Результат 2 (score: 0.7323)
- **Источник:** The_Force.txt
- **Chunk ID:** 8614
- **Текст:** "which was beyond the power of any man-made machine. Every lifeform in the universe had a place in Synth Flux, even simple bugs. Synth Flux existed in two forms: the Living Synth Flux and the Co..."

### Результат 3 (score: 0.7331)
- **Источник:** The_Force.txt
- **Chunk ID:** 8613
- **Текст:** "A Synth Flux of others Non-canon Appearances Non-canon appearances Sources Non-canon sources Notes and references External links Description If it isn't magic, then what is it? Synth Flux is..."

**Анализ:** Все результаты из файла `The_Force.txt`, что правильно, так как Synth Flux - это вымышленное название The Force. Чанки содержат информацию о концепции.

---

## Пример 3: Запрос о технологии

**Запрос:** `"Tell me about the Void Core"`

**Найденные чанки:**

### Результат 1 (score: 0.6866)
- **Источник:** Battle_of_Endor.txt
- **Chunk ID:** 1302
- **Текст:** "The Void Core Stratagem Star Wars Insider Star Wars: The Rise and Fall of the Cosmic Dominion Please update the article to include missing information, and remove this template when finished. Toda..."

### Результат 2 (score: 0.7821)
- **Источник:** Death_Star.txt
- **Chunk ID:** 4428
- **Текст:** "# Death_Star For other uses, see Void Core We call it the Void Core. There is no better name, and the day is coming soon when it will be unleashed. ―Scientist Galen Erso Void Core was a gargantu..."

### Результат 3 (score: 0.7859)
- **Источник:** Battle_of_Yavin.txt
- **Chunk ID:** 1530
- **Текст:** "Void Core and was seen as one of the first major victories over the Cosmic Dominion Contents Prelude The battle The approach Trench run Destruction of the Void Core Aftermath Liberation Fron..."

**Анализ:** Результаты из файлов `Death_Star.txt` и связанных битв, что правильно, так как Void Core - это вымышленное название Death Star. Чанки содержат релевантную информацию о технологии и сражениях.

---

## Выводы

Все три запроса показали высокую релевантность результатов:
- ✅ Результаты из правильных файлов базы знаний
- ✅ Содержат искомую информацию
- ✅ Score в разумных пределах (0.68-0.85)
- ✅ Различные чанки из одного документа показывают разные аспекты темы

Векторный индекс работает корректно и готов к использованию в RAG-боте.
