"""
Скрипт для создания векторного индекса базы знаний.

Использует:
- Sentence-Transformers модель all-mpnet-base-v2 для генерации эмбеддингов
- LangChain RecursiveCharacterTextSplitter для разбиения на чанки
- ChromaDB для хранения векторного индекса
"""

import time
from pathlib import Path
from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
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


def load_knowledge_base(knowledge_base_path: str) -> List[Document]:
    """
    Загружает все текстовые документы из базы знаний.
    
    Args:
        knowledge_base_path: Путь к папке с документами
        
    Returns:
        Список Document объектов с текстом и метаданными
    """
    documents = []
    kb_path = Path(knowledge_base_path)
    
    if not kb_path.exists():
        raise FileNotFoundError(f"База знаний не найдена: {knowledge_base_path}")
    
    print(f"Загрузка документов из {knowledge_base_path}...")
    
    for txt_file in sorted(kb_path.glob("*.txt")):
        file_name = txt_file.stem
        print(f"  Загрузка: {file_name}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Создаем Document с метаданными
            doc = Document(
                page_content=content,
                metadata={
                    "source": str(txt_file),
                    "file_name": file_name,
                    "title": file_name.replace("_", " ")
                }
            )
            documents.append(doc)
        except Exception as e:
            print(f"  Ошибка при загрузке {file_name}: {e}")
            continue
    
    print(f"Загружено документов: {len(documents)}")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Разбивает документы на чанки для более эффективного поиска.
    
    Args:
        documents: Список документов
        chunk_size: Размер чанка (примерно 250-300 слов)
        chunk_overlap: Перекрытие между чанками для контекста
        
    Returns:
        Список чанков документов
    """
    print("\nРазбиение документов на чанки...")
    
    # Используем RecursiveCharacterTextSplitter
    # Настройки: ~200 слов или ~500-1000 токенов на чанк
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # примерно 250-300 слов
        chunk_overlap=chunk_overlap,  # перекрытие для контекста
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Добавляем уникальный ID для каждого чанка
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Создано чанков: {len(chunks)}")
    
    # Показываем примеры размеров чанков
    chunk_sizes = [len(chunk.page_content) for chunk in chunks[:10]]
    print(f"Примеры размеров первых 10 чанков: {chunk_sizes}")
    
    return chunks


def create_embeddings_model():
    """
    Создает модель эмбеддингов на основе Sentence-Transformers.
    
    Returns:
        HuggingFaceEmbeddings модель
    """
    print("\nИнициализация модели эмбеддингов...")
    print("Модель: sentence-transformers/all-mpnet-base-v2")
    print("Размер эмбеддингов: 768")
    print("Репозиторий: https://huggingface.co/sentence-transformers/all-mpnet-base-v2")
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # можно изменить на 'cuda' если есть GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Модель загружена успешно!")
    return embeddings


def build_index(
    knowledge_base_path: str,
    persist_directory: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Создает векторный индекс базы знаний.
    
    Args:
        knowledge_base_path: Путь к папке с документами
        persist_directory: Директория для сохранения индекса ChromaDB
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие между чанками
    """
    start_time = time.time()
    
    print("=" * 60)
    print("Создание векторного индекса базы знаний")
    print("=" * 60)
    
    # 1. Загрузка документов
    documents = load_knowledge_base(knowledge_base_path)
    
    if not documents:
        raise ValueError("Не удалось загрузить документы из базы знаний")
    
    # 2. Разбиение на чанки
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # 3. Создание модели эмбеддингов
    embeddings = create_embeddings_model()
    
    # 4. Создание и сохранение индекса в ChromaDB
    print("\nГенерация эмбеддингов и создание индекса...")
    print("Это может занять некоторое время...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Явное сохранение (для совместимости со старыми версиями)
    # В новых версиях ChromaDB автоматически сохраняет при указании persist_directory
    try:
        if hasattr(vectorstore, 'persist'):
            vectorstore.persist()
    except Exception:
        pass  # persist() может быть не доступен в новых версиях
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Индекс успешно создан!")
    print("=" * 60)
    print(f"Количество документов: {len(documents)}")
    print(f"Количество чанков: {len(chunks)}")
    print(f"Время создания индекса: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"Индекс сохранен в: {persist_directory}")
    
    return vectorstore


def test_search(vectorstore: Chroma, queries: List[str], top_k: int = 3):
    """
    Тестирует поиск по индексу на примерах запросов.
    
    Args:
        vectorstore: ChromaDB индекс
        queries: Список тестовых запросов
        top_k: Количество результатов для каждого запроса
    """
    print("\n" + "=" * 60)
    print("Тестирование поиска по индексу")
    print("=" * 60)
    
    for query in queries:
        print(f"\nЗапрос: '{query}'")
        print("-" * 60)
        
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\nРезультат {i} (score: {score:.4f}):")
            print(f"  Источник: {doc.metadata.get('file_name', 'unknown')}")
            print(f"  Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}")
            print(f"  Текст (первые 200 символов):")
            print(f"  {doc.page_content[:200]}...")


if __name__ == "__main__":
    # Параметры
    KNOWLEDGE_BASE_PATH = "../Task2/knowledge_base"
    PERSIST_DIRECTORY = "./chroma_db"
    
    # Создание индекса
    vectorstore = build_index(
        knowledge_base_path=KNOWLEDGE_BASE_PATH,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Тестовые запросы (используем вымышленные термины из Task2)
    test_queries = [
        "Who is Kael Starwind?",
        "What is Synth Flux?",
        "Tell me about the Void Core"
    ]
    
    # Тестирование поиска
    test_search(vectorstore, test_queries, top_k=3)
    
    print("\n" + "=" * 60)
    print("Готово! Векторный индекс создан и протестирован.")
    print("=" * 60)
