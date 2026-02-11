#!/usr/bin/env python3
"""
Скрипт для очистки HTML и замены ключевых терминов на вымышленные.
Обрабатывает скачанные страницы и создает чистые текстовые документы.
"""

import os
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path

def load_terms_map(map_file="terms_map.json"):
    """Загружает словарь замен терминов."""
    with open(map_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_text_from_html(html_file):
    """Извлекает основной текст из HTML страницы."""
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Удаляем скрипты и стили
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.decompose()
    
    # Ищем основной контент статьи
    content_div = soup.find('div', {'id': 'mw-content-text'}) or soup.find('article')
    
    if not content_div:
        # Если не нашли основной контент, берем body
        content_div = soup.find('body') or soup
    
    # Извлекаем текст
    text = content_div.get_text(separator='\n', strip=True)
    
    # Очищаем от лишних пробелов и пустых строк
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) > 3:  # Пропускаем слишком короткие строки
            lines.append(line)
    
    return '\n\n'.join(lines)

def replace_terms(text, terms_map):
    """
    Заменяет все термины в тексте согласно словарю.
    Сначала заменяет длинные фразы, затем короткие, чтобы избежать перекрытий.
    """
    # Сортируем термины по длине (от длинных к коротким)
    sorted_terms = sorted(terms_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in sorted_terms:
        # Используем word boundaries для точной замены
        # Учитываем регистр и различные варианты написания
        patterns = [
            re.escape(original),  # Точное совпадение
            re.escape(original.capitalize()),  # С заглавной
            re.escape(original.title()),  # Title Case
        ]
        
        for pattern in patterns:
            # Заменяем только целые слова/фразы
            text = re.sub(
                r'\b' + pattern + r'\b',
                replacement,
                text,
                flags=re.IGNORECASE
            )
    
    return text

def process_page(html_file, output_dir, terms_map):
    """Обрабатывает одну HTML страницу: извлекает текст и заменяет термины."""
    page_name = Path(html_file).stem
    
    print(f"Обрабатываю: {page_name}")
    
    # Извлекаем текст
    text = extract_text_from_html(html_file)
    
    if not text or len(text) < 100:
        print(f"  ⚠ Предупреждение: слишком короткий текст для {page_name}")
    
    # Заменяем термины
    processed_text = replace_terms(text, terms_map)
    
    # Сохраняем в текстовый файл
    output_file = os.path.join(output_dir, f"{page_name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# {page_name}\n\n")
        f.write(processed_text)
    
    print(f"  ✓ Сохранено: {output_file}")
    return True

def main():
    """Обрабатывает все скачанные HTML страницы."""
    raw_dir = "raw_pages"
    output_dir = "knowledge_base"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем словарь замен
    terms_map = load_terms_map()
    print(f"Загружено {len(terms_map)} замен терминов")
    
    # Обрабатываем все HTML файлы
    html_files = list(Path(raw_dir).glob("*.html"))
    print(f"\nНайдено {len(html_files)} HTML файлов для обработки")
    
    success_count = 0
    for html_file in html_files:
        if process_page(html_file, output_dir, terms_map):
            success_count += 1
    
    print(f"\nОбработано успешно: {success_count}/{len(html_files)}")
    print(f"Документы сохранены в: {output_dir}/")

if __name__ == "__main__":
    main()
