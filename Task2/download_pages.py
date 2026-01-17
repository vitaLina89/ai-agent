#!/usr/bin/env python3
"""
Скрипт для скачивания страниц из Star Wars Fandom Wiki.
Скачивает HTML страницы для дальнейшей обработки.
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import json
from urllib.parse import quote

# Список страниц для скачивания (30+ сущностей)
PAGES = [
    # Персонажи
    "Darth_Vader",
    "Luke_Skywalker",
    "Princess_Leia",
    "Han_Solo",
    "Obi-Wan_Kenobi",
    "Yoda",
    "Darth_Sidious",
    "Anakin_Skywalker",
    "Chewbacca",
    "R2-D2",
    "C-3PO",
    "Lando_Calrissian",
    "Boba_Fett",
    "Emperor_Palpatine",
    
    # Планеты
    "Tatooine",
    "Alderaan",
    "Coruscant",
    "Naboo",
    "Endor",
    "Hoth",
    "Dagobah",
    "Mustafar",
    
    # Технологии и объекты
    "Death_Star",
    "Lightsaber",
    "Millennium_Falcon",
    "X-wing_starfighter",
    "TIE_fighter",
    "The_Force",
    
    # События
    "Battle_of_Yavin",
    "Battle_of_Hoth",
    "Battle_of_Endor",
    "Order_66",
    "Rise_of_the_Empire",
    "Galactic_Civil_War",
]

BASE_URL = "https://starwars.fandom.com/wiki/"

def download_page(page_name, output_dir="raw_pages"):
    """Скачивает HTML страницу и сохраняет её."""
    os.makedirs(output_dir, exist_ok=True)
    
    url = BASE_URL + page_name
    print(f"Скачиваю: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Сохраняем HTML
        file_path = os.path.join(output_dir, f"{page_name}.html")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"  ✓ Сохранено: {file_path}")
        return True
    
    except Exception as e:
        print(f"  ✗ Ошибка при скачивании {page_name}: {e}")
        return False

def main():
    """Скачивает все страницы из списка."""
    print(f"Начинаю скачивание {len(PAGES)} страниц...")
    
    success_count = 0
    for page in PAGES:
        if download_page(page):
            success_count += 1
        time.sleep(1)  # Вежливая задержка между запросами
    
    print(f"\nСкачано успешно: {success_count}/{len(PAGES)}")
    
    # Сохраняем список страниц
    with open("pages_list.json", "w", encoding="utf-8") as f:
        json.dump(PAGES, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
