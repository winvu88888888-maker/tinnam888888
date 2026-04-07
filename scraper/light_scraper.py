"""
Light Scraper — Uses requests + BeautifulSoup (NO Selenium needed).
Works on Streamlit Cloud where Chrome is NOT available.
Scrapes from ketquadientoan.com using simple HTTP requests.
"""
import requests
import re
import os
import sys
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scraper.data_manager import insert_mega645, insert_power655, get_count, get_latest_date


# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

MEGA_URL = "https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html"
POWER_URL = "https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html"


def parse_date(date_str):
    """Parse Vietnamese date string to YYYY-MM-DD."""
    match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return date_str


def scrape_mega645_light():
    """Scrape Mega 6/45 using requests (no Selenium)."""
    print("[LightScraper] Fetching Mega 6/45...")
    try:
        # Try POST with date range to get all data
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # First get the page to establish session
        resp = session.get(MEGA_URL, timeout=30)
        resp.raise_for_status()
        
        # Try with date parameters
        data = {
            'datef': '01-07-2016',
            'datet': datetime.now().strftime('%d-%m-%Y'),
        }
        resp = session.post(MEGA_URL, data=data, timeout=30)
        if resp.status_code != 200:
            resp = session.get(MEGA_URL, timeout=30)
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find all table rows with lottery data
        rows = soup.select('table tbody tr')
        if not rows:
            rows = soup.select('table tr')
        
        db_rows = []
        for row in rows:
            tds = row.find_all('td')
            if len(tds) < 2:
                continue
            
            date_text = tds[0].get_text(strip=True)
            if not re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_text):
                continue
            
            # Find ball numbers
            balls = tds[1].find_all('span', class_=re.compile(r'whiteball|ball'))
            numbers = []
            for ball in balls:
                txt = ball.get_text(strip=True)
                try:
                    num = int(txt)
                    if 1 <= num <= 45:
                        numbers.append(num)
                except ValueError:
                    continue
            
            if len(numbers) < 6:
                # Try alternative: find all numbers in the cell
                all_nums = re.findall(r'\b(\d{1,2})\b', tds[1].get_text())
                numbers = [int(n) for n in all_nums if 1 <= int(n) <= 45]
            
            if len(numbers) >= 6:
                date_str = parse_date(date_text)
                jackpot = tds[2].get_text(strip=True) if len(tds) >= 3 else ''
                db_rows.append((date_str, numbers[0], numbers[1], numbers[2],
                              numbers[3], numbers[4], numbers[5], jackpot))
        
        print(f"[LightScraper] Found {len(db_rows)} Mega 6/45 draws")
        
        if db_rows:
            inserted = insert_mega645(db_rows)
            print(f"[LightScraper] Inserted {inserted} new Mega draws")
            return inserted
        
        return 0
        
    except Exception as e:
        print(f"[LightScraper] Error scraping Mega 6/45: {e}")
        return 0


def scrape_power655_light():
    """Scrape Power 6/55 using requests (no Selenium)."""
    print("[LightScraper] Fetching Power 6/55...")
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        
        resp = session.get(POWER_URL, timeout=30)
        resp.raise_for_status()
        
        data = {
            'datef': '01-01-2017',
            'datet': datetime.now().strftime('%d-%m-%Y'),
        }
        resp = session.post(POWER_URL, data=data, timeout=30)
        if resp.status_code != 200:
            resp = session.get(POWER_URL, timeout=30)
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        rows = soup.select('table tbody tr')
        if not rows:
            rows = soup.select('table tr')
        
        db_rows = []
        for row in rows:
            tds = row.find_all('td')
            if len(tds) < 2:
                continue
            
            date_text = tds[0].get_text(strip=True)
            if not re.search(r'\d{1,2}/\d{1,2}/\d{4}', date_text):
                continue
            
            # Regular balls (not bonus)
            regular_balls = tds[1].find_all('span', class_=re.compile(r'whiteball(?!.*jphu)'))
            bonus_ball = tds[1].find('span', class_=re.compile(r'jphu'))
            
            numbers = []
            for ball in regular_balls:
                txt = ball.get_text(strip=True)
                try:
                    num = int(txt)
                    if 1 <= num <= 55:
                        numbers.append(num)
                except ValueError:
                    continue
            
            bonus = 0
            if bonus_ball:
                try:
                    bonus = int(bonus_ball.get_text(strip=True))
                except ValueError:
                    pass
            
            if len(numbers) < 6:
                all_nums = re.findall(r'\b(\d{1,2})\b', tds[1].get_text())
                parsed = [int(n) for n in all_nums if 1 <= int(n) <= 55]
                if len(parsed) >= 7:
                    numbers = parsed[:6]
                    bonus = parsed[6] if bonus == 0 else bonus
                elif len(parsed) >= 6:
                    numbers = parsed[:6]
            
            if len(numbers) >= 6:
                date_str = parse_date(date_text)
                jackpot = tds[2].get_text(strip=True) if len(tds) >= 3 else ''
                db_rows.append((date_str, numbers[0], numbers[1], numbers[2],
                              numbers[3], numbers[4], numbers[5], bonus, jackpot))
        
        print(f"[LightScraper] Found {len(db_rows)} Power 6/55 draws")
        
        if db_rows:
            inserted = insert_power655(db_rows)
            print(f"[LightScraper] Inserted {inserted} new Power draws")
            return inserted
        
        return 0
        
    except Exception as e:
        print(f"[LightScraper] Error scraping Power 6/55: {e}")
        return 0


def scrape_all_light():
    """Scrape both lottery types using light scraper."""
    print("=" * 60)
    print("  LIGHT SCRAPER (requests + BeautifulSoup)")
    print("  No Selenium/Chrome needed — works on Streamlit Cloud")
    print("=" * 60)
    
    mega_new = scrape_mega645_light()
    power_new = scrape_power655_light()
    
    mega_total = get_count('mega')
    power_total = get_count('power')
    
    print(f"\n  Results: Mega={mega_total} (+{mega_new}), Power={power_total} (+{power_new})")
    return mega_new, power_new


if __name__ == '__main__':
    scrape_all_light()
