"""
Auto-Updater V2 — Incremental data updater for Streamlit app.
Priority: Light scraper (requests) → Selenium fallback.
Works on Streamlit Cloud without Chrome.
"""
import time
import os
import json
import threading
from datetime import datetime, timedelta

_COOLDOWN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'data', '.last_update'
)
_COOLDOWN_HOURS = 1  # Reduced from 6 → 1 hour for fresher data
_update_lock = threading.Lock()


def _get_last_update_time():
    """Read last successful update timestamp from disk."""
    try:
        if os.path.exists(_COOLDOWN_FILE):
            with open(_COOLDOWN_FILE, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
    except Exception:
        pass
    return datetime(2000, 1, 1)


def _save_update_time():
    """Save current timestamp as last update time."""
    try:
        os.makedirs(os.path.dirname(_COOLDOWN_FILE), exist_ok=True)
        with open(_COOLDOWN_FILE, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'method': 'auto_update_v2'
            }, f)
    except Exception:
        pass


def _needs_update():
    """Check if we should attempt an update based on cooldown."""
    last = _get_last_update_time()
    elapsed = datetime.now() - last
    return elapsed > timedelta(hours=_COOLDOWN_HOURS)


def _is_draw_day():
    """Check if today is a draw day for any lottery.
    Mega: Wed(2), Fri(4), Sun(6)
    Power: Tue(1), Thu(3), Sat(5)
    """
    wd = datetime.now().weekday()
    return wd in {1, 2, 3, 4, 5, 6}  # Every day except Monday


def _data_is_current():
    """Check if data is already up to date."""
    from scraper.data_manager import get_latest_date
    today = datetime.now().date()
    
    mega_latest = get_latest_date('mega')
    power_latest = get_latest_date('power')
    
    mega_ok = False
    power_ok = False
    
    if mega_latest:
        try:
            ld = datetime.strptime(mega_latest, '%Y-%m-%d').date()
            if (today - ld).days <= 2:  # Within 2 days = likely current
                mega_ok = True
        except Exception:
            pass
    
    if power_latest:
        try:
            ld = datetime.strptime(power_latest, '%Y-%m-%d').date()
            if (today - ld).days <= 2:
                power_ok = True
        except Exception:
            pass
    
    return mega_ok and power_ok


def auto_update_data():
    """
    Auto-update lottery data. Uses light scraper (requests-based) first,
    falls back to Selenium if available.
    
    Returns dict with status info.
    """
    from scraper.data_manager import get_count, get_latest_date

    # Check cooldown
    if not _needs_update():
        mega_c = get_count('mega')
        power_c = get_count('power')
        mega_latest = get_latest_date('mega') or 'N/A'
        power_latest = get_latest_date('power') or 'N/A'
        return {
            'status': 'skipped',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': f'Cooldown active ({_COOLDOWN_HOURS}h). '
                       f'Mega: {mega_c} kỳ (→{mega_latest}), '
                       f'Power: {power_c} kỳ (→{power_latest}).',
        }

    # Check if data is already current
    if _data_is_current():
        _save_update_time()
        mega_c = get_count('mega')
        power_c = get_count('power')
        mega_latest = get_latest_date('mega') or 'N/A'
        power_latest = get_latest_date('power') or 'N/A'
        return {
            'status': 'skipped',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': f'Data đã mới nhất. '
                       f'Mega: {mega_c} kỳ (→{mega_latest}), '
                       f'Power: {power_c} kỳ (→{power_latest}).',
        }

    # Acquire lock to prevent concurrent scraping
    if not _update_lock.acquire(blocking=False):
        mega_c = get_count('mega')
        power_c = get_count('power')
        return {
            'status': 'skipped',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': 'Đang cập nhật ở luồng khác...',
        }

    try:
        mega_before = get_count('mega')
        power_before = get_count('power')
        method_used = 'none'

        # === STRATEGY 1: Light scraper (requests + BeautifulSoup) ===
        try:
            from scraper.light_scraper import scrape_mega645_light, scrape_power655_light
            
            mega_new_light = scrape_mega645_light()
            power_new_light = scrape_power655_light()
            
            if mega_new_light > 0 or power_new_light > 0:
                method_used = 'light_scraper'
        except ImportError:
            print("[AutoUpdate] light_scraper not available")
        except Exception as e:
            print(f"[AutoUpdate] Light scraper error: {e}")

        # === STRATEGY 2: Selenium fallback (only if light scraper found nothing) ===
        mega_after_light = get_count('mega')
        power_after_light = get_count('power')
        
        if method_used == 'none':
            try:
                from scraper.scraper import create_driver, scrape_mega645, scrape_power655
                driver = create_driver()
                try:
                    scrape_mega645(driver)
                    scrape_power655(driver)
                    method_used = 'selenium'
                finally:
                    driver.quit()
            except ImportError:
                print("[AutoUpdate] Selenium not available (Streamlit Cloud)")
            except Exception as e:
                print(f"[AutoUpdate] Selenium error: {e}")

        # Calculate results
        mega_after = get_count('mega')
        power_after = get_count('power')
        mega_new = mega_after - mega_before
        power_new = power_after - power_before
        mega_latest = get_latest_date('mega') or 'N/A'
        power_latest = get_latest_date('power') or 'N/A'

        _save_update_time()

        if mega_new > 0 or power_new > 0:
            return {
                'status': 'updated',
                'mega_count': mega_after,
                'power_count': power_after,
                'mega_new': mega_new,
                'power_new': power_new,
                'message': f'✅ Cập nhật thành công ({method_used})! '
                           f'Mega: +{mega_new} kỳ (tổng {mega_after}, →{mega_latest}), '
                           f'Power: +{power_new} kỳ (tổng {power_after}, →{power_latest}).',
            }
        else:
            return {
                'status': 'checked',
                'mega_count': mega_after,
                'power_count': power_after,
                'mega_new': 0,
                'power_new': 0,
                'message': f'Kiểm tra xong — không có kỳ mới. '
                           f'Mega: {mega_after} kỳ (→{mega_latest}), '
                           f'Power: {power_after} kỳ (→{power_latest}).',
            }

    except Exception as e:
        mega_c = get_count('mega')
        power_c = get_count('power')
        return {
            'status': 'error',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': f'Dùng data hiện có. Mega: {mega_c} kỳ, Power: {power_c} kỳ.',
            'error': str(e),
        }
    finally:
        _update_lock.release()
