"""
Auto-Updater — Incremental scraper that runs on app startup.
Only scrapes draws newer than the latest date in database.
Uses 6-hour cooldown to avoid redundant scraping.
"""
import time
import os
import json
import threading
from datetime import datetime, timedelta

# Cooldown file path (stores last update timestamp)
_COOLDOWN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'data', '.last_update'
)
_COOLDOWN_HOURS = 6
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
            json.dump({'timestamp': datetime.now().isoformat()}, f)
    except Exception:
        pass


def _needs_update():
    """Check if we should attempt an update based on cooldown."""
    last = _get_last_update_time()
    elapsed = datetime.now() - last
    return elapsed > timedelta(hours=_COOLDOWN_HOURS)


def _format_date_for_site(dt):
    """Convert datetime to dd-mm-yyyy for the website input."""
    return dt.strftime('%d-%m-%Y')


def auto_update_data():
    """
    Auto-update lottery data if cooldown has passed.
    
    Returns:
        dict with keys:
            - 'status': 'updated' | 'skipped' | 'error'
            - 'mega_count': total mega draws after update
            - 'power_count': total power draws after update
            - 'mega_new': new mega draws added (0 if skipped)
            - 'power_new': new power draws added (0 if skipped)
            - 'message': human-readable status message
            - 'error': error message if status == 'error'
    """
    from scraper.data_manager import get_count, get_latest_date

    # Check cooldown
    if not _needs_update():
        mega_c = get_count('mega')
        power_c = get_count('power')
        return {
            'status': 'skipped',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': f'Data đã cập nhật gần đây (cooldown {_COOLDOWN_HOURS}h). '
                       f'Mega: {mega_c} kỳ, Power: {power_c} kỳ.',
        }

    # Check if data is already up to date (latest draw date == today or yesterday)
    mega_latest = get_latest_date('mega')
    power_latest = get_latest_date('power')
    today = datetime.now().date()

    # Vietlott draws: Mega Wed/Fri/Sun, Power Mon/Wed/Thu/Sat
    # If latest is today or yesterday, probably up to date
    mega_up_to_date = False
    power_up_to_date = False
    if mega_latest:
        try:
            ld = datetime.strptime(mega_latest, '%Y-%m-%d').date()
            if (today - ld).days <= 1:
                mega_up_to_date = True
        except Exception:
            pass
    if power_latest:
        try:
            ld = datetime.strptime(power_latest, '%Y-%m-%d').date()
            if (today - ld).days <= 1:
                power_up_to_date = True
        except Exception:
            pass

    if mega_up_to_date and power_up_to_date:
        _save_update_time()
        mega_c = get_count('mega')
        power_c = get_count('power')
        return {
            'status': 'skipped',
            'mega_count': mega_c,
            'power_count': power_c,
            'mega_new': 0,
            'power_new': 0,
            'message': f'Data đã mới nhất. Mega: {mega_c} kỳ (→ {mega_latest}), '
                       f'Power: {power_c} kỳ (→ {power_latest}).',
        }

    # Need to scrape — acquire lock to prevent concurrent scraping
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

        try:
            from scraper.scraper import create_driver, scrape_mega645, scrape_power655
        except ImportError:
            # Selenium not available (e.g. Streamlit Cloud) — skip gracefully
            _save_update_time()
            return {
                'status': 'skipped',
                'mega_count': mega_before,
                'power_count': power_before,
                'mega_new': 0,
                'power_new': 0,
                'message': 'Auto-update không khả dụng trên cloud. Dùng data hiện có.',
            }

        driver = create_driver()
        try:
            scrape_mega645(driver)
            scrape_power655(driver)
        finally:
            driver.quit()

        mega_after = get_count('mega')
        power_after = get_count('power')
        mega_new = mega_after - mega_before
        power_new = power_after - power_before

        _save_update_time()

        if mega_new > 0 or power_new > 0:
            return {
                'status': 'updated',
                'mega_count': mega_after,
                'power_count': power_after,
                'mega_new': mega_new,
                'power_new': power_new,
                'message': f'✅ Đã cập nhật thành công! '
                           f'Mega: +{mega_new} kỳ (tổng {mega_after}), '
                           f'Power: +{power_new} kỳ (tổng {power_after}).',
            }
        else:
            return {
                'status': 'updated',
                'mega_count': mega_after,
                'power_count': power_after,
                'mega_new': 0,
                'power_new': 0,
                'message': f'Kiểm tra xong — không có kỳ mới. '
                           f'Mega: {mega_after} kỳ, Power: {power_after} kỳ.',
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
            'message': f'⚠️ Cập nhật thất bại: {str(e)[:100]}',
            'error': str(e),
        }
    finally:
        _update_lock.release()
