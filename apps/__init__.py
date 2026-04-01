"""
앱 레지스트리 - apps/ 하위 모듈 자동 검색
새 앱 추가: apps/{app_name}/config.py (APP_CONFIG) + apps/{app_name}/page.py (렌더 함수)
"""

import importlib
import os
import pkgutil


def discover_apps() -> dict:
    """
    apps/ 디렉토리에서 config.py + page.py를 가진 모듈을 자동 검색하여 반환.

    Returns:
        dict: {app_id: {"config": APP_CONFIG, "page": page_module}, ...}
    """
    apps = {}
    apps_dir = os.path.dirname(__file__)
    for _finder, name, ispkg in pkgutil.iter_modules([apps_dir]):
        if ispkg:
            try:
                config_mod = importlib.import_module(f"apps.{name}.config")
                page_mod = importlib.import_module(f"apps.{name}.page")
                if hasattr(config_mod, "APP_CONFIG"):
                    config = config_mod.APP_CONFIG
                    # page 모듈 필수 함수 검증
                    for fn in ("init_app_session", "render_sidebar", "render_main"):
                        if not hasattr(page_mod, fn):
                            raise AttributeError(
                                f"apps.{name}.page에 {fn}() 함수가 없습니다"
                            )
                    apps[config["id"]] = {
                        "config": config,
                        "page": page_mod,
                    }
            except Exception as e:
                print(f"⚠️ apps.{name} 로드 실패: {e}")
    return apps
