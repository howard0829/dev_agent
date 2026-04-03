"""
워크스페이스 탭 - 파일 관리 UI
FastAPI 파일 서버와 통신하여 파일 업로드/다운로드/편집 기능 제공
"""

import os
import hashlib as _hl
import httpx
import streamlit as st

from core.session import ns, get_state, set_state
from config import FILE_SERVER_URL


def render_workspace_tab(prefix: str):
    """워크스페이스 파일 관리 탭 렌더링"""

    # ── 서버 연결 확인 ──
    server_ok = False
    ws_size, ws_quota, session_id = "?", "?", ""
    try:
        si = httpx.get(f"{FILE_SERVER_URL}/api/session", timeout=3)
        if si.status_code == 200:
            info = si.json()
            ws_path = info.get("workspace_path", "")
            if ws_path and os.path.isdir(ws_path):
                set_state(prefix, "working_dir", ws_path)
            ws_size = info.get("workspace_size", "?")
            ws_quota = info.get("workspace_quota", "?")
            session_id = info.get("session_id", "")
            server_ok = True
        else:
            st.warning("⚠️ 파일 서버 응답 오류")
    except Exception:
        st.warning("⚠️ 파일 서버 미연결 — `python server.py` 실행 필요")

    if not server_ok:
        return

    # ── 헤더 정보 ──
    hdr_a, hdr_b = st.columns([5, 1])
    with hdr_a:
        st.markdown(
            f"**📁 워크스페이스**  ·  💾 `{ws_size}` / `{ws_quota}`  ·  🔑 `{session_id}`"
        )
        st.caption(
            "📌 파일당 최대 100MB · 전체 용량 최대 100MB · "
            "허용 확장자: md, txt, py, json, yaml, csv, html, js, sh, env, toml, cfg, log"
        )
    with hdr_b:
        if st.button("🔄 새로고침", key=f"{prefix}_ws_reload", use_container_width=True):
            st.rerun()

    st.divider()
    file_col, editor_col = st.columns([2, 3])

    # ── 파일 목록 (루트 기준 플랫 리스트) ──
    with file_col:
        try:
            lr = httpx.get(
                f"{FILE_SERVER_URL}/api/files/listdir",
                params={"path": ""},
                timeout=5,
            )
            files = (
                [i for i in lr.json().get("items", []) if i["type"] == "file"]
                if lr.status_code == 200
                else []
            )
        except Exception:
            files = []

        if files:
            st.caption(f"📄 파일 ({len(files)}개)")
            for fitem in files:
                fname = fitem["name"]
                fsize = fitem.get("size", "")
                fmod = fitem.get("modified", "")
                fc1, fc2, fc3 = st.columns([5, 1, 1])
                with fc1:
                    if st.button(
                        f"📄 {fname}",
                        key=f"{prefix}_opn_{fname}",
                        help=f"{fsize}  ·  {fmod}",
                    ):
                        set_state(prefix, "editing_file", fname)
                with fc2:
                    if st.button("⬇", key=f"{prefix}_dlb_{fname}", help="다운로드"):
                        st.session_state[f"{prefix}_dl_{fname}"] = True
                    if st.session_state.pop(f"{prefix}_dl_{fname}", False):
                        try:
                            dl = httpx.get(
                                f"{FILE_SERVER_URL}/api/files/download/{fname}",
                                timeout=10,
                            )
                            if dl.status_code == 200:
                                st.download_button(
                                    "💾",
                                    data=dl.content,
                                    file_name=fname,
                                    key=f"{prefix}_dls_{fname}",
                                )
                        except Exception:
                            pass
                with fc3:
                    if st.button("🗑", key=f"{prefix}_del_{fname}"):
                        try:
                            r = httpx.delete(
                                f"{FILE_SERVER_URL}/api/files/{fname}", timeout=5
                            )
                            if r.status_code == 200:
                                if get_state(prefix, "editing_file") == fname:
                                    set_state(prefix, "editing_file", None)
                                st.rerun()
                        except Exception:
                            pass
        else:
            st.caption("업로드된 파일이 없습니다.")

        # ── 하단 액션 ──
        st.divider()
        action = st.radio(
            "작업",
            ["📄 새 파일", "⬆ 업로드"],
            horizontal=True,
            label_visibility="collapsed",
            key=f"{prefix}_ws_act",
        )

        if action == "📄 새 파일":
            nfn = st.text_input("파일명", key=f"{prefix}_ws_nfn", placeholder="예: notes.md")
            if (
                st.button("만들기", key=f"{prefix}_ws_mknfn", use_container_width=True)
                and nfn.strip()
            ):
                try:
                    wr = httpx.post(
                        f"{FILE_SERVER_URL}/api/files/write",
                        json={"path": nfn.strip(), "content": ""},
                        timeout=5,
                    )
                    if wr.status_code == 200:
                        set_state(prefix, "editing_file", nfn.strip())
                        st.rerun()
                    else:
                        st.error(wr.json().get("detail", "생성 실패"))
                except Exception as e:
                    st.error(f"❌ {e}")

        else:  # 업로드
            uploaded = st.file_uploader(
                "파일 선택 (최대 100MB)",
                label_visibility="visible",
                type=[
                    e.lstrip(".")
                    for e in [
                        ".md", ".txt", ".py", ".json", ".yaml", ".yml",
                        ".csv", ".html", ".js", ".ts", ".sh", ".env",
                        ".toml", ".ini", ".cfg", ".log",
                    ]
                ],
                key=f"{prefix}_ws_upl",
            )
            if uploaded is not None:
                fhash = _hl.md5(uploaded.getvalue()).hexdigest()
                if fhash != get_state(prefix, "_uploaded_hash"):
                    try:
                        resp = httpx.post(
                            f"{FILE_SERVER_URL}/api/files/upload",
                            files={
                                "file": (
                                    uploaded.name,
                                    uploaded.getvalue(),
                                    uploaded.type or "application/octet-stream",
                                )
                            },
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            set_state(prefix, "_uploaded_hash", fhash)
                            st.success(f"✅ '{uploaded.name}' 업로드 완료")
                            st.rerun()
                        else:
                            st.error(resp.json().get("detail", "업로드 실패"))
                    except Exception as e:
                        st.error(f"❌ {e}")

    # ── 파일 편집기 ──
    with editor_col:
        editing = get_state(prefix, "editing_file")
        if editing:
            st.markdown(f"✏️ **편집:** `{editing}`")
            try:
                rr = httpx.get(
                    f"{FILE_SERVER_URL}/api/files/read/{editing}", timeout=5
                )
                if rr.status_code == 200:
                    orig = rr.json().get("content", "")
                    edited = st.text_area(
                        "내용",
                        value=orig,
                        height=500,
                        key=f"{prefix}_ed_{editing}",
                        label_visibility="collapsed",
                    )
                    sc, cc = st.columns([1, 1])
                    with sc:
                        if st.button(
                            "💾 저장",
                            key=f"{prefix}_sv_{editing}",
                            use_container_width=True,
                        ):
                            try:
                                wr = httpx.post(
                                    f"{FILE_SERVER_URL}/api/files/write",
                                    json={"path": editing, "content": edited},
                                    timeout=10,
                                )
                                if wr.status_code == 200:
                                    st.success("✅ 저장 완료")
                                else:
                                    st.error(wr.json().get("detail", "저장 실패"))
                            except Exception as ex:
                                st.error(f"❌ {ex}")
                    with cc:
                        if st.button(
                            "✖ 닫기",
                            key=f"{prefix}_cl_{editing}",
                            use_container_width=True,
                        ):
                            set_state(prefix, "editing_file", None)
                            st.rerun()
                elif rr.status_code == 400:
                    st.info("🔒 바이너리 파일은 텍스트 편집 불가")
                    if st.button("닫기", key=f"{prefix}_bin_cl"):
                        set_state(prefix, "editing_file", None)
                        st.rerun()
            except Exception as ex:
                st.error(f"❌ {ex}")
        else:
            st.markdown("✏️ **파일 편집기**")
            st.caption("← 왼쪽에서 파일을 클릭하면 여기서 편집할 수 있습니다.")
