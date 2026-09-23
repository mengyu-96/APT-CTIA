from __future__ import annotations

import streamlit as st

from apt_ui.services import ui
from apt_ui.services.api_client import get_json


def _matches(actor: dict, query: str) -> bool:
    if not query:
        return True
    searchable = [
        actor.get("name", ""),
        actor.get("country", ""),
        *actor.get("aliases", []),
        *actor.get("target_sectors", []),
        *actor.get("tools", []),
        *actor.get("techniques", []),
    ]
    needle = query.casefold()
    return any(needle in str(value).casefold() for value in searchable)


def render_threat_actors() -> None:
    ui.page_header("APT 组织情报库", "查询组织别名、目标行业、常用工具与 TTP", icon="fa-user-secret")

    actors = get_json("/api/gangs", timeout=5, default=[])
    if not actors:
        ui.empty_state("组织情报库暂无数据。", icon="fa-user-secret")
        return

    query = st.text_input(
        "搜索组织、别名、国家/地区、行业、工具或技术",
        placeholder="例如 Lazarus、金融、PowerShell、T1059",
    ).strip()
    filtered = [actor for actor in actors if _matches(actor, query)]
    if not filtered:
        st.info("没有找到匹配的组织画像。")
        return

    st.caption(f"共 {len(filtered)} 个匹配结果；情报画像应结合引用来源和最新情报人工核验。")
    labels = {
        f"{actor.get('name', 'Unknown')} · {actor.get('country', '关联未知')}": actor
        for actor in filtered
    }
    selected_label = st.selectbox("选择组织", list(labels), label_visibility="collapsed")
    actor = labels[selected_label]

    left, right = st.columns([1, 2], gap="large")
    with left:
        with ui.section_card("基础画像", icon="fa-address-card"):
            st.markdown(f"### {actor.get('name', 'Unknown')}")
            st.markdown(f"**国家/地区关联：** {actor.get('country', '未知')}")
            aliases = actor.get("aliases") or []
            st.markdown(f"**别名：** {'、'.join(aliases) if aliases else '暂无'}")
            sectors = actor.get("target_sectors") or []
            st.markdown(f"**主要目标行业：** {'、'.join(sectors) if sectors else '暂无'}")

    with right:
        with ui.section_card("技术与工具", icon="fa-diagram-project"):
            st.markdown("**常见技术 / TTP**")
            techniques = actor.get("techniques") or []
            st.markdown("、".join(f"`{value}`" for value in techniques) if techniques else "暂无")
            st.markdown("**常用工具或恶意软件**")
            tools = actor.get("tools") or []
            st.markdown("、".join(f"`{value}`" for value in tools) if tools else "暂无")

    with ui.section_card("情报来源", icon="fa-link"):
        sources = actor.get("sources") or actor.get("source_urls") or []
        if not sources:
            st.info("暂无引用来源。")
        for source in sources:
            if isinstance(source, dict):
                st.markdown(f"- [{source.get('name', '来源')}]({source.get('url', '#')})")
            else:
                st.markdown(f"- [MITRE ATT&CK 组织条目]({source})")
