from __future__ import annotations

import datetime
import html
import logging
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


LOGGER = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    os.getenv("REPORT_FONT_PATH", "").strip(),
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJKsc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKSC-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _escape(text: Any) -> str:
    return html.escape(str(text or ""))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


class ReportGenerator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_name = self._register_font()
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _register_font(self) -> str:
        for candidate in _FONT_CANDIDATES:
            if not candidate:
                continue
            try:
                font_path = Path(candidate)
                if not font_path.exists():
                    continue
                font_name = f"APTReportFont_{font_path.stem}"
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                LOGGER.info("Using PDF font file: %s", font_path)
                return font_name
            except Exception as exc:
                LOGGER.warning("Failed to register PDF font %s: %s", candidate, exc)

        fallback_name = "STSong-Light"
        try:
            if fallback_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont(fallback_name))
            LOGGER.info("Falling back to built-in CID font: %s", fallback_name)
            return fallback_name
        except Exception as exc:
            LOGGER.warning("Failed to register built-in CID font %s: %s", fallback_name, exc)

        LOGGER.warning("No usable CJK font found for PDF report generation; falling back to Helvetica.")
        return "Helvetica"

    def _setup_styles(self) -> None:
        self.styles.add(
            ParagraphStyle(
                name="CNTitle",
                fontName=self.font_name,
                fontSize=22,
                leading=28,
                alignment=1,
                spaceAfter=16,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CNH1",
                fontName=self.font_name,
                fontSize=16,
                leading=20,
                spaceBefore=12,
                spaceAfter=8,
                textColor=colors.HexColor("#123B66"),
                keepWithNext=True,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CNH2",
                fontName=self.font_name,
                fontSize=12,
                leading=16,
                spaceBefore=8,
                spaceAfter=6,
                textColor=colors.HexColor("#2C4A63"),
                keepWithNext=True,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CNBody",
                fontName=self.font_name,
                fontSize=9.5,
                leading=14,
                spaceAfter=5,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CNBodySmall",
                fontName=self.font_name,
                fontSize=8.3,
                leading=11.5,
                spaceAfter=4,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="CNCap",
                fontName=self.font_name,
                fontSize=8.5,
                leading=11,
                alignment=1,
                textColor=colors.grey,
            )
        )

    def generate_report(
        self,
        task_id: str,
        analysis_results: Dict[str, Any],
        output_filename: str = "report.pdf",
    ) -> str:
        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=42,
            leftMargin=42,
            topMargin=42,
            bottomMargin=28,
        )

        result_records = self._normalize_records(analysis_results.get("result_records"))
        distribution = self._normalize_distribution(
            analysis_results.get("label_distribution"),
            analysis_results.get("attributions", []),
            analysis_results.get("total_samples"),
        )
        total_samples = max(
            int(analysis_results.get("total_samples", 0) or 0),
            len(result_records),
            sum(distribution.values()),
            1,
        )

        top_attr = analysis_results.get("top_attribution") or (max(distribution, key=distribution.get) if distribution else "Unknown")
        top_score = self._find_top_score(top_attr, analysis_results.get("attributions", []), distribution, total_samples)
        explanation = analysis_results.get("explanation") or {}
        sample_explanations = self._top_records(result_records, limit=6)
        if sample_explanations and not explanation:
            explanation = sample_explanations[0].get("explanation", {})
            analysis_results["graph_data"] = analysis_results.get("graph_data") or sample_explanations[0].get("graph_data")
            analysis_results["attention_data"] = analysis_results.get("attention_data") or sample_explanations[0].get("attention_data")

        confidence_stats = self._confidence_stats(result_records)
        dominant_signal = self._dominant_signal(result_records)
        common_entities = self._common_entity_types(result_records)
        common_techniques = self._common_techniques(result_records)
        representative_samples = self._representative_samples(result_records, top_attr)
        evidence_quality = explanation.get("evidence_quality") or {}
        graph_img = None
        if not evidence_quality or evidence_quality.get("reliable", False):
            graph_img = self._generate_graph_image(
                analysis_results.get("graph_data") or explanation.get("graph_data"),
                analysis_results.get("attention_data"),
            )

        story: List[Any] = []
        story.extend(self._build_cover(task_id, total_samples))
        story.extend(
            self._build_summary_section(
                top_attr=top_attr,
                top_score=top_score,
                total_samples=total_samples,
                confidence_stats=confidence_stats,
                dominant_signal=dominant_signal,
                common_techniques=common_techniques,
            )
        )
        story.extend(self._build_overview_metrics(distribution, total_samples, confidence_stats))
        story.extend(self._build_attribution_distribution(distribution, total_samples))
        story.extend(self._build_decision_section(explanation, common_entities, common_techniques))
        if graph_img:
            story.extend(self._build_graph_section(graph_img))
        story.extend(self._build_representative_samples(representative_samples, section_number=6 if graph_img else 5))
        story.extend(self._build_appendix(result_records))

        doc.build(story, onFirstPage=self._draw_page_number, onLaterPages=self._draw_page_number)
        LOGGER.info("Report generated successfully: %s", pdf_path)
        return str(pdf_path)

    def _build_cover(self, task_id: str, total_samples: int) -> List[Any]:
        return [
            Spacer(1, 0.75 * inch),
            Paragraph("APT 归因综合研判报告", self.styles["CNTitle"]),
            Paragraph(f"任务 ID: {_escape(task_id)}", self.styles["CNBody"]),
            Paragraph(f"样本总量: {total_samples}", self.styles["CNBody"]),
            Paragraph(
                f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self.styles["CNBody"],
            ),
            Spacer(1, 0.28 * inch),
            Paragraph(
                "本报告基于整批样本的归因结果、置信度分布、关键证据、MITRE ATT&CK 映射及代表样本解释信息生成，"
                "用于支持归因研判、汇报和后续分析决策。",
                self.styles["CNBody"],
            ),
        ]

    def _build_summary_section(
        self,
        *,
        top_attr: str,
        top_score: float,
        total_samples: int,
        confidence_stats: dict[str, float],
        dominant_signal: str,
        common_techniques: list[tuple[str, int]],
    ) -> List[Any]:
        top_techniques = "、".join(name for name, _ in common_techniques[:5]) if common_techniques else "暂无明显技术聚类"
        summary_lines = [
            f"整体结论：本批 {total_samples} 个样本中，最主要的归因目标为 {_escape(top_attr)}，占比约 {top_score:.2%}。",
            f"置信度概况：平均置信度 {confidence_stats['mean']:.2%}，中位数 {confidence_stats['median']:.2%}，最高 {confidence_stats['max']:.2%}。",
            f"决策信号：当前代表样本中更常见的主导信号为 {_escape(dominant_signal)}。",
            f"技术映射：高频 ATT&CK 技术包括 { _escape(top_techniques) }。",
        ]
        story = [Paragraph("1. 执行摘要", self.styles["CNH1"])]
        for line in summary_lines:
            story.append(Paragraph(line, self.styles["CNBody"]))
        return story

    def _build_overview_metrics(
        self,
        distribution: dict[str, int],
        total_samples: int,
        confidence_stats: dict[str, float],
    ) -> List[Any]:
        actor_count = len(distribution)
        top3 = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:3]
        top3_text = "；".join(f"{name} {count} ({count / total_samples:.2%})" for name, count in top3) if top3 else "无"
        rows = [
            ["指标", "数值", "说明"],
            ["样本总数", str(total_samples), "本次进入归因流程的全部样本数"],
            ["命中组织数", str(actor_count), "至少被一个样本命中的 APT 组织数"],
            ["平均置信度", f"{confidence_stats['mean']:.2%}", "全部样本预测置信度均值"],
            ["高置信样本", str(confidence_stats["high_count"]), "置信度 >= 0.90 的样本数"],
            ["主要归因分布", top3_text, "Top-3 组织占比概览"],
        ]
        return [Paragraph("2. 总体概览", self.styles["CNH1"]), self._build_table(rows, [1.4 * inch, 1.4 * inch, 3.4 * inch])]

    def _build_attribution_distribution(self, distribution: dict[str, int], total_samples: int) -> List[Any]:
        rows = [["APT组织", "样本数", "占比", "风险判断"]]
        ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        for index, (name, count) in enumerate(ordered[:12], start=1):
            ratio = count / total_samples if total_samples else 0.0
            risk = "高" if index <= 2 or ratio >= 0.15 else "中" if ratio >= 0.08 else "低"
            rows.append([name, str(count), f"{ratio:.2%}", risk])
        story = [Paragraph("3. 归因分布", self.styles["CNH1"])]
        story.append(
            Paragraph(
                "下表展示本批样本在不同 APT 组织上的归因集中程度，可用于判断本轮结果是否呈现单一主导目标、双峰分布或明显分散特征。",
                self.styles["CNBody"],
            )
        )
        story.append(self._build_table(rows, [2.2 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch]))
        return story

    def _build_decision_section(
        self,
        explanation: dict[str, Any],
        common_entities: list[tuple[str, int]],
        common_techniques: list[tuple[str, int]],
    ) -> List[Any]:
        story = [Paragraph("4. 证据与决策依据", self.styles["CNH1"])]

        decision_mode = explanation.get("decision_mode") or {}
        if decision_mode:
            story.append(
                Paragraph(
                    (
                        f"当前代表样本的门控决策显示：语义分支权重约 {decision_mode.get('semantic_gate', 0):.2%}，"
                        f"结构分支权重约 {decision_mode.get('structural_gate', 0):.2%}。"
                        f"{_escape(decision_mode.get('description', ''))}"
                    ),
                    self.styles["CNBody"],
                )
            )

        evidence_quality = explanation.get("evidence_quality") or {}
        node_quality = evidence_quality.get("node_attention") or {}
        if node_quality and not node_quality.get("reliable", False):
            key_node_basis = evidence_quality.get("key_node_basis")
            if key_node_basis == "incident_edge_attention":
                quality_message = (
                    "解释质量提示：节点注意力分布接近均匀；下列关键实体由具有区分度的非自环关联边支持，"
                    "证据分数表示相邻关系注意力，而非节点池化注意力。"
                )
            else:
                quality_message = (
                    "解释质量提示：节点注意力分布接近均匀，当前结果不足以形成具有显著区分度的关键证据排序。"
                    "报告不会将任意高分节点表述为可靠关键证据。"
                )
            story.append(
                Paragraph(
                    quality_message,
                    self.styles["CNBody"],
                )
            )

        if common_entities:
            entity_text = "；".join(f"{name} ({count})" for name, count in common_entities[:8])
            story.append(Paragraph(f"高频证据类型：{_escape(entity_text)}。", self.styles["CNBody"]))

        key_nodes = explanation.get("key_nodes") or []
        if key_nodes:
            rows = [["实体", "类型", "证据分数", "证据来源", "段落"]]
            for item in key_nodes[:8]:
                basis = "节点注意力" if item.get("evidence_basis") == "node_attention" else "关联边支持"
                rows.append(
                    [
                        str(item.get("text", "")),
                        str(item.get("type", "")),
                        f"{_safe_float(item.get('evidence_score', item.get('attention'))):.3f}",
                        basis,
                        str(item.get("paragraph_index", "-")),
                    ]
                )
            story.append(Paragraph("4.1 关键实体证据", self.styles["CNH2"]))
            story.append(self._build_table(rows, [2.1 * inch, 1.0 * inch, 0.75 * inch, 1.15 * inch, 0.65 * inch], small=True))
        elif evidence_quality:
            story.append(Paragraph("4.1 关键实体证据", self.styles["CNH2"]))
            story.append(Paragraph("暂无通过区分度校验的关键实体证据。", self.styles["CNBodySmall"]))

        key_edges = explanation.get("key_edges") or []
        if key_edges:
            rows = [["源节点", "目标节点", "关系", "强度"]]
            for item in key_edges[:8]:
                rows.append(
                    [
                        str(item.get("source_text", "")),
                        str(item.get("target_text", "")),
                        str(item.get("relation", "entity_context")),
                        f"{_safe_float(item.get('weight')):.3f}",
                    ]
                )
            story.append(Paragraph("4.2 关键关联边", self.styles["CNH2"]))
            story.append(self._build_table(rows, [2.0 * inch, 2.0 * inch, 1.25 * inch, 0.65 * inch], small=True))

        evidence_paths = explanation.get("evidence_paths") or []
        if evidence_paths:
            story.append(Paragraph("4.3 核心证据路径", self.styles["CNH2"]))
            for idx, item in enumerate(evidence_paths[:5], start=1):
                path_text = " -> ".join(str(part) for part in item.get("path_texts", []))
                story.append(
                    Paragraph(
                        f"路径 {idx}: {_escape(path_text)}",
                        self.styles["CNBodySmall"],
                    )
                )

        if common_techniques:
            rows = [["Technique", "出现次数"]]
            for technique, count in common_techniques[:10]:
                rows.append([technique, str(count)])
            story.append(Paragraph("4.4 MITRE ATT&CK 高频技术", self.styles["CNH2"]))
            story.append(self._build_table(rows, [3.8 * inch, 1.2 * inch], small=True))

        return story

    def _build_graph_section(self, graph_img: str) -> List[Any]:
        return [
            Paragraph("5. 证据图谱", self.styles["CNH1"]),
            Paragraph(
                "以下图谱展示代表样本中最重要的节点与边，节点大小和颜色反映贡献度，边的粗细反映结构关联强度。",
                self.styles["CNBody"],
            ),
            Image(graph_img, width=6.1 * inch, height=4.3 * inch),
            Paragraph("图 1. 代表样本证据图谱", self.styles["CNCap"]),
        ]

    def _build_representative_samples(
        self,
        samples: list[dict[str, Any]],
        *,
        section_number: int,
    ) -> List[Any]:
        story = [Paragraph(f"{section_number}. 代表样本分析", self.styles["CNH1"])]
        if not samples:
            story.append(Paragraph("当前没有可用于展开说明的代表样本。", self.styles["CNBody"]))
            return story

        for idx, sample in enumerate(samples[:5], start=1):
            explanation = sample.get("explanation") or {}
            top3 = sample.get("top3") or []
            top3_text = "；".join(
                f"{item.get('label', 'Unknown')} { _safe_float(item.get('score')):.2%}" for item in top3[:3]
            ) or "无"
            summary_lines = explanation.get("summary_lines") or []
            story.append(
                Paragraph(
                    f"{section_number}.{idx} 样本 { _escape(sample.get('report_id', 'Unknown')) }",
                    self.styles["CNH2"],
                )
            )
            story.append(
                Paragraph(
                    (
                        f"预测结果：{_escape(sample.get('predicted_label', 'Unknown'))}；"
                        f"置信度：{_safe_float(sample.get('confidence')):.2%}；"
                        f"Top-3 候选：{_escape(top3_text)}。"
                    ),
                    self.styles["CNBody"],
                )
            )
            evidence_quality = explanation.get("evidence_quality") or {}
            if evidence_quality and not evidence_quality.get("reliable", False):
                story.append(
                    Paragraph(
                        "解释证据区分度不足，未展示关键节点和关联边排序。",
                        self.styles["CNBodySmall"],
                    )
                )
            else:
                for line in summary_lines[:3]:
                    story.append(Paragraph(_escape(line), self.styles["CNBodySmall"]))
        return story

    def _build_appendix(self, records: list[dict[str, Any]]) -> List[Any]:
        story = [Paragraph("附录 A. 样本明细", self.styles["CNH1"])]
        rows = [["样本ID", "预测组织", "置信度", "Top-2 候选"]]
        for item in self._top_records(records, limit=20):
            top3 = item.get("top3") or []
            alt = "；".join(
                f"{cand.get('label', 'Unknown')} { _safe_float(cand.get('score')):.2%}" for cand in top3[:2]
            )
            rows.append(
                [
                    str(item.get("report_id", "")),
                    str(item.get("predicted_label", "")),
                    f"{_safe_float(item.get('confidence')):.2%}",
                    alt,
                ]
            )
        story.append(
            Paragraph(
                "附录列出高置信度样本的基本信息，便于人工快速抽检和追溯具体样本。",
                self.styles["CNBody"],
            )
        )
        story.append(self._build_table(rows, [2.0 * inch, 1.3 * inch, 0.9 * inch, 2.2 * inch], small=True))
        return story

    def _build_table(self, rows: List[List[str]], widths: List[float], *, small: bool = False) -> Table:
        available_width = A4[0] - 84
        if sum(widths) > available_width:
            scale = available_width / sum(widths)
            widths = [width * scale for width in widths]
        flowable_rows = [
            [self._table_cell(value, header=row_index == 0, small=small) for value in row]
            for row_index, row in enumerate(rows)
        ]
        table = LongTable(flowable_rows, colWidths=widths, repeatRows=1, splitByRow=1)
        font_size = 8 if small else 9
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B66")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, -1), self.font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), font_size),
                    ("LEADING", (0, 0), (-1, -1), 10 if small else 11),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C3CE")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F5F8FB")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        table.hAlign = "LEFT"
        return table

    def _table_cell(self, value: Any, *, header: bool, small: bool) -> Paragraph:
        style = ParagraphStyle(
            name=f"TableCell-{header}-{small}",
            parent=self.styles["CNBodySmall" if small else "CNBody"],
            fontName=self.font_name,
            fontSize=8 if small else 9,
            leading=10.5 if small else 12,
            textColor=colors.white if header else colors.HexColor("#1F2933"),
            wordWrap="CJK",
            splitLongWords=1,
            spaceBefore=0,
            spaceAfter=0,
        )
        text = _escape(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
        return Paragraph(text, style)

    def _draw_page_number(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(self.font_name, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawRightString(A4[0] - 42, 16, f"第 {doc.page} 页")
        canvas.restoreState()

    def _normalize_records(self, raw_records: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_records, list):
            return []
        return [item for item in raw_records if isinstance(item, dict)]

    def _normalize_distribution(
        self,
        raw_distribution: Any,
        attributions: Iterable[dict[str, Any]],
        total_samples: Any,
    ) -> dict[str, int]:
        if isinstance(raw_distribution, dict) and raw_distribution:
            return {str(key): int(value) for key, value in raw_distribution.items()}

        total = max(int(total_samples or 0), 1)
        derived: dict[str, int] = {}
        for item in attributions:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "Unknown"))
            ratio = _safe_float(item.get("score"))
            derived[name] = max(int(round(ratio * total)), 0)
        return derived

    def _find_top_score(self, top_attr: str, attributions: Any, distribution: dict[str, int], total_samples: int) -> float:
        if isinstance(attributions, list):
            for item in attributions:
                if isinstance(item, dict) and item.get("name") == top_attr:
                    return _safe_float(item.get("score"))
        if distribution and total_samples:
            return distribution.get(top_attr, 0) / total_samples
        return 0.0

    def _confidence_stats(self, records: list[dict[str, Any]]) -> dict[str, float]:
        values = [_safe_float(item.get("confidence")) for item in records if item.get("confidence") is not None]
        if not values:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "high_count": 0}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "high_count": int(sum(1 for value in values if value >= 0.90)),
        }

    def _dominant_signal(self, records: list[dict[str, Any]]) -> str:
        signals = Counter()
        for item in records[:30]:
            explanation = item.get("explanation") or {}
            decision_mode = explanation.get("decision_mode") or {}
            signal = decision_mode.get("dominant_signal")
            if signal:
                signals[str(signal)] += 1
        return signals.most_common(1)[0][0] if signals else "unknown"

    def _common_entity_types(self, records: list[dict[str, Any]]) -> list[tuple[str, int]]:
        counter = Counter()
        for item in self._top_records(records, limit=30):
            explanation = item.get("explanation") or {}
            for node in explanation.get("key_nodes", [])[:8]:
                counter[str(node.get("type", "UNKNOWN"))] += 1
        return counter.most_common()

    def _common_techniques(self, records: list[dict[str, Any]]) -> list[tuple[str, int]]:
        counter = Counter()
        for item in self._top_records(records, limit=30):
            explanation = item.get("explanation") or {}
            techniques = (explanation.get("mitre_attack") or {}).get("techniques") or []
            for technique in techniques[:8]:
                counter[str(technique.get("technique_id", ""))] += 1
        return counter.most_common()

    def _representative_samples(self, records: list[dict[str, Any]], top_attr: str) -> list[dict[str, Any]]:
        selected = []
        seen = set()
        for item in self._top_records(records, limit=40):
            label = str(item.get("predicted_label", ""))
            if label == top_attr or len(selected) < 2:
                report_id = str(item.get("report_id", ""))
                if report_id and report_id not in seen:
                    selected.append(item)
                    seen.add(report_id)
            if len(selected) >= 5:
                break
        return selected

    def _top_records(self, records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        return sorted(records, key=lambda item: _safe_float(item.get("confidence")), reverse=True)[:limit]

    def _generate_graph_image(
        self,
        graph_data: Optional[Dict[str, Any]],
        attention_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        del attention_data
        if not graph_data or "nodes" not in graph_data or "edges" not in graph_data:
            return None

        try:
            graph = nx.Graph()
            for node in graph_data.get("nodes", []):
                graph.add_node(
                    int(node.get("id")),
                    label=str(node.get("label", "")),
                    importance=float(node.get("importance", 0.0) or 0.0),
                )
            for edge in graph_data.get("edges", []):
                graph.add_edge(
                    int(edge.get("source")),
                    int(edge.get("target")),
                    weight=float(edge.get("weight", 0.0) or 0.0),
                )
            if graph.number_of_nodes() == 0:
                return None

            plt.figure(figsize=(10, 7))
            pos = nx.spring_layout(graph, seed=42, k=0.45)
            node_sizes = [200 + graph.nodes[n].get("importance", 0.0) * 1400 for n in graph.nodes()]
            node_colors = [graph.nodes[n].get("importance", 0.0) for n in graph.nodes()]
            edge_widths = [1.0 + graph.edges[e].get("weight", 0.0) * 4.5 for e in graph.edges()]

            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=node_colors,
                cmap=plt.cm.YlOrRd,
                node_size=node_sizes,
                alpha=0.92,
            )
            nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#7D8CA3", alpha=0.65)

            labels = {}
            top_nodes = sorted(
                graph.nodes(),
                key=lambda node_id: graph.nodes[node_id].get("importance", 0.0),
                reverse=True,
            )[:8]
            for node_id in top_nodes:
                labels[node_id] = graph.nodes[node_id].get("label", "")
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_family="sans-serif")

            img_path = self.output_dir / "temp_graph.png"
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(img_path, format="png", dpi=220, bbox_inches="tight")
            plt.close()
            return str(img_path)
        except Exception as exc:
            LOGGER.error("Failed to generate graph image: %s", exc)
            plt.close()
            return None
