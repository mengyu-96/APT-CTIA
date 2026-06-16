from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


LOGGER = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font_name = self._register_font()
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _register_font(self) -> str:
        try:
            font_path = r"C:\Windows\Fonts\simhei.ttf"
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont("SimHei", font_path))
                return "SimHei"
        except Exception:
            pass
        return "Helvetica"

    def _setup_styles(self) -> None:
        self.styles.add(ParagraphStyle(
            name="CNTitle",
            fontName=self.font_name,
            fontSize=22,
            leading=28,
            alignment=1,
            spaceAfter=16,
        ))
        self.styles.add(ParagraphStyle(
            name="CNH1",
            fontName=self.font_name,
            fontSize=16,
            leading=20,
            spaceBefore=10,
            spaceAfter=8,
            textColor=colors.HexColor("#123b66"),
        ))
        self.styles.add(ParagraphStyle(
            name="CNH2",
            fontName=self.font_name,
            fontSize=12,
            leading=16,
            spaceBefore=8,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name="CNBody",
            fontName=self.font_name,
            fontSize=9.5,
            leading=14,
            spaceAfter=5,
        ))
        self.styles.add(ParagraphStyle(
            name="CNCap",
            fontName=self.font_name,
            fontSize=8.5,
            leading=11,
            alignment=1,
            textColor=colors.grey,
        ))

    def generate_report(self, task_id: str, analysis_results: Dict[str, Any], output_filename: str = "report.pdf") -> str:
        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=48,
            leftMargin=48,
            topMargin=48,
            bottomMargin=24,
        )
        story: List[Any] = []

        explanation = analysis_results.get("explanation") or {}
        sample_explanations = analysis_results.get("sample_explanations") or []
        if sample_explanations and not explanation:
            explanation = sample_explanations[0].get("explanation", {})
            analysis_results["graph_data"] = analysis_results.get("graph_data") or sample_explanations[0].get("graph_data")
            analysis_results["attention_data"] = analysis_results.get("attention_data") or sample_explanations[0].get("attention_data")

        top_attr = analysis_results.get("top_attribution", "Unknown")
        top_score = 0.0
        for item in analysis_results.get("attributions", []):
            if item.get("name") == top_attr:
                top_score = float(item.get("score", 0.0) or 0.0)
                break

        story.append(Spacer(1, 0.6 * inch))
        story.append(Paragraph("APT 归因可解释研判报告", self.styles["CNTitle"]))
        story.append(Paragraph(f"任务 ID: {task_id}", self.styles["CNBody"]))
        story.append(Paragraph(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles["CNBody"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("1. 执行摘要", self.styles["CNH1"]))
        summary = (
            f"系统对目标样本进行了自动归因分析，当前最可能的APT组织为 <b>{top_attr}</b>，"
            f"综合置信度为 <b>{top_score:.2%}</b>。"
        )
        story.append(Paragraph(summary, self.styles["CNBody"]))
        for line in explanation.get("summary_lines", [])[:4]:
            story.append(Paragraph(line, self.styles["CNBody"]))

        story.append(Paragraph("2. 归因结果", self.styles["CNH1"]))
        table_rows = [["APT组织", "得分", "风险等级"]]
        for item in analysis_results.get("attributions", [])[:10]:
            table_rows.append([
                str(item.get("name", "Unknown")),
                f"{float(item.get('score', 0.0) or 0.0):.2%}",
                str(item.get("risk", "Medium")),
            ])
        story.append(self._build_table(table_rows, [2.8 * inch, 1.3 * inch, 1.4 * inch]))

        decision_mode = explanation.get("decision_mode", {})
        if decision_mode:
            story.append(Paragraph("3. 决策偏好", self.styles["CNH1"]))
            story.append(Paragraph(
                f"语义分支权重: {decision_mode.get('semantic_gate', 0):.2%}；"
                f"结构分支权重: {decision_mode.get('structural_gate', 0):.2%}；"
                f"{decision_mode.get('description', '')}",
                self.styles["CNBody"],
            ))

        key_nodes = explanation.get("key_nodes", [])
        if key_nodes:
            story.append(Paragraph("4. 关键实体证据", self.styles["CNH1"]))
            node_rows = [["实体", "类型", "注意力"]]
            for item in key_nodes[:8]:
                node_rows.append([
                    str(item.get("text", "")),
                    str(item.get("type", "")),
                    f"{float(item.get('attention', 0.0) or 0.0):.3f}",
                ])
            story.append(self._build_table(node_rows, [3.4 * inch, 1.5 * inch, 0.9 * inch]))

        key_edges = explanation.get("key_edges", [])
        if key_edges:
            story.append(Paragraph("5. 关键关联边", self.styles["CNH1"]))
            edge_rows = [["源节点", "目标节点", "权重"]]
            for item in key_edges[:8]:
                edge_rows.append([
                    str(item.get("source_text", "")),
                    str(item.get("target_text", "")),
                    f"{float(item.get('weight', 0.0) or 0.0):.3f}",
                ])
            story.append(self._build_table(edge_rows, [2.6 * inch, 2.6 * inch, 0.8 * inch]))

        evidence_paths = explanation.get("evidence_paths", [])
        if evidence_paths:
            story.append(Paragraph("6. 决策路径", self.styles["CNH1"]))
            for idx, item in enumerate(evidence_paths[:5], start=1):
                story.append(Paragraph(
                    f"路径 {idx}: {' -> '.join(item.get('path_texts', []))}",
                    self.styles["CNBody"],
                ))

        mitre_attack = explanation.get("mitre_attack", {})
        if mitre_attack.get("techniques"):
            story.append(Paragraph("7. MITRE ATT&CK 映射", self.styles["CNH1"]))
            mitre_rows = [["Technique", "Tactics", "Score"]]
            for item in mitre_attack.get("techniques", [])[:8]:
                mitre_rows.append([
                    str(item.get("technique_id", "")),
                    ", ".join(item.get("tactics", [])) or "-",
                    f"{float(item.get('score', 0.0) or 0.0):.3f}",
                ])
            story.append(self._build_table(mitre_rows, [1.1 * inch, 4.2 * inch, 0.8 * inch]))

        graph_img = self._generate_graph_image(
            analysis_results.get("graph_data") or explanation.get("graph_data"),
            analysis_results.get("attention_data"),
        )
        if graph_img:
            story.append(Paragraph("8. 证据图谱", self.styles["CNH1"]))
            story.append(Image(graph_img, width=6.2 * inch, height=4.4 * inch))
            story.append(Paragraph("图中节点颜色与大小反映实体贡献度，边粗细反映关联强度。", self.styles["CNCap"]))

        iocs = analysis_results.get("iocs") or {}
        if iocs:
            story.append(Paragraph("9. IOC 摘要", self.styles["CNH1"]))
            for key, values in iocs.items():
                vals = values[:10] if isinstance(values, list) else [str(values)]
                story.append(Paragraph(f"{key}: {', '.join(str(v) for v in vals)}", self.styles["CNBody"]))

        doc.build(story)
        LOGGER.info("Report generated successfully: %s", pdf_path)
        return str(pdf_path)

    def _build_table(self, rows: List[List[str]], widths: List[float]) -> Table:
        table = Table(rows, colWidths=widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b66")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, -1), self.font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story_spacer = Spacer(1, 0.1 * inch)
        table.hAlign = "LEFT"
        return table

    def _generate_graph_image(self, graph_data: Optional[Dict[str, Any]], attention_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
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
            pos = nx.spring_layout(graph, seed=42, k=0.4)
            node_sizes = [180 + graph.nodes[n].get("importance", 0.0) * 1200 for n in graph.nodes()]
            node_colors = [graph.nodes[n].get("importance", 0.0) for n in graph.nodes()]
            edge_widths = [0.8 + graph.edges[e].get("weight", 0.0) * 4.0 for e in graph.edges()]

            nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.YlOrRd, node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#7d8ca3", alpha=0.6)

            labels = {}
            top_nodes = sorted(graph.nodes(), key=lambda n: graph.nodes[n].get("importance", 0.0), reverse=True)[:6]
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
