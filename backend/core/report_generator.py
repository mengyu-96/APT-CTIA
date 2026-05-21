import os
import datetime
import io
import json
import logging
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates PDF reports for APT attribution analysis.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register a font that supports Chinese characters if available
        # Ideally, we should ship a font file with the project.
        # For now, we'll try to use a system font or fallback to standard.
        try:
            # Common path for SimHei on Windows
            font_path = "C:\\Windows\\Fonts\\simhei.ttf" 
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('SimHei', font_path))
                self.font_name = 'SimHei'
            else:
                self.font_name = 'Helvetica' # Fallback
        except:
            self.font_name = 'Helvetica'

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(name='ChineseTitle', fontName=self.font_name, fontSize=24, leading=30, alignment=1, spaceAfter=20))
        self.styles.add(ParagraphStyle(name='ChineseHeading1', fontName=self.font_name, fontSize=18, leading=22, spaceBefore=15, spaceAfter=10, textColor=colors.HexColor('#003366')))
        self.styles.add(ParagraphStyle(name='ChineseHeading2', fontName=self.font_name, fontSize=14, leading=18, spaceBefore=10, spaceAfter=5))
        self.styles.add(ParagraphStyle(name='ChineseBody', fontName=self.font_name, fontSize=10, leading=14, spaceAfter=5))
        self.styles.add(ParagraphStyle(name='ChineseCaption', fontName=self.font_name, fontSize=9, leading=12, alignment=1, textColor=colors.grey))

    def generate_report(self, task_id, analysis_results, output_filename="report.pdf"):
        """
        Generates the full report.
        """
        pdf_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title Page
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("APT 归因分析报告", self.styles['ChineseTitle']))
        story.append(Paragraph(f"任务 ID: {task_id}", self.styles['ChineseBody']))
        story.append(Paragraph(f"生成日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['ChineseBody']))
        story.append(Spacer(1, 1*inch))
        
        # Executive Summary
        story.append(Paragraph("1. 执行摘要 (Executive Summary)", self.styles['ChineseHeading1']))
        
        top_attribution = analysis_results.get('top_attribution', 'Unknown')
        confidence_level = "中等"
        for attr in analysis_results.get('attributions', []):
             if attr['name'] == top_attribution:
                 score = attr.get('score', 0)
                 if score > 0.8: confidence_level = "极高"
                 elif score > 0.6: confidence_level = "高"
                 break

        summary_text = (
            f"本报告针对任务 <b>{task_id}</b> 提交的威胁情报样本进行了深入的自动化溯源分析。<br/><br/>"
            "系统利用 <b>RGAT (Relation-aware Graph Attention Network)</b> 深度学习模型，"
            "对样本的文本语义特征与行为拓扑结构进行了双流提取与聚类。<br/><br/>"
            f"分析结论显示，目标样本与已知 APT 组织 <b>{top_attribution}</b> 具有高度同源性（置信度：{confidence_level}）。"
            "该组织通常针对政府、能源及金融领域发起定向攻击，请相关单位立即启动应急响应流程。"
        )
        story.append(Paragraph(summary_text, self.styles['ChineseBody']))

        # Analysis Details
        story.append(Paragraph("2. 详细分析 (Analysis Details)", self.styles['ChineseHeading1']))
        
        # Attribution Result Table
        story.append(Paragraph("2.1 归因结果", self.styles['ChineseHeading2']))
        data = [['APT 组织', '置信度 (Confidence)', '风险等级']]
        
        for attr in analysis_results.get('attributions', []):
            data.append([attr['name'], f"{attr['score']:.2%}", attr['risk']])
            
        t = Table(data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))

        # Visualization (Graph)
        story.append(Paragraph("2.2 行为图谱可视化", self.styles['ChineseHeading2']))
        graph_img_path = self._generate_graph_image(
            analysis_results.get('graph_data'),
            analysis_results.get('attention_data')
        )
        if graph_img_path:
            img = Image(graph_img_path, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Paragraph("图 1: 样本行为与实体关联图 (高亮显示 RGAT 关键路径)", self.styles['ChineseCaption']))
        
        # Training Plots (New)
        story.append(Paragraph("2.3 训练评估可视化", self.styles['ChineseHeading2']))
        
        # Confusion Matrix
        if analysis_results.get('confusion_matrix_plot') and os.path.exists(analysis_results['confusion_matrix_plot']):
            try:
                img_cm = Image(analysis_results['confusion_matrix_plot'], width=5*inch, height=4*inch)
                story.append(img_cm)
                story.append(Paragraph("图 2: 混淆矩阵 (Confusion Matrix)", self.styles['ChineseCaption']))
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                LOGGER.error(f"Failed to add CM image: {e}")

        # Training History
        if analysis_results.get('history_plot') and os.path.exists(analysis_results['history_plot']):
            try:
                img_hist = Image(analysis_results['history_plot'], width=6*inch, height=2.5*inch)
                story.append(img_hist)
                story.append(Paragraph("图 3: 训练过程曲线 (Training History)", self.styles['ChineseCaption']))
            except Exception as e:
                LOGGER.error(f"Failed to add history image: {e}")

        # IOCs
        story.append(Paragraph("3. 威胁指标 (IOCs)", self.styles['ChineseHeading1']))
        if analysis_results.get('iocs'):
            for ioc_type, iocs in analysis_results['iocs'].items():
                story.append(Paragraph(f"{ioc_type}:", self.styles['ChineseHeading2']))
                
                # Split IOCs into chunks to avoid long pages
                chunk_size = 20
                for i in range(0, len(iocs), chunk_size):
                    chunk = iocs[i:i+chunk_size]
                    # Use a table for IOCs for better formatting
                    ioc_data = [[Paragraph(ioc, self.styles['ChineseBody'])] for ioc in chunk]
                    t = Table(ioc_data, colWidths=[6*inch])
                    t.setStyle(TableStyle([
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("未检测到显著的威胁指标。", self.styles['ChineseBody']))

        # Conclusion
        story.append(Paragraph("4. 结论与建议 (Conclusion)", self.styles['ChineseHeading1']))
        
        # Dynamic conclusion based on attribution
        top_attr = analysis_results.get('top_attribution', 'Unknown')
        conclusion_text = (
            f"基于 RGAPT 系统的深度分析，本次威胁事件与 {top_attr} 组织具有极高的关联度。"
            "该结论综合了样本的语义特征（文本描述）与结构特征（行为拓扑），置信度较高。"
            "建议安全团队重点排查与该组织相关的 TTPs（战术、技术与过程），特别是报告中高亮的攻击路径。"
            "立即对受影响主机进行隔离，阻断与相关 C2 服务器的通信，并关注注册表持久化项及特定的 DLL 注入行为。"
        )
        
        story.append(Paragraph(conclusion_text, self.styles['ChineseBody']))
        
        # Add Disclaimer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("声明：本报告由 RGAPT 自动化系统生成，仅供参考，不作为最终法律依据。", self.styles['ChineseCaption']))

        doc.build(story)
        LOGGER.info(f"Report generated successfully: {pdf_path}")
        return str(pdf_path)

    def _generate_graph_image(self, graph_data, attention_data=None):
        """
        Generates a temporary image file of the graph visualization using NetworkX and Matplotlib.
        If attention_data is provided, highlights edges and nodes based on RGAT attention weights.
        """
        if not graph_data or 'nodes' not in graph_data or 'edges' not in graph_data:
            return None
            
        try:
            G = nx.Graph()
            # Map node IDs to indices (0..N-1) to match PyG geometric data
            node_id_to_idx = {node['id']: i for i, node in enumerate(graph_data['nodes'])}
            
            for i, node in enumerate(graph_data['nodes']):
                G.add_node(i, label=node.get('label', ''), type=node.get('type', ''))
            
            for edge in graph_data['edges']:
                u = node_id_to_idx.get(edge['source'])
                v = node_id_to_idx.get(edge['target'])
                if u is not None and v is not None:
                    G.add_edge(u, v)
                
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42, k=0.3) # k controls spacing
            
            # Default styles
            node_sizes = [50] * len(G.nodes)
            node_colors = ['#00d4ff'] * len(G.nodes)
            edge_widths = [1.0] * len(G.edges)
            edge_colors = ['#cccccc'] * len(G.edges)
            
            if attention_data:
                # 1. Process Node Attention (Size)
                node_att = attention_data.get('node_attention', [])
                if node_att and len(node_att) == len(G.nodes):
                    # Normalize attention for visualization
                    min_att = min(node_att)
                    max_att = max(node_att)
                    if max_att > min_att:
                        norm_att = [(x - min_att) / (max_att - min_att) for x in node_att]
                        node_sizes = [100 + x * 500 for x in norm_att] # 100 to 600
                        # Color: Blue (low) to Red (high)
                        cm = plt.cm.coolwarm
                        node_colors = [cm(x) for x in norm_att]
                    else:
                        node_sizes = [100] * len(G.nodes)

                # 2. Process Edge Attention (Width & Color)
                edge_att = attention_data.get('edge_attention', [])
                edge_idx = attention_data.get('edge_index', [])
                
                if edge_att and edge_idx and len(edge_idx) == 2:
                    # Map attention to edges
                    # edge_idx is [2, E_att], edge_att is [E_att]
                    att_map = {}
                    us = edge_idx[0]
                    vs = edge_idx[1]
                    for k in range(len(edge_att)):
                        u, v = int(us[k]), int(vs[k])
                        w = float(edge_att[k])
                        # Use max weight for undirected edge
                        pair = tuple(sorted((u, v)))
                        att_map[pair] = max(att_map.get(pair, 0.0), w)
                    
                    # Update edge styles
                    new_widths = []
                    new_colors = []
                    
                    # Normalize edge weights
                    all_weights = list(att_map.values())
                    max_w = max(all_weights) if all_weights else 1.0
                    
                    for u, v in G.edges():
                        pair = tuple(sorted((u, v)))
                        w = att_map.get(pair, 0.0)
                        
                        # Scale width
                        width = 0.5 + (w / max_w) * 4.0 if max_w > 0 else 0.5
                        new_widths.append(width)
                        
                        # Color: Grey for low attention, Red for high
                        if w > 0.1 * max_w: # Threshold
                            alpha = min(1.0, 0.3 + (w / max_w) * 0.7)
                            new_colors.append((1.0, 0.0, 0.0, alpha)) # Red
                        else:
                            new_colors.append((0.8, 0.8, 0.8, 0.3)) # Light Grey
                            
                    edge_widths = new_widths
                    edge_colors = new_colors

            # Draw
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
            
            # Draw labels for high attention/degree nodes
            # If attention data exists, label top 5 nodes by attention
            labels = {}
            if attention_data and 'node_attention' in attention_data:
                 node_att = attention_data['node_attention']
                 top_indices = sorted(range(len(node_att)), key=lambda i: node_att[i], reverse=True)[:5]
                 for idx in top_indices:
                     if idx in G.nodes:
                         labels[idx] = G.nodes[idx]['label']
            else:
                 # Fallback to degree
                 degrees = dict(G.degree)
                 top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
                 for n in top_nodes:
                     labels[n] = G.nodes[n]['label']
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
            
            img_path = self.output_dir / "temp_graph.png"
            plt.savefig(img_path, format="png", dpi=300, bbox_inches='tight')
            plt.close()
            return str(img_path)
        except Exception as e:
            LOGGER.error(f"Failed to generate graph image: {e}")
            return None
