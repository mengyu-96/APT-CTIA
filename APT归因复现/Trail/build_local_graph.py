import os
import pandas as pd
import torch
from torch_geometric.data import Data
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'Trail-main', 'src'))
from csr import CSR
from build_dataset.label_mapper.apt_label_mapper import AptLabelMapper

def build_graph():
    dataset_dir = 'd:/git/APT归因复现/Trail/dataset/'
    mapper = AptLabelMapper()
    
    print("Loading data...")
    ips_df = pd.read_csv(os.path.join(dataset_dir, 'ips.csv'), sep='\t')
    domains_df = pd.read_csv(os.path.join(dataset_dir, 'domains.csv'), sep='\t')
    urls_df = pd.read_csv(os.path.join(dataset_dir, 'urls.csv'), sep='\t')
    rel_df = pd.read_csv(os.path.join(dataset_dir, 'report_ioc_relationships.csv'))
    enriched_rel_path = os.path.join(dataset_dir, 'ioc_enriched_relationships.csv')
    enr_df = pd.read_csv(enriched_rel_path) if os.path.exists(enriched_rel_path) else pd.DataFrame()
    
    # Define node types
    IP_TYPE = 0
    URL_TYPE = 1
    DOMAIN_TYPE = 2
    ASN_TYPE = 3
    EVENT_TYPE = 4
    
    TYPE_MAP = {'ips': IP_TYPE, 'urls': URL_TYPE, 'domains': DOMAIN_TYPE, 'ASN': ASN_TYPE, 'EVENT': EVENT_TYPE}
    
    # Collect all unique nodes
    unique_reports = sorted(rel_df['report'].unique())
    unique_ips = sorted(ips_df['ioc'].unique())
    unique_domains = sorted(domains_df['ioc'].unique())
    unique_urls = sorted(urls_df['ioc'].unique())
    
    print(f"Unique nodes: {len(unique_reports)} reports, {len(unique_ips)} IPs, {len(unique_domains)} domains, {len(unique_urls)} URLs")
    
    # Create node mapping: node_name -> global_id
    node_to_id = {}
    node_types = []
    node_names = []
    
    # Add Events first (important for train_gnn logic)
    for report in unique_reports:
        node_to_id[report] = len(node_to_id)
        node_types.append(EVENT_TYPE)
        node_names.append(report)
        
    # Add IPs
    for ip in unique_ips:
        node_to_id[ip] = len(node_to_id)
        node_types.append(IP_TYPE)
        node_names.append(ip)
        
    # Add Domains
    for domain in unique_domains:
        node_to_id[domain] = len(node_to_id)
        node_types.append(DOMAIN_TYPE)
        node_names.append(domain)
        
    # Add URLs
    for url in unique_urls:
        node_to_id[url] = len(node_to_id)
        node_types.append(URL_TYPE)
        node_names.append(url)
        
    print(f"Total nodes: {len(node_to_id)}")
    
    # Build edge_index
    print("Building edges...")
    edges = set()
    
    # Event <-> IOC edges
    for _, row in rel_df.iterrows():
        u = node_to_id[row['report']]
        v = node_to_id[row['ioc']]
        edges.add((u, v))
        edges.add((v, u))
        
    # URL <-> Domain edges
    from urllib.parse import urlparse
    for url in unique_urls:
        try:
            # Simple extraction for robustness
            if '://' in url:
                domain = urlparse(url).netloc
            else:
                # If no scheme, urlparse might fail to get netloc
                domain = url.split('/')[0]
            
            if domain and domain in node_to_id:
                u = node_to_id[url]
                v = node_to_id[domain]
                edges.add((u, v))
                edges.add((v, u))
        except:
            pass

    # Locally enriched IOC <-> IOC edges (if available)
    if not enr_df.empty:
        for _, row in enr_df.iterrows():
            src = row.get('src_ioc')
            dst = row.get('dst_ioc')
            if src in node_to_id and dst in node_to_id:
                u = node_to_id[src]
                v = node_to_id[dst]
                edges.add((u, v))
                edges.add((v, u))
            
    edge_index = torch.tensor(list(edges)).t().contiguous()
    x = torch.tensor(node_types, dtype=torch.long)
    
    # Labels for Event nodes
    print("Assigning labels...")
    all_apts = sorted(rel_df['report'].apply(lambda x: mapper.get_label_from_str(x.split('_')[0].upper())).unique())
    lmap = {i: apt for i, apt in enumerate(all_apts)}
    inv_lmap = {apt: i for i, apt in lmap.items()}
    
    event_ids = []
    ys = []
    sources = []
    src_map = {'local': 0}
    
    for i, name in enumerate(node_names):
        if node_types[i] == EVENT_TYPE:
            event_ids.append(i)
            apt = mapper.get_label_from_str(name.split('_')[0].upper())
            ys.append(inv_lmap[apt])
            sources.append(0) # local
            
    # feat_map: mapping nodes to their row index in the feature DataFrames
    feat_map = torch.full((len(node_names),), -1, dtype=torch.long)
    
    # IP feat map
    ip_to_idx = {ioc: i for i, ioc in enumerate(unique_ips)}
    for i, name in enumerate(node_names):
        if node_types[i] == IP_TYPE:
            feat_map[i] = ip_to_idx[name]
            
    # Domain feat map
    dom_to_idx = {ioc: i for i, ioc in enumerate(unique_domains)}
    for i, name in enumerate(node_names):
        if node_types[i] == DOMAIN_TYPE:
            feat_map[i] = dom_to_idx[name]
            
    # URL feat map
    url_to_idx = {ioc: i for i, ioc in enumerate(unique_urls)}
    for i, name in enumerate(node_names):
        if node_types[i] == URL_TYPE:
            feat_map[i] = url_to_idx[name]
            
    print("Creating Data object...")
    data = Data(
        x=x,
        edge_index=edge_index,
        label_map=lmap,
        feat_map=feat_map,
        y=torch.tensor(ys, dtype=torch.long),
        sources=torch.tensor(sources),
        src_map=src_map,
        event_ids=torch.tensor(event_ids, dtype=torch.long),
        ntypes=TYPE_MAP,
        node_names=node_names
    )
    
    # Add CSR
    print("Building CSR...")
    data.edge_csr = CSR(data.edge_index)
    
    # Save
    out_path = os.path.join(dataset_dir, 'full_graph_csr.pt')
    torch.save(data, out_path)
    print(f"Saved graph to {out_path}")
    print(f"Nodes: {len(node_names)}, Edges: {edge_index.size(1)}")
    print(f"Events: {len(event_ids)}, Labels: {len(lmap)}")

if __name__ == '__main__':
    build_graph()
