import os
import re
import csv
import pandas as pd
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse
import json

# Add src to path so we can import project modules
sys.path.append(os.path.join(os.getcwd(), 'Trail-main', 'src'))

from feature_extraction.featurizer import Featurizer
from build_dataset.utils import IP_PATTERN, IP6_PATTERN
from build_dataset.label_mapper.apt_label_mapper import AptLabelMapper
from config import config

# Robust regex for extracting from text
URL_REGEX = re.compile(r'https?://[^\s/$.?#].[^\s]*', re.IGNORECASE)
BARE_URL_PATH_REGEX = re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]+)\b')
IP_REGEX = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
# Simple domain regex - matches strings with dots that aren't IPs or URLs
DOMAIN_REGEX = re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b')
ALNUM_ONLY = re.compile(r'[^A-Z0-9]+')
MAX_ENRICH_LINKS = 8
DEFAULT_TS = "1970-01-01T00:00:00Z"
MIN_COOCCUR_REPORTS = 1
SUPPORTED_RELATIONS = {
    'url_has_host_ip',
    'url_has_host_domain',
    'subdomain_of',
    'cooccur_dns',
    'offline_passive_dns'
}
BAD_TLDS = {
    'the', 'and', 'for', 'from', 'with', 'this', 'that', 'there', 'then',
    'than', 'into', 'onto', 'such', 'have', 'your', 'ours', 'their', 'was',
    'were', 'will', 'would', 'could', 'should'
}
TRAILING_PUNCT = '.,;:)]}>"\''
SPACED_IP_REGEX = re.compile(r'\b\d{1,3}(?:\s*\.\s*\d{1,3}){3}\b')


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == '':
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_relation_allowlist():
    raw = os.environ.get('RELATION_ALLOWLIST', '')
    if not raw.strip():
        return set(SUPPORTED_RELATIONS)
    rels = {x.strip() for x in raw.split(',') if x.strip()}
    valid = rels.intersection(SUPPORTED_RELATIONS)
    return valid if valid else set(SUPPORTED_RELATIONS)


MAX_ENRICH_LINKS = _env_int('MAX_ENRICH_LINKS', MAX_ENRICH_LINKS)
MIN_COOCCUR_REPORTS = _env_int('MIN_COOCCUR_REPORTS', MIN_COOCCUR_REPORTS)
RELATION_ALLOWLIST = _env_relation_allowlist()


def normalize_apt_label(raw_label, mapper):
    """
    Normalize local report filename prefix to official APT label when possible.
    """
    if not raw_label:
        return raw_label

    base = str(raw_label).strip().upper()
    candidates = {
        base,
        base.replace('-', ' '),
        base.replace('_', ' '),
        base.replace(' ', ''),
        ALNUM_ONLY.sub('', base),
    }

    # Common local alias seen in filenames.
    if base == 'DEEPPANDA':
        candidates.update({'DEEP PANDA', 'APT19'})

    for cand in candidates:
        mapped = mapper.get_label_from_str(cand)
        if mapped != cand:
            return mapped

    # If no alias mapping found, keep uppercase label.
    return base

def refang(text):
    # Common defanging patterns
    text = text.replace('[.]', '.').replace('(.)', '.')
    text = text.replace('[at]', '@').replace('(at)', '@')
    text = text.replace('[:]', ':')
    text = re.sub(r'hxxp', 'http', text, flags=re.IGNORECASE)
    return text

def extract_iocs_from_text(text):
    text = refang(text)

    def _extract_chunk(chunk, ips_out, domains_out, urls_out):
        urls = URL_REGEX.findall(chunk)
        urls.extend([f'http://{u}' for u in BARE_URL_PATH_REGEX.findall(chunk)])
        for u in urls:
            nu = normalize_url(u)
            if not nu:
                continue
            host = extract_url_host(nu)
            if not host:
                continue
            if re.match(IP_PATTERN, host):
                ips_out.add(host)
            elif is_valid_domain(host):
                domains_out.add(host)
            else:
                continue
            urls_out.add(nu)

        chunk_no_urls = URL_REGEX.sub(' ', chunk)
        for ip in IP_REGEX.findall(chunk_no_urls):
            if re.match(IP_PATTERN, ip):
                ips_out.add(ip)
        for match in SPACED_IP_REGEX.findall(chunk_no_urls):
            ip = re.sub(r'\s+', '', match)
            if re.match(IP_PATTERN, ip):
                ips_out.add(ip)

        for d in DOMAIN_REGEX.findall(chunk_no_urls):
            if not re.match(IP_PATTERN, d) and is_valid_domain(d):
                domains_out.add(normalize_domain(d))

    ips, domains, urls = set(), set(), set()
    _extract_chunk(text, ips, domains, urls)

    lines = text.splitlines()

    # Paragraph-level parsing catches wrapped IoCs split across lines.
    para = []
    for line in lines + ['']:
        if line.strip():
            para.append(line.strip())
            continue
        if para:
            _extract_chunk(' '.join(para), ips, domains, urls)
            para = []

    # Table-like row parsing from markdown/csv/tsv style reports.
    for line in lines:
        if '|' in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
            for cell in cells:
                _extract_chunk(cell, ips, domains, urls)
        if '\t' in line:
            cells = [c.strip() for c in line.split('\t') if c.strip()]
            for cell in cells:
                _extract_chunk(cell, ips, domains, urls)

    return sorted(ips), sorted(domains), sorted(urls)


def _safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_offline_intel(dataset_dir):
    intel_dir = os.path.join(dataset_dir, 'offline_intel')
    if not os.path.isdir(intel_dir):
        return {
            'domain_dns': defaultdict(list),
            'ip_domains': defaultdict(list),
            'ip_whois': {},
            'domain_whois': {},
            'url_intel': {}
        }

    domain_dns = defaultdict(list)
    ip_domains = defaultdict(list)
    ip_whois = {}
    domain_whois = {}
    url_intel = {}

    passive_dns_df = _safe_read_csv(os.path.join(intel_dir, 'passive_dns.csv'))
    if not passive_dns_df.empty and {'domain', 'ip'}.issubset(passive_dns_df.columns):
        for _, row in passive_dns_df.iterrows():
            domain = normalize_domain(row.get('domain'))
            ip = str(row.get('ip', '')).strip()
            if not domain or not is_valid_domain(domain) or not re.match(IP_PATTERN, ip):
                continue
            rec = {
                'address': ip,
                'first': row.get('first_seen', DEFAULT_TS),
                'last': row.get('last_seen', DEFAULT_TS),
                'record_type': row.get('record_type', 'A'),
                'asn': row.get('asn')
            }
            domain_dns[domain].append(rec)
            ip_domains[ip].append({
                'host': domain,
                'record_type': rec['record_type'],
                'first_seen': rec['first'],
                'last_seen': rec['last']
            })

    ip_whois_df = _safe_read_csv(os.path.join(intel_dir, 'whois_ip.csv'))
    if not ip_whois_df.empty and 'ip' in ip_whois_df.columns:
        for _, row in ip_whois_df.iterrows():
            ip = str(row.get('ip', '')).strip()
            if not re.match(IP_PATTERN, ip):
                continue
            ip_whois[ip] = {
                'asn': row.get('asn'),
                'country_code': row.get('country_code'),
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude')
            }

    dom_whois_df = _safe_read_csv(os.path.join(intel_dir, 'whois_domain.csv'))
    if not dom_whois_df.empty and 'domain' in dom_whois_df.columns:
        for _, row in dom_whois_df.iterrows():
            domain = normalize_domain(row.get('domain'))
            if not domain or not is_valid_domain(domain):
                continue
            domain_whois[domain] = {
                'registrar': row.get('registrar'),
                'created': row.get('created'),
                'updated': row.get('updated'),
                'expires': row.get('expires'),
                'country_code': row.get('country_code')
            }

    url_df = _safe_read_csv(os.path.join(intel_dir, 'url_intel.csv'))
    if not url_df.empty and 'url' in url_df.columns:
        for _, row in url_df.iterrows():
            url = normalize_url(row.get('url'))
            if not url:
                continue
            host = extract_url_host(url)
            if not host:
                continue
            url_intel[url] = {
                'hostname': host,
                'ip': row.get('ip'),
                'country_code': row.get('country_code'),
                'asn': row.get('asn')
            }

    return {
        'domain_dns': domain_dns,
        'ip_domains': ip_domains,
        'ip_whois': ip_whois,
        'domain_whois': domain_whois,
        'url_intel': url_intel
    }


def normalize_domain(domain):
    if not domain:
        return ''
    return str(domain).strip().strip('.').lower()


def is_valid_domain(domain):
    d = normalize_domain(domain)
    if not d or '.' not in d:
        return False
    parts = d.split('.')
    if len(parts) < 2:
        return False
    tld = parts[-1]
    if not tld.isalpha() or not (2 <= len(tld) <= 24):
        return False
    if tld in BAD_TLDS:
        return False
    # Avoid artifacts like "1.the" and malformed labels.
    if not any(ch.isalpha() for ch in d):
        return False
    for p in parts:
        if not p or len(p) > 63:
            return False
        if p.startswith('-') or p.endswith('-'):
            return False
        if not re.fullmatch(r'[a-z0-9-]+', p):
            return False
    return True


def normalize_url(url):
    if not url:
        return ''
    u = refang(str(url).strip())
    u = u.strip(TRAILING_PUNCT)
    return u


def extract_url_host(url):
    if not url:
        return ''
    try:
        parsed = urlparse(url)
        host = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    except Exception:
        host = str(url).split('/')[0]
    host = host.split('@')[-1]
    if host.startswith('[') and host.endswith(']'):
        host = host[1:-1]
    if ':' in host:
        host = host.split(':')[0]
    return normalize_domain(host)


def get_parent_domains(domain):
    parts = normalize_domain(domain).split('.')
    if len(parts) < 3:
        return []
    return ['.'.join(parts[i:]) for i in range(1, len(parts) - 1)]


def build_local_relation_maps(report_rows, offline_intel):
    domain_to_ips = defaultdict(Counter)
    ip_to_domains = defaultdict(Counter)
    rel_rows = []

    for row in report_rows:
        ips = row['ips']
        domains = row['domains']
        urls = row['urls']

        derived_domains = set(domains)
        derived_ips = set(ips)
        report_name = row['report']

        # URL -> host, and host's parent domains
        for u in urls:
            host = extract_url_host(u)
            if not host:
                continue
            if re.match(IP_PATTERN, host):
                derived_ips.add(host)
                if 'url_has_host_ip' in RELATION_ALLOWLIST:
                    rel_rows.append({
                        'report': report_name,
                        'src_ioc': u,
                        'src_type': 'URL',
                        'dst_ioc': host,
                        'dst_type': 'IP',
                        'relation': 'url_has_host_ip'
                    })
            else:
                derived_domains.add(host)
                if 'url_has_host_domain' in RELATION_ALLOWLIST:
                    rel_rows.append({
                        'report': report_name,
                        'src_ioc': u,
                        'src_type': 'URL',
                        'dst_ioc': host,
                        'dst_type': 'domain',
                        'relation': 'url_has_host_domain'
                    })
                for parent in get_parent_domains(host):
                    derived_domains.add(parent)
                    if 'subdomain_of' in RELATION_ALLOWLIST:
                        rel_rows.append({
                            'report': report_name,
                            'src_ioc': host,
                            'src_type': 'domain',
                            'dst_ioc': parent,
                            'dst_type': 'domain',
                            'relation': 'subdomain_of'
                        })

        # Keep parent domains for extracted domains as well
        for d in list(derived_domains):
            for parent in get_parent_domains(d):
                derived_domains.add(parent)
                if 'subdomain_of' in RELATION_ALLOWLIST:
                    rel_rows.append({
                        'report': report_name,
                        'src_ioc': d,
                        'src_type': 'domain',
                        'dst_ioc': parent,
                        'dst_type': 'domain',
                        'relation': 'subdomain_of'
                    })

        row['domains'] = sorted(derived_domains)
        row['ips'] = sorted(derived_ips)
        # Build evidence pairs from local context, but keep report-level
        # cartesian connectivity as primary proxy for local passive DNS.
        evidence_pairs = set()
        for line in row.get('lines', []):
            line = refang(line)
            line_ips = {ip for ip in IP_REGEX.findall(line) if re.match(IP_PATTERN, ip)}
            line_domains = {
                normalize_domain(d)
                for d in DOMAIN_REGEX.findall(line)
                if is_valid_domain(d)
            }
            for u in URL_REGEX.findall(line):
                host = extract_url_host(normalize_url(u))
                if not host:
                    continue
                if re.match(IP_PATTERN, host):
                    line_ips.add(host)
                elif is_valid_domain(host):
                    line_domains.add(host)
            evidence_pairs.update({(d, ip) for d in line_domains for ip in line_ips})

        cartesian_pairs = {(d, ip) for d in derived_domains for ip in derived_ips}
        offline_pairs = set()
        for d in derived_domains:
            for rec in offline_intel['domain_dns'].get(d, []):
                addr = str(rec.get('address', '')).strip()
                if re.match(IP_PATTERN, addr):
                    offline_pairs.add((d, addr))
        for ip in derived_ips:
            for rec in offline_intel['ip_domains'].get(ip, []):
                host = normalize_domain(rec.get('host'))
                if is_valid_domain(host):
                    offline_pairs.add((host, ip))

        row['candidate_domain_ip_pairs'] = cartesian_pairs.union(evidence_pairs).union(offline_pairs)

    # Keep only domain-ip pairs that recur across reports; this is a
    # conservative local proxy of stable passive-DNS relations.
    pair_counter = Counter()
    for row in report_rows:
        pair_counter.update(row['candidate_domain_ip_pairs'])

    for row in report_rows:
        report_name = row['report']
        for d, ip in row['candidate_domain_ip_pairs']:
            support = pair_counter[(d, ip)]
            if support < MIN_COOCCUR_REPORTS:
                continue
            domain_to_ips[d][ip] += support
            ip_to_domains[ip][d] += support
            if 'cooccur_dns' in RELATION_ALLOWLIST:
                rel_rows.append({
                    'report': report_name,
                    'src_ioc': d,
                    'src_type': 'domain',
                    'dst_ioc': ip,
                    'dst_type': 'IP',
                    'relation': 'cooccur_dns',
                    'support': support
                })
            if 'offline_passive_dns' in RELATION_ALLOWLIST and offline_intel['domain_dns'].get(d):
                rel_rows.append({
                    'report': report_name,
                    'src_ioc': d,
                    'src_type': 'domain',
                    'dst_ioc': ip,
                    'dst_type': 'IP',
                    'relation': 'offline_passive_dns',
                    'support': support
                })

    return domain_to_ips, ip_to_domains, rel_rows

def main():
    dataset_dir = config['DATASET']
    ml_data_dir = config['ML_DATA']
    mapper = AptLabelMapper()
    
    if not os.path.exists(ml_data_dir):
        os.makedirs(ml_data_dir, exist_ok=True)
        
    reports = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    print(f"Found {len(reports)} reports in {dataset_dir}")
    offline_intel = load_offline_intel(dataset_dir)
    
    report_rows = []
    for report_file in reports:
        raw_label = report_file.split('_')[0].upper()
        apt_label = normalize_apt_label(raw_label, mapper)

        with open(os.path.join(dataset_dir, report_file), 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        ips, domains, urls = extract_iocs_from_text(content)
        report_rows.append({
            'report': report_file,
            'apt': apt_label,
            'ips': sorted(set(ips)),
            'domains': sorted(set(normalize_domain(d) for d in domains if normalize_domain(d))),
            'urls': sorted(set(urls)),
            'lines': content.splitlines()
        })

    domain_to_ips, ip_to_domains, rel_rows = build_local_relation_maps(report_rows, offline_intel)

    all_samples = []
    report_ioc_data = []
    for row in report_rows:
        apt_label = row['apt']
        report_name = row['report']

        for ip in row['ips']:
            top_domains = [d for d, _ in ip_to_domains[ip].most_common(MAX_ENRICH_LINKS)]
            top_domains.extend([
                rec.get('host') for rec in offline_intel['ip_domains'].get(ip, [])
                if rec.get('host')
            ])
            # dedupe while preserving order
            top_domains = list(dict.fromkeys([normalize_domain(d) for d in top_domains if is_valid_domain(d)]))
            top_domains = top_domains[:MAX_ENRICH_LINKS]
            resolves_to = [{
                'host': d,
                'record_type': 'A',
                'first_seen': DEFAULT_TS,
                'last_seen': DEFAULT_TS
            } for d in top_domains]
            sample = {
                'ioc': ip,
                'type': 'IP',
                'apt': apt_label,
                'source_report': report_name,
                'resolves_to': resolves_to
            }
            sample.update({k: v for k, v in offline_intel['ip_whois'].get(ip, {}).items() if pd.notna(v)})
            all_samples.append(sample)
            report_ioc_data.append({'report': report_name, 'ioc': ip, 'type': 'IP'})

        for d in row['domains']:
            top_ips = [ip for ip, _ in domain_to_ips[d].most_common(MAX_ENRICH_LINKS)]
            dns_records = [{
                'address': ip,
                'first': DEFAULT_TS,
                'last': DEFAULT_TS,
                'record_type': 'A',
                'asn': None
            } for ip in top_ips]
            dns_records.extend(offline_intel['domain_dns'].get(d, []))
            uniq_dns = []
            seen_dns = set()
            for rec in dns_records:
                key = (
                    rec.get('address'),
                    rec.get('record_type'),
                    rec.get('first'),
                    rec.get('last')
                )
                if key in seen_dns:
                    continue
                seen_dns.add(key)
                uniq_dns.append(rec)

            sample = {
                'ioc': d,
                'type': 'domain',
                'apt': apt_label,
                'source_report': report_name,
                'dns_records': uniq_dns[:max(MAX_ENRICH_LINKS, len(offline_intel['domain_dns'].get(d, [])))]
            }
            sample.update({k: v for k, v in offline_intel['domain_whois'].get(d, {}).items() if pd.notna(v)})
            all_samples.append(sample)
            report_ioc_data.append({'report': report_name, 'ioc': d, 'type': 'domain'})

        for u in row['urls']:
            sample = {
                'ioc': u,
                'type': 'URL',
                'apt': apt_label,
                'source_report': report_name,
                'hostname': extract_url_host(u)
            }
            sample.update({k: v for k, v in offline_intel['url_intel'].get(u, {}).items() if pd.notna(v)})
            all_samples.append(sample)
            report_ioc_data.append({'report': report_name, 'ioc': u, 'type': 'URL'})
            
    print(f"Extracted {len(all_samples)} total IOC samples.")
    
    parser = Featurizer()

    dedup_stats = {}

    def process_and_save(samples, type_name, filename):
        if not samples:
            print(f"No samples for {type_name}")
            dedup_stats[type_name] = {'before': 0, 'after': 0}
            return

        unique_samples = []
        seen = set()
        for s in samples:
            key = (s.get('ioc'), s.get('apt'))
            if key in seen:
                continue
            seen.add(key)
            unique_samples.append(s)
        dedup_stats[type_name] = {'before': len(samples), 'after': len(unique_samples)}
        samples = unique_samples

        if type_name == 'IP':
            df = parser.get_ip_features(samples)
        elif type_name == 'domain':
            df = parser.get_domain_features(samples)
        elif type_name == 'URL':
            df = parser.get_url_features(samples)

        df['apt'] = [s['apt'] for s in samples]
        df['ioc'] = [s['ioc'] for s in samples]
        df['source_report'] = [s.get('source_report') for s in samples]
        
        out_path = os.path.join(dataset_dir, filename)
        df.to_csv(out_path, sep='\t', index=False, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        print(f"Saved {len(df)} {type_name} features to {out_path}")

    ips_samples = [s for s in all_samples if s['type'] == 'IP']
    domains_samples = [s for s in all_samples if s['type'] == 'domain']
    urls_samples = [s for s in all_samples if s['type'] == 'URL']
    
    process_and_save(ips_samples, 'IP', 'ips.csv')
    process_and_save(domains_samples, 'domain', 'domains.csv')
    process_and_save(urls_samples, 'URL', 'urls.csv')

    rel_df = pd.DataFrame(report_ioc_data)
    rel_df = rel_df.drop_duplicates()
    rel_path = os.path.join(dataset_dir, 'report_ioc_relationships.csv')
    rel_df.to_csv(rel_path, index=False, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
    print(f"Saved report-IOC relationships to {rel_path}")

    enr_df = pd.DataFrame(rel_rows).drop_duplicates()
    if not enr_df.empty and 'relation' in enr_df.columns:
        enr_df = enr_df[enr_df['relation'].isin(RELATION_ALLOWLIST)].copy()
    enr_path = os.path.join(dataset_dir, 'ioc_enriched_relationships.csv')
    enr_df.to_csv(enr_path, index=False, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
    print(f"Saved IOC enriched relationships to {enr_path} ({len(enr_df)} rows)")

    audit = {
        'max_enrich_links': MAX_ENRICH_LINKS,
        'min_cooccur_reports': MIN_COOCCUR_REPORTS,
        'relation_allowlist': sorted(list(RELATION_ALLOWLIST)),
        'dedup_stats': dedup_stats,
        'offline_intel_counts': {
            'domain_dns': int(sum(len(v) for v in offline_intel['domain_dns'].values())),
            'ip_domains': int(sum(len(v) for v in offline_intel['ip_domains'].values())),
            'ip_whois': int(len(offline_intel['ip_whois'])),
            'domain_whois': int(len(offline_intel['domain_whois'])),
            'url_intel': int(len(offline_intel['url_intel']))
        }
    }
    audit_path = os.path.join(dataset_dir, 'preprocess_audit.json')
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Saved preprocessing audit to {audit_path}")

if __name__ == "__main__":
    main()
