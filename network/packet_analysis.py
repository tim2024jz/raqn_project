import pyshark

def analyze_pcap(pcap_file):
    cap = pyshark.FileCapture(pcap_file, display_filter='udp')
    delays = []
    for pkt in cap:
        if hasattr(pkt, 'udp'):
            delay = float(pkt.sniff_timestamp)
            delays.append(delay)
    return delays