import os

def generate_traffic(src, dst, rate):
    os.system(f"iperf -c {dst} -u -b {rate}M -t 10 -i 1 &")
    os.system(f"iperf -s &")
