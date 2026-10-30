"""
extract_network_features_step2.py
Member 3 - Network-Traffic Classification (Step 2)

First feature extraction: reads every Cuckoo pcap under CUCKOO_ROOT and writes one row of
basic traffic statistics per analysis to FEATURES_STEP2 (see config.py).

Features per pcap (computed with tshark, Wireshark's command-line tool)
  packet_count, total_bytes, duration_seconds, average_packet_bytes
  tcp/udp/dns/http/tls/icmp/other_packets - packets per protocol, using the protocol
      tshark shows in its "Protocol" column (the highest layer it recognises, e.g. a DNS
      packet counts as dns, not udp)
  status - ok, empty (no packets) or error: <message>

These 11 statistics are the features of the final network model
(train_network_compatible_api.py). Original pcaps are only read, never modified.

Usage (lab machine, needs tshark):  python extract_network_features_step2.py
"""
import os
import csv
import subprocess
from collections import Counter
from config import CUCKOO_ROOT, FEATURES_STEP2

# Input folder with the Cuckoo analyses and output CSV, both set in config.py
ROOT = str(CUCKOO_ROOT)
OUTPUT = str(FEATURES_STEP2)

# Columns of the output CSV (one row per pcap)
FIELDS = [
    "analysis_id",
    "pcap_path",
    "packet_count",
    "total_bytes",
    "duration_seconds",
    "average_packet_bytes",
    "tcp_packets",
    "udp_packets",
    "dns_packets",
    "http_packets",
    "tls_packets",
    "icmp_packets",
    "other_packets",
    "status"
]

def extract_features(pcap_path):
    # Ask tshark for three fields per packet: timestamp, length and protocol column
    command = [
        "tshark", "-n", "-r", pcap_path,
        "-T", "fields",
        "-E", "separator=\t",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-e", "_ws.col.Protocol"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "TShark failed")

    # Running totals over all packets of this pcap
    packet_count = 0
    total_bytes = 0
    timestamps = []
    protocols = Counter()

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        parts += [""] * (3 - len(parts))

        timestamp, length, protocol = parts[:3]

        try:
            packet_bytes = int(length)
        except ValueError:
            continue

        packet_count += 1
        total_bytes += packet_bytes

        try:
            timestamps.append(float(timestamp))
        except ValueError:
            pass

        # Map tshark's protocol name to one of our protocol groups
        protocol = protocol.strip().upper()

        if protocol == "TCP":
            protocols["tcp"] += 1
        elif protocol == "UDP":
            protocols["udp"] += 1
        elif protocol == "DNS":
            protocols["dns"] += 1
        elif protocol in ("HTTP", "HTTP2"):
            protocols["http"] += 1
        elif protocol in ("TLS", "SSL"):
            protocols["tls"] += 1
        elif protocol in ("ICMP", "ICMPV6"):
            protocols["icmp"] += 1
        else:
            protocols["other"] += 1

    # Capture duration = time between first and last packet
    duration = (
        max(timestamps) - min(timestamps)
        if len(timestamps) >= 2 else 0.0
    )

    return {
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "duration_seconds": round(duration, 6),
        "average_packet_bytes": round(
            total_bytes / packet_count, 4
        ) if packet_count else 0,
        "tcp_packets": protocols["tcp"],
        "udp_packets": protocols["udp"],
        "dns_packets": protocols["dns"],
        "http_packets": protocols["http"],
        "tls_packets": protocols["tls"],
        "icmp_packets": protocols["icmp"],
        "other_packets": protocols["other"]
    }

# Find every .pcap file under the Cuckoo analyses folder
pcaps = []

for directory, subdirectories, filenames in os.walk(ROOT):
    for filename in filenames:
        if filename.lower().endswith(".pcap"):
            pcaps.append(os.path.join(directory, filename))

pcaps.sort()

print("PCAP files found:", len(pcaps))
print("Writing features to:", OUTPUT)

with open(OUTPUT, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
    writer.writeheader()

    for index, path in enumerate(pcaps, start=1):
        # The Cuckoo task id is the name of the folder that holds the pcap
        analysis_id = os.path.basename(os.path.dirname(path))

        row = {
            "analysis_id": analysis_id,
            "pcap_path": path,
            "packet_count": 0,
            "total_bytes": 0,
            "duration_seconds": 0,
            "average_packet_bytes": 0,
            "tcp_packets": 0,
            "udp_packets": 0,
            "dns_packets": 0,
            "http_packets": 0,
            "tls_packets": 0,
            "icmp_packets": 0,
            "other_packets": 0,
            "status": "ok"
        }

        # Default row with zeros, overwritten by the extracted features; an empty or
        # unreadable pcap keeps the zeros and gets a status instead
        try:
            row.update(extract_features(path))
            if row["packet_count"] == 0:
                row["status"] = "empty"
        except Exception as error:
            row["status"] = "error: " + str(error).replace("\n", " ")

        # Write each row immediately so a crash does not lose finished work
        writer.writerow(row)
        csvfile.flush()

        if index % 25 == 0 or index == len(pcaps):
            print("Processed", index, "of", len(pcaps))

print("Finished. Original PCAP files were not modified.")
