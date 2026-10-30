#!/usr/bin/env python3
"""
extract_network_features_advanced.py
Member 3 - Network-Traffic Classification (Step 2)

Second, richer feature extraction used for the feature-set experiments. For each analysis
listed in FEATURES_CLEAN it runs tshark on the pcap and computes:
  - size and timing: total bytes, average packet size, duration, mean/max inter-arrival time
  - protocol counts: packets containing ARP, IPv4/6, TCP, UDP, ICMP, DNS, HTTP, TLS, LLMNR,
    NBNS, SSDP, SMB, DHCP, NetBIOS (a packet can count for several protocols)
  - host and flow diversity: unique IPs, IP pairs, ports, TCP/UDP flows (5-tuples)
  - TCP flag counts: SYN, ACK, FIN, RST
Output: FEATURES_ADVANCED. Existing files are never overwritten (the script stops instead).

Usage (lab machine, needs tshark):  python extract_network_features_advanced.py
"""
import csv
import os
import subprocess
import sys

from config import FEATURES_CLEAN, FEATURES_ADVANCED

INPUT_CSV = str(FEATURES_CLEAN)
OUTPUT_CSV = str(FEATURES_ADVANCED)
TEMP_CSV = OUTPUT_CSV + ".tmp"

# Packet fields requested from tshark (one tab-separated line per packet)
FIELDS = [
    "frame.time_epoch", "frame.len", "frame.protocols",
    "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.srcport", "tcp.dstport",
    "tcp.flags.syn", "tcp.flags.ack",
    "tcp.flags.fin", "tcp.flags.reset",
    "udp.srcport", "udp.dstport"
]

# Feature name -> protocol names that count for it (matched in frame.protocols)
PROTOCOLS = {
    "arp_packets": {"arp"},
    "ipv4_packets": {"ip"},
    "ipv6_packets": {"ipv6"},
    "tcp_packets": {"tcp"},
    "udp_packets": {"udp"},
    "icmp_packets": {"icmp"},
    "icmpv6_packets": {"icmpv6"},
    "igmp_packets": {"igmp"},
    "dns_packets": {"dns"},
    "http_packets": {"http"},
    "tls_packets": {"tls", "ssl"},
    "llmnr_packets": {"llmnr"},
    "nbns_packets": {"nbns"},
    "ssdp_packets": {"ssdp"},
    "smb_packets": {"smb"},
    "dhcp_bootp_packets": {"dhcp", "bootp"},
    "netbios_datagram_packets": {"nbdgm"}
}

BASE_COLUMNS = [
    "analysis_id", "pcap_path", "status",
    "packet_count", "total_bytes", "duration_seconds",
    "average_packet_bytes", "avg_interarrival_seconds",
    "max_interarrival_seconds"
]
EXTRA_COLUMNS = [
    "unique_src_ips", "unique_dst_ips", "unique_ip_addresses",
    "unique_ip_pairs", "unique_tcp_ports", "unique_udp_ports",
    "unique_tcp_flows", "unique_udp_flows",
    "tcp_syn_packets", "tcp_ack_packets",
    "tcp_fin_packets", "tcp_rst_packets"
]
COLUMNS = BASE_COLUMNS + list(PROTOCOLS.keys()) + EXTRA_COLUMNS


def empty_row(analysis_id, path, status):
    # Row with every feature at 0, used for missing, empty or failed pcaps
    row = {c: 0 for c in COLUMNS}
    row["analysis_id"] = analysis_id
    row["pcap_path"] = path
    row["status"] = status
    return row


def field(values, index):
    return values[index].strip() if index < len(values) else ""


def extract(analysis_id, path):
    # Compute all features of one pcap (see the module docstring)
    if not os.path.isfile(path):
        return empty_row(analysis_id, path, "missing_pcap")

    command = ["tshark", "-n", "-r", path, "-T", "fields",
               "-E", "occurrence=f"]
    for name in FIELDS:
        command.extend(["-e", name])

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        message = (result.stderr or "TShark failed").strip()
        return empty_row(
            analysis_id, path, "error: " + message.replace("\n", " ")[:200]
        )

    row = empty_row(analysis_id, path, "ok")
    times, sizes = [], []
    src_ips, dst_ips, all_ips, ip_pairs = set(), set(), set(), set()
    tcp_ports, udp_ports, tcp_flows, udp_flows = set(), set(), set(), set()

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        values = line.split("\t")
        values += [""] * max(0, len(FIELDS) - len(values))

        try:
            if field(values, 0):
                times.append(float(field(values, 0)))
        except ValueError:
            pass

        try:
            if field(values, 1):
                sizes.append(int(field(values, 1)))
        except ValueError:
            pass

        row["packet_count"] += 1
        tokens = set(field(values, 2).split(":"))

        # Each protocol feature counts packets containing that protocol.
        for feature, names in PROTOCOLS.items():
            if tokens.intersection(names):
                row[feature] += 1

        # IPv4 address if present, otherwise IPv6
        src = field(values, 3) or field(values, 5)
        dst = field(values, 4) or field(values, 6)

        if src:
            src_ips.add(src)
            all_ips.add(src)
        if dst:
            dst_ips.add(dst)
            all_ips.add(dst)
        if src and dst:
            ip_pairs.add(tuple(sorted((src, dst))))

        # TCP flags and flows; a flow is the sorted pair (ip:port, ip:port), so both
        # directions of a connection count once
        tcp_sport, tcp_dport = field(values, 7), field(values, 8)

        if "tcp" in tokens:
            for index, feature in [
                (9, "tcp_syn_packets"), (10, "tcp_ack_packets"),
                (11, "tcp_fin_packets"), (12, "tcp_rst_packets")
            ]:
                if field(values, index) == "1":
                    row[feature] += 1

            if src and dst and tcp_sport and tcp_dport:
                a = (src, tcp_sport)
                b = (dst, tcp_dport)
                tcp_ports.update((tcp_sport, tcp_dport))
                tcp_flows.add(tuple(sorted((a, b))))

        udp_sport, udp_dport = field(values, 13), field(values, 14)
        if "udp" in tokens and src and dst and udp_sport and udp_dport:
            a = (src, udp_sport)
            b = (dst, udp_dport)
            udp_ports.update((udp_sport, udp_dport))
            udp_flows.add(tuple(sorted((a, b))))

    if row["packet_count"] == 0:
        row["status"] = "empty"
        return row

    # Aggregate statistics over all packets
    row["total_bytes"] = sum(sizes)
    if sizes:
        row["average_packet_bytes"] = round(sum(sizes) / len(sizes), 4)

    if len(times) >= 2:
        times.sort()
        row["duration_seconds"] = round(times[-1] - times[0], 6)
        intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
        row["avg_interarrival_seconds"] = round(
            sum(intervals) / len(intervals), 6
        )
        row["max_interarrival_seconds"] = round(max(intervals), 6)

    row["unique_src_ips"] = len(src_ips)
    row["unique_dst_ips"] = len(dst_ips)
    row["unique_ip_addresses"] = len(all_ips)
    row["unique_ip_pairs"] = len(ip_pairs)
    row["unique_tcp_ports"] = len(tcp_ports)
    row["unique_udp_ports"] = len(udp_ports)
    row["unique_tcp_flows"] = len(tcp_flows)
    row["unique_udp_flows"] = len(udp_flows)
    return row


def main():
    # Never overwrite an existing output or temporary file.
    if os.path.exists(OUTPUT_CSV) or os.path.exists(TEMP_CSV):
        print("STOP: output or temporary file already exists.")
        print("No file was overwritten. Check:", OUTPUT_CSV, TEMP_CSV)
        return 2

    if not os.path.isfile(INPUT_CSV):
        print("ERROR: input file not found:", INPUT_CSV)
        return 2

    if subprocess.call(
        ["which", "tshark"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) != 0:
        print("ERROR: tshark was not found.")
        return 2

    with open(INPUT_CSV, "r", newline="") as f:
        analyses = list(csv.DictReader(f))

    print("Analyses to process:", len(analyses))
    processed = 0
    errors = 0

    try:
        with open(TEMP_CSV, "x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()

            for item in analyses:
                row = extract(item["analysis_id"], item["pcap_path"])
                writer.writerow(row)
                processed += 1

                if row["status"] not in ("ok", "empty"):
                    errors += 1

                if processed % 25 == 0 or processed == len(analyses):
                    print("Processed {}/{}; last status: {}".format(
                        processed, len(analyses), row["status"]
                    ))

        # Publish the finished file only after extraction completes.
        os.rename(TEMP_CSV, OUTPUT_CSV)

    except Exception as exc:
        print("ERROR:", str(exc))
        print("Temporary file, if created:", TEMP_CSV)
        print("Existing project data was not overwritten.")
        return 1

    print("\nFinished!")
    print("Output:", OUTPUT_CSV)
    print("Rows:", processed)
    print("Rows with errors:", errors)
    print("Original PCAPs and previous CSV files were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
