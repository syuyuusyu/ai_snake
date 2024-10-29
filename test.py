import subprocess
import re

def ping_domain(domain):
    try:
        # 执行 ping 命令并捕获输出
        result = subprocess.run(['ping', '-c', '1', domain], stdout=subprocess.PIPE, text=True)
        output = result.stdout

        # 使用正则表达式提取 IP 地址
        ip_match = re.search(r'PING [^\(]*\(([\d\.]+)\)', output)
        if ip_match:
            ip_address = ip_match.group(1)
        else:
            ip_address = None

        # 判断是否能 ping 通，根据 returncode 来判断
        if result.returncode == 0:
            return ip_address, True  # IP 和能 ping 通
        else:
            return ip_address, False  # IP 和不能 ping 通

    except subprocess.CalledProcessError as e:
        print(f"Error executing ping command: {e}")
        return None, False  # 出现异常，不能 ping 通

# 示例
domains = ["c50s1.portablesubmarines.com","c50s2.portablesubmarines.com","c50s3.portablesubmarines.com","c50s4.portablesubmarines.com","c50s5.portablesubmarines.com","c50s801.portablesubmarines.com"]
for domain in domains:
    ip, is_reachable = ping_domain(domain)

    if is_reachable:
        print(f"Ping {domain} succeeded. IP: {ip}")
    else:
        print(f"Ping {domain} failed. IP: {ip if ip else 'Not available'}")


arr = [1,2,3]
a = arr.index(1)
print(a)

