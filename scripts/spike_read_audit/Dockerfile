# Read-audit spike image (D0). Not the D3 sandbox image.
FROM python:3.14-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends strace coreutils \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pandas pyogrio \
 && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin agent
COPY entrypoint.sh /opt/audit/entrypoint.sh
COPY cases.py /opt/cases/cases.py
RUN chmod 0755 /opt/audit/entrypoint.sh && chmod 0644 /opt/cases/cases.py
ENTRYPOINT ["/opt/audit/entrypoint.sh"]
