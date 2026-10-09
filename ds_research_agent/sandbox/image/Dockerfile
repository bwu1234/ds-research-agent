# D3 sandbox image. Built and tagged by SandboxRunner.build_image(); the
# runner records the image ID and package list with each run.
# Base pinned by digest: python:3.14-slim (Debian 13.7, Python 3.14.8).
FROM python:3.14-slim@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2
COPY requirements.txt /opt/sandbox/requirements.txt
RUN apt-get update \
 && apt-get install -y --no-install-recommends strace \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --no-deps -r /opt/sandbox/requirements.txt \
 && pip check \
 && useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin agent
COPY entrypoint.sh /opt/sandbox/entrypoint.sh
RUN chmod 0755 /opt/sandbox/entrypoint.sh
ENV HOME=/scratch \
    LANG=C.UTF-8 \
    MPLCONFIGDIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /scratch
ENTRYPOINT ["/opt/sandbox/entrypoint.sh"]
