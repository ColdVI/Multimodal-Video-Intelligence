#!/usr/bin/env bash
# Faz 6 Colab handoff - ClickHouse'u resmi static tgz'den, ROOT GEREKTIRMEDEN
# kullanici alaninda kurar/baslatir. Surum backend_versions.json'dan SABIT
# okunur (latest degil - spec madde 8/10). Veri dizini /content/vector_bench
# altinda (Drive'a DEGIL - spec madde 5).
#
# Kullanim:
#   bash scripts/install_clickhouse_colab.sh install   # indir+ac, baslatma
#   bash scripts/install_clickhouse_colab.sh start      # arka planda baslat
#   bash scripts/install_clickhouse_colab.sh health      # HTTP ping (8123)
#   bash scripts/install_clickhouse_colab.sh stop        # durdur
#   bash scripts/install_clickhouse_colab.sh cleanup     # veri dizinini sil
set -eu
# pipefail bazi ortamlarda (ör. bash bir POSIX-sh varyantina sembolik link
# oldugunda) "invalid option name" ile cakiyor - destekleniyorsa acan,
# desteklenmiyorsa sessizce atlayan taşınabilir kontrol.
(set -o pipefail 2>/dev/null) && set -o pipefail || true

VERSION="${CLICKHOUSE_VERSION:-24.8.4.13}"
INSTALL_ROOT="${CLICKHOUSE_INSTALL_ROOT:-/content/vector_bench/clickhouse}"
DATA_DIR="${INSTALL_ROOT}/data"
LOG_DIR="${INSTALL_ROOT}/logs"
PID_FILE="${INSTALL_ROOT}/clickhouse.pid"
TGZ_URL="https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-${VERSION}-amd64.tgz"
TGZ_PATH="${INSTALL_ROOT}/clickhouse-${VERSION}.tgz"
EXTRACT_DIR="${INSTALL_ROOT}/extracted"
HTTP_PORT="${CLICKHOUSE_HTTP_PORT:-8123}"
TCP_PORT="${CLICKHOUSE_TCP_PORT:-9000}"

_find_binary() {
    find "${EXTRACT_DIR}" -type f -name clickhouse -perm -u+x 2>/dev/null | head -1
}

cmd_install() {
    mkdir -p "${INSTALL_ROOT}" "${DATA_DIR}" "${LOG_DIR}" "${EXTRACT_DIR}"
    if [ -n "$(_find_binary)" ]; then
        echo "ClickHouse binary zaten var: $(_find_binary)"
        return 0
    fi
    echo "ClickHouse ${VERSION} indiriliyor: ${TGZ_URL}"
    curl -fL -o "${TGZ_PATH}" "${TGZ_URL}" || {
        echo "HATA: indirme basarisiz - ${TGZ_URL} erisilebilir mi kontrol edin (scripts/colab_preflight.py)"
        return 1
    }
    tar -xzf "${TGZ_PATH}" -C "${EXTRACT_DIR}"
    BIN="$(_find_binary)"
    if [ -z "${BIN}" ]; then
        echo "HATA: cikarilan arsivde 'clickhouse' calistirilabilir dosyasi bulunamadi."
        return 1
    fi
    chmod +x "${BIN}"
    echo "Kuruldu: ${BIN}"
}

cmd_start() {
    BIN="$(_find_binary)"
    if [ -z "${BIN}" ]; then
        echo "HATA: once 'install' calistirin."
        return 1
    fi
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "ClickHouse zaten calisiyor (pid $(cat "${PID_FILE}"))."
        return 0
    fi
    nohup "${BIN}" server \
        --path "${DATA_DIR}" \
        -- --http_port="${HTTP_PORT}" --tcp_port="${TCP_PORT}" \
           --listen_host=127.0.0.1 \
        > "${LOG_DIR}/server.log" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "Baslatildi (pid $!), log: ${LOG_DIR}/server.log"
}

cmd_health() {
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${HTTP_PORT}/ping" > /dev/null 2>&1; then
            echo "SAGLIKLI: http://127.0.0.1:${HTTP_PORT}/ping yanit veriyor."
            return 0
        fi
        sleep 1
    done
    echo "SAGLIKSIZ: ${HTTP_PORT} portu 30sn icinde yanit vermedi."
    return 1
}

cmd_stop() {
    if [ -f "${PID_FILE}" ]; then
        PID="$(cat "${PID_FILE}")"
        kill "${PID}" 2>/dev/null || true
        rm -f "${PID_FILE}"
        echo "Durduruldu (pid ${PID})."
    else
        echo "PID dosyasi yok - zaten durmus olabilir."
    fi
}

cmd_cleanup() {
    cmd_stop || true
    rm -rf "${DATA_DIR}"
    echo "Veri dizini silindi: ${DATA_DIR} (binary/log KORUNDU, sadece install icin tekrar 'install' cagirin gerekirse)"
}

case "${1:-}" in
    install) cmd_install ;;
    start) cmd_start ;;
    health) cmd_health ;;
    stop) cmd_stop ;;
    cleanup) cmd_cleanup ;;
    *) echo "Kullanim: $0 {install|start|health|stop|cleanup}"; exit 1 ;;
esac
