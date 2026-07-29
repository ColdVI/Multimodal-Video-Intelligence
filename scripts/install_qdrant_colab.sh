#!/usr/bin/env bash
# Faz 6 Colab handoff - Qdrant. Politika (spec madde 8): ONCE sabitlenmis
# resmi static Linux binary; basarisizsa Docker; Qdrant Cloud ASLA
# kullanilmaz. Local Mode SADECE smoke test icindir, server/HNSW benchmark
# sonucu olarak KULLANILMAZ (bkz. notebook 04/05).
#
# Kullanim: install | start | health | stop | cleanup
set -euo pipefail

VERSION="${QDRANT_VERSION:-v1.12.4}"
INSTALL_ROOT="${QDRANT_INSTALL_ROOT:-/content/vector_bench/qdrant}"
DATA_DIR="${INSTALL_ROOT}/data"
LOG_DIR="${INSTALL_ROOT}/logs"
PID_FILE="${INSTALL_ROOT}/qdrant.pid"
DOCKER_CONTAINER_NAME="phase6_qdrant"
TAR_URL="https://github.com/qdrant/qdrant/releases/download/${VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
TAR_PATH="${INSTALL_ROOT}/qdrant-${VERSION}.tar.gz"
EXTRACT_DIR="${INSTALL_ROOT}/extracted"
HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
METHOD_FILE="${INSTALL_ROOT}/install_method.txt"

_binary_path() { echo "${EXTRACT_DIR}/qdrant"; }

cmd_install() {
    mkdir -p "${INSTALL_ROOT}" "${DATA_DIR}" "${LOG_DIR}" "${EXTRACT_DIR}"
    echo "Yontem 1/2: resmi static binary (${TAR_URL})"
    if curl -fL -o "${TAR_PATH}" "${TAR_URL}" 2>/dev/null; then
        tar -xzf "${TAR_PATH}" -C "${EXTRACT_DIR}"
        if [ -x "$(_binary_path)" ] || [ -f "$(_binary_path)" ]; then
            chmod +x "$(_binary_path)"
            echo "static_binary" > "${METHOD_FILE}"
            echo "Kuruldu (static binary): $(_binary_path)"
            return 0
        fi
    fi
    echo "Static binary basarisiz/bulunamadi. Yontem 2/2: Docker deneniyor."
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        docker pull "qdrant/qdrant:${VERSION}" || { echo "HATA: docker pull basarisiz."; echo "environment_unavailable" > "${METHOD_FILE}"; return 1; }
        echo "docker" > "${METHOD_FILE}"
        echo "Docker imaji hazir: qdrant/qdrant:${VERSION}"
        return 0
    fi
    echo "HATA: ne static binary ne Docker calisiyor - environment_unavailable."
    echo "environment_unavailable" > "${METHOD_FILE}"
    return 1
}

cmd_start() {
    METHOD="$(cat "${METHOD_FILE}" 2>/dev/null || echo "")"
    if [ "${METHOD}" = "static_binary" ]; then
        if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            echo "Qdrant zaten calisiyor (pid $(cat "${PID_FILE}"))."
            return 0
        fi
        QDRANT__STORAGE__STORAGE_PATH="${DATA_DIR}" \
        QDRANT__SERVICE__HTTP_PORT="${HTTP_PORT}" \
        QDRANT__SERVICE__GRPC_PORT="${GRPC_PORT}" \
        nohup "$(_binary_path)" > "${LOG_DIR}/server.log" 2>&1 &
        echo $! > "${PID_FILE}"
        echo "Baslatildi (static binary, pid $!)."
    elif [ "${METHOD}" = "docker" ]; then
        docker run -d --name "${DOCKER_CONTAINER_NAME}" \
            -p "${HTTP_PORT}:6333" -p "${GRPC_PORT}:6334" \
            -v "${DATA_DIR}:/qdrant/storage" \
            "qdrant/qdrant:${VERSION}"
        echo "Baslatildi (docker container ${DOCKER_CONTAINER_NAME})."
    else
        echo "HATA: install_method belirlenemedi - once 'install' calistirin."
        return 1
    fi
}

cmd_health() {
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${HTTP_PORT}/readyz" > /dev/null 2>&1; then
            echo "SAGLIKLI: http://127.0.0.1:${HTTP_PORT}/readyz yanit veriyor."
            return 0
        fi
        sleep 1
    done
    echo "SAGLIKSIZ: ${HTTP_PORT} portu 30sn icinde yanit vermedi."
    return 1
}

cmd_stop() {
    METHOD="$(cat "${METHOD_FILE}" 2>/dev/null || echo "")"
    if [ "${METHOD}" = "docker" ]; then
        docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
        echo "Docker container durduruldu/silindi."
    elif [ -f "${PID_FILE}" ]; then
        kill "$(cat "${PID_FILE}")" 2>/dev/null || true
        rm -f "${PID_FILE}"
        echo "Static binary sureci durduruldu."
    fi
}

cmd_cleanup() {
    cmd_stop || true
    rm -rf "${DATA_DIR}"
    echo "Veri dizini silindi: ${DATA_DIR}"
}

case "${1:-}" in
    install) cmd_install ;;
    start) cmd_start ;;
    health) cmd_health ;;
    stop) cmd_stop ;;
    cleanup) cmd_cleanup ;;
    *) echo "Kullanim: $0 {install|start|health|stop|cleanup}"; exit 1 ;;
esac
