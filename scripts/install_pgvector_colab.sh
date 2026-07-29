#!/usr/bin/env bash
# Faz 6 Colab handoff - PostgreSQL + pgvector extension. Politika (spec
# madde 8): yerel PostgreSQL + pgvector extension kurulur. apt-get, servis
# baslatma veya CREATE EXTENSION basarisiz olursa REMOTE DATABASE'E
# GECILMEZ - environment_unavailable olarak bildirilir. Root gerekliligi
# YALNIZCA bu gecici Colab VM'i ve buradaki yerel PostgreSQL instance'i
# icindir, baska hicbir sisteme yayilmaz.
#
# Kullanim: install | start | health | stop | cleanup
set -euo pipefail

PG_VERSION="${PG_VERSION:-16}"
PGVECTOR_VERSION="${PGVECTOR_VERSION:-v0.8.0}"
DATA_DIR="/content/vector_bench/pgvector/data"
LOG_DIR="/content/vector_bench/pgvector/logs"
BUILD_DIR="/content/vector_bench/pgvector/build"
DB_NAME="${PGVECTOR_DB_NAME:-phase6_vector_bench}"
PORT="${PGVECTOR_PORT:-5432}"
STATUS_FILE="/content/vector_bench/pgvector/status.txt"

_apt() { apt-get "$@" -y -qq; }

cmd_install() {
    mkdir -p "${DATA_DIR}" "${LOG_DIR}" "${BUILD_DIR}"
    echo "apt-get guncelleniyor ve postgresql-${PG_VERSION} + build araclari kuruluyor..."
    if ! (sudo -n true 2>/dev/null || [ "$(id -u)" = "0" ]); then
        echo "HATA: root/sudo yok - Colab disi bir ortamda mi calisiyorsunuz?"
        echo "environment_unavailable" > "${STATUS_FILE}"
        return 1
    fi
    if ! _apt update; then
        echo "HATA: apt-get update basarisiz."
        echo "environment_unavailable" > "${STATUS_FILE}"
        return 1
    fi
    if ! _apt install "postgresql-${PG_VERSION}" "postgresql-server-dev-${PG_VERSION}" build-essential git; then
        echo "HATA: apt-get install basarisiz (postgresql-${PG_VERSION} paketi bu Ubuntu surumunde olmayabilir)."
        echo "environment_unavailable" > "${STATUS_FILE}"
        return 1
    fi

    echo "pgvector ${PGVECTOR_VERSION} kaynaktan derleniyor..."
    if [ ! -d "${BUILD_DIR}/pgvector" ]; then
        git clone --branch "${PGVECTOR_VERSION}" --depth 1 \
            https://github.com/pgvector/pgvector "${BUILD_DIR}/pgvector" || {
            echo "HATA: pgvector git clone basarisiz."
            echo "environment_unavailable" > "${STATUS_FILE}"
            return 1
        }
    fi
    ( cd "${BUILD_DIR}/pgvector" && make -j"$(nproc)" && make install ) || {
        echo "HATA: pgvector derleme/kurulum basarisiz."
        echo "environment_unavailable" > "${STATUS_FILE}"
        return 1
    }
    echo "ok" > "${STATUS_FILE}"
    echo "Kuruldu: postgresql-${PG_VERSION} + pgvector ${PGVECTOR_VERSION}"
}

cmd_start() {
    if [ "$(cat "${STATUS_FILE}" 2>/dev/null)" != "ok" ]; then
        echo "HATA: once 'install' basariyla tamamlanmali."
        return 1
    fi
    PG_BIN="/usr/lib/postgresql/${PG_VERSION}/bin"
    if [ ! -d "${DATA_DIR}/base" ]; then
        # initdb yerel scratch veri dizinine - Drive'a DEGIL (spec madde 5)
        sudo -u postgres "${PG_BIN}/initdb" -D "${DATA_DIR}" 2>&1 | tee -a "${LOG_DIR}/initdb.log" \
            || "${PG_BIN}/initdb" -D "${DATA_DIR}"
    fi
    sudo -u postgres "${PG_BIN}/pg_ctl" -D "${DATA_DIR}" -l "${LOG_DIR}/server.log" \
        -o "-p ${PORT} -k /tmp" start 2>&1 || \
        "${PG_BIN}/pg_ctl" -D "${DATA_DIR}" -l "${LOG_DIR}/server.log" -o "-p ${PORT}" start
    sleep 2
    sudo -u postgres psql -p "${PORT}" -c "CREATE DATABASE ${DB_NAME};" 2>&1 || \
        psql -p "${PORT}" -c "CREATE DATABASE ${DB_NAME};" 2>&1 || true
    sudo -u postgres psql -p "${PORT}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1 || \
        psql -p "${PORT}" -d "${DB_NAME}" -c "CREATE EXTENSION IF NOT EXISTS vector;" || {
        echo "HATA: CREATE EXTENSION vector basarisiz."
        echo "environment_unavailable" > "${STATUS_FILE}"
        return 1
    }
    echo "PostgreSQL+pgvector calisiyor: port ${PORT}, db=${DB_NAME}"
}

cmd_health() {
    PG_BIN="/usr/lib/postgresql/${PG_VERSION}/bin"
    if pg_isready -p "${PORT}" >/dev/null 2>&1 || "${PG_BIN}/pg_isready" -p "${PORT}" >/dev/null 2>&1; then
        echo "SAGLIKLI: pg_isready port ${PORT} basarili."
        return 0
    fi
    echo "SAGLIKSIZ: pg_isready basarisiz."
    return 1
}

cmd_stop() {
    PG_BIN="/usr/lib/postgresql/${PG_VERSION}/bin"
    sudo -u postgres "${PG_BIN}/pg_ctl" -D "${DATA_DIR}" stop 2>&1 || \
        "${PG_BIN}/pg_ctl" -D "${DATA_DIR}" stop 2>&1 || true
    echo "Durduruldu."
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
