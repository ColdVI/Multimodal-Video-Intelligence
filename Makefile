MODEL ?= xclip_hf_zeroshot
SEQUENCE ?=

.PHONY: help download-data infra-up infra-down schema frames windows detect embed load \
        ingest groundtruth eval search-report bench fiftyone test clean

help:
	@echo "make infra-up      - ClickHouse + MinIO baslatir"
	@echo "make download-data - resmi VisDrone-MOT train setini indirir/dogrular"
	@echo "make schema        - ClickHouse tablolarini olusturur"
	@echo "make ingest        - frames->windows->detect->embed->load (MODEL=...)"
	@echo "make groundtruth   - VisDrone anotasyonlarindan GT uretir"
	@echo "make eval          - model x filtre kiyaslamasini kosar"
	@echo "make search-report - exact/vector/hybrid SQL kanit raporu uretir"
	@echo "make bench         - Faz 1 benchmark harness'ini kosar, HTML/JSON rapor uretir"
	@echo "make fiftyone      - sonuclari FiftyOne'da acar"
	@echo "make test          - saf-mantik pytest (GPU/veri gerekmez)"

download-data:
	python scripts/download_visdrone.py

infra-up:
	docker compose up -d
	@echo "ClickHouse hazir olana kadar birkac saniye bekleyin, sonra: make schema"

infra-down:
	docker compose down

schema:
	docker compose exec -T clickhouse clickhouse-client --multiquery < schema.sql

# VisDrone-MOT verisi data/raw/VisDrone2019-MOT-train/ altinda
# sequences/ ve annotations/ dizinleriyle bulunmalidir. Tek-sekans smoke icin
# make frames SEQUENCE=uav0000138_00000_v kullanilabilir.

frames:
	python ingest/01_frames_to_video.py $(if $(SEQUENCE),--sequence $(SEQUENCE),)

windows:
	python ingest/02_windowing.py

detect:
	python ingest/04_detect.py

embed:
	python ingest/03_embed.py --model $(MODEL)

load:
	python ingest/05_load_clickhouse.py --model $(MODEL)

ingest: frames windows detect embed load

groundtruth:
	python eval/make_groundtruth.py

eval:
	python eval/run_eval.py

search-report:
	python scripts/run_clickhouse_search_report.py

bench:
	python -m bench.runner

fiftyone:
	python notebooks/inspect_fiftyone.py

test:
	pytest tests/ -v

clean:
	rm -f data/windows.json data/features.json data/embeddings_*.json \
	      results.json results_detail.json
	rm -rf data/groundtruth
