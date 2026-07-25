"""RunSpec: bir bench kosumunun parametreleri. bench/runner.py bunu alir,
ilgili pipeline parcasini kosar, artifacts/bench/<run_id>/ altina yazar."""
import dataclasses
import hashlib


@dataclasses.dataclass(frozen=True)
class RunSpec:
    model_name: str
    use_filters: bool = True
    # Faz 2'de anlam kazanir: 'auto' | 'prefilter' | 'postfilter_rescore' | 'exact'
    strategy: str = "auto"
    top_k: int = 200
    # 'cpu' | 'gt1030_cuda' | 'colab_t4' | ... - hangi donanim sinifinda
    # olculdugunu rapor okuyanin karistirmamasi icin zorunlu alan.
    hardware_profile: str = "cpu"
    # Faz 3'te cesitlenir (VisDrone-tuned varyantlar).
    yolo_variant: str = "yolo26x"

    @property
    def run_id(self) -> str:
        payload = "|".join(str(v) for v in dataclasses.astuple(self))
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        filt = "filt" if self.use_filters else "nofilt"
        return f"{self.model_name}_{filt}_{self.strategy}_{self.hardware_profile}_{digest}"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)
