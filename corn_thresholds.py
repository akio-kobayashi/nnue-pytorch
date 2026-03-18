import argparse
import csv
import math
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional for dry output only
    yaml = None


PACKED_SFEN_VALUE_BYTES = 40
SCORE_OFFSET_BYTES = 32
SCORE_BYTES = 2


def _load_scores_from_csvs(input_csvs: list[str], score_column: str) -> list[float]:
    scores = []
    for input_csv in input_csvs:
        csv_path = Path(input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} が見つかりません。")
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or score_column not in reader.fieldnames:
                raise ValueError(f"入力CSV {csv_path} に '{score_column}' 列が必要です。")
            for row in reader:
                try:
                    scores.append(float(row[score_column]))
                except (TypeError, ValueError):
                    continue
    if not scores:
        raise ValueError("有効な評価値が入力CSVから読み取れませんでした。")
    scores.sort()
    return scores


def _load_scores_from_bins(input_bins: list[str]) -> list[float]:
    scores = []
    for input_bin in input_bins:
        bin_path = Path(input_bin)
        if not bin_path.exists():
            raise FileNotFoundError(f"{bin_path} が見つかりません。")
        data = bin_path.read_bytes()
        remainder = len(data) % PACKED_SFEN_VALUE_BYTES
        if remainder != 0:
            raise ValueError(
                f"{bin_path} のサイズ {len(data)} bytes は PackedSfenValue({PACKED_SFEN_VALUE_BYTES} bytes) の倍数ではありません。"
            )
        for base in range(0, len(data), PACKED_SFEN_VALUE_BYTES):
            score_bytes = data[base + SCORE_OFFSET_BYTES:base + SCORE_OFFSET_BYTES + SCORE_BYTES]
            scores.append(float(int.from_bytes(score_bytes, byteorder="little", signed=True)))
    if not scores:
        raise ValueError("有効な評価値が入力BINから読み取れませんでした。")
    scores.sort()
    return scores


def _quantile(sorted_values: list[float], q: float) -> float:
    if q <= 0.0:
        return sorted_values[0]
    if q >= 1.0:
        return sorted_values[-1]

    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _dedupe_sorted(values) -> list[float]:
    result = []
    last = None
    for value in sorted(values):
        rounded = round(float(value), 6)
        if last is None or rounded != last:
            result.append(rounded)
            last = rounded
    return result


def _load_scaling_from_config(config_path: Path) -> tuple[Optional[float], Optional[float]]:
    if yaml is None:
        return None, None
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} が見つかりません。")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    model_cfg = config.get("model", {})
    score_scaling = model_cfg.get("score_scaling")
    teacher_temperature = model_cfg.get("teacher_temperature")
    return (
        float(score_scaling) if score_scaling is not None else None,
        float(teacher_temperature) if teacher_temperature is not None else None,
    )


def _resolve_logit_scale(args: argparse.Namespace) -> tuple[float, float]:
    score_scaling = args.score_scaling
    teacher_temperature = args.teacher_temperature
    if args.config and (score_scaling is None or teacher_temperature is None):
        cfg_scaling, cfg_temperature = _load_scaling_from_config(Path(args.config))
        if score_scaling is None:
            score_scaling = cfg_scaling
        if teacher_temperature is None:
            teacher_temperature = cfg_temperature
    if score_scaling is None:
        score_scaling = 361.0
    if teacher_temperature is None:
        teacher_temperature = 1.0
    if score_scaling <= 0.0 or teacher_temperature <= 0.0:
        raise ValueError("score_scaling と teacher_temperature は正である必要があります。")
    return float(score_scaling), float(teacher_temperature)


def _cp_to_logit(value_cp: float, score_scaling: float, teacher_temperature: float) -> float:
    return value_cp / (score_scaling * teacher_temperature)


def _build_cp_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds:
        return _dedupe_sorted(args.thresholds)

    input_scores = []
    if args.input_csv:
        input_scores.extend(_load_scores_from_csvs(args.input_csv, args.score_column))
    if args.input_bin:
        input_scores.extend(_load_scores_from_bins(args.input_bin))

    if input_scores:
        if args.num_thresholds <= 0:
            raise ValueError("--num-thresholds は1以上を指定してください。")
        input_scores.sort()
        quantiles = [
            _quantile(input_scores, i / (args.num_thresholds + 1))
            for i in range(1, args.num_thresholds + 1)
        ]
        thresholds = _dedupe_sorted(quantiles)
        if not thresholds:
            raise ValueError("分位点から有効な閾値を生成できませんでした。")
        return thresholds

    if args.num_thresholds <= 0:
        raise ValueError("--num-thresholds は1以上を指定してください。")
    if args.min_score >= args.max_score:
        raise ValueError("--min-score は --max-score 未満である必要があります。")

    if args.num_thresholds == 1:
        return [float(args.min_score + args.max_score) / 2.0]

    step = (args.max_score - args.min_score) / (args.num_thresholds - 1)
    return _dedupe_sorted(args.min_score + step * i for i in range(args.num_thresholds))


def build_thresholds(args: argparse.Namespace) -> tuple[list[float], list[float], float, float]:
    cp_thresholds = _build_cp_thresholds(args)
    score_scaling, teacher_temperature = _resolve_logit_scale(args)
    logit_thresholds = [
        round(_cp_to_logit(value, score_scaling, teacher_temperature), 6)
        for value in cp_thresholds
    ]
    return cp_thresholds, _dedupe_sorted(logit_thresholds), score_scaling, teacher_temperature


def print_outputs(
    cp_thresholds: list[float],
    logit_thresholds: list[float],
    weight: float,
    score_scaling: float,
    teacher_temperature: float,
) -> None:
    print("corn_aux_thresholds (logit space):")
    for value in logit_thresholds:
        print(f"  - {value:g}")
    print(f"corn_aux_weight: {weight:g}")
    print("")
    print("derived_from_cp_thresholds:")
    for value in cp_thresholds:
        print(f"  - {value:g}")
    print("")
    print(
        "logit conversion: cp / (score_scaling * teacher_temperature)"
        f" = cp / ({score_scaling:g} * {teacher_temperature:g})"
    )
    cli_values = ",".join(f"{v:g}" for v in logit_thresholds)
    print("")
    print("CLI example:")
    print(f"  --model.corn_aux_weight={weight:g} --model.corn_aux_thresholds=[{cli_values}]")


def update_config(config_path: Path, thresholds: list[float], weight: float) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML が必要です。`python -m pip install PyYAML` を実行してください。")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    model_cfg = config.setdefault("model", {})
    model_cfg["corn_aux_thresholds"] = thresholds
    model_cfg["corn_aux_weight"] = float(weight)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or write CORN auxiliary-loss thresholds for nnue-pytorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--thresholds", nargs="*", type=float, help="cp 空間で閾値を明示指定する場合のリスト。")
    parser.add_argument("--input-csv", nargs="+", help="評価済みCSVから評価値分布を読む。複数指定可。")
    parser.add_argument("--input-bin", nargs="+", help="PackedSfenValue .bin から評価値分布を読む。複数指定可。")
    parser.add_argument("--score-column", default="eval_score_cp", help="入力CSVで使う評価値列名。")
    parser.add_argument("--min-score", type=float, default=-300.0, help="linspace生成時の最小cp閾値。")
    parser.add_argument("--max-score", type=float, default=300.0, help="linspace生成時の最大cp閾値。")
    parser.add_argument("--num-thresholds", type=int, default=7, help="linspaceまたは分位点生成時の閾値数。")
    parser.add_argument("--score-scaling", type=float, help="ロジット化に使う model.score_scaling。")
    parser.add_argument("--teacher-temperature", type=float, help="ロジット化に使う model.teacher_temperature。")
    parser.add_argument("--weight", type=float, default=0.1, help="CORN補助損失の重み。")
    parser.add_argument("--config", help="指定すると config.yaml の model.corn_aux_* を更新する。")
    args = parser.parse_args()

    cp_thresholds, logit_thresholds, score_scaling, teacher_temperature = build_thresholds(args)
    print_outputs(cp_thresholds, logit_thresholds, args.weight, score_scaling, teacher_temperature)

    if args.config:
        config_path = Path(args.config)
        update_config(config_path, logit_thresholds, args.weight)
        print("")
        print(f"Updated config: {config_path}")


if __name__ == "__main__":
    main()
