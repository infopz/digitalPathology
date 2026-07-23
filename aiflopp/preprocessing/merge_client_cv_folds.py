from pathlib import Path


DATASET_FOLD_DIRS = [
    Path("/home/ubuntu/giodir/digitalPathology/data/manifests/reggio_client/fl_cad_binary_diff_5cv"),
    Path("/home/ubuntu/giodir/digitalPathology/data/manifests/trento_client/fl_cad_binary_diff_5cv"),
]
OUTPUT_DATASET_DIR = Path(
    "/home/ubuntu/giodir/digitalPathology/data/manifests/mergedRT/fl_cad_binary_diff_5cv"
)

SPLIT_FILES = ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]


def _fold_names(dataset_dir: Path) -> set[str]:
    return {
        path.name
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.startswith("fold_")
    }


def _get_common_fold_names(dataset_dirs: list[Path]) -> list[str]:
    missing_dirs = [str(path) for path in dataset_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Dataset folders do not exist: {missing_dirs}")

    expected_folds = _fold_names(dataset_dirs[0])
    if not expected_folds:
        raise ValueError(f"No fold_* folders found in {dataset_dirs[0]}")

    for dataset_dir in dataset_dirs[1:]:
        fold_names = _fold_names(dataset_dir)
        if fold_names != expected_folds:
            raise ValueError(
                f"Fold mismatch between {dataset_dirs[0]} and {dataset_dir}: "
                f"{sorted(expected_folds)} != {sorted(fold_names)}"
            )

    return sorted(expected_folds)


def _merge_csv_files(input_paths: list[Path], output_path: Path) -> int:
    header = None
    rows_written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        for input_path in input_paths:
            if not input_path.is_file():
                raise FileNotFoundError(f"Missing split file: {input_path}")

            with input_path.open("r", newline="") as input_file:
                input_header = input_file.readline()
                if not input_header:
                    raise ValueError(f"Empty CSV file: {input_path}")

                if header is None:
                    header = input_header
                    output_file.write(header)
                elif input_header != header:
                    raise ValueError(
                        f"Header mismatch in {input_path}: "
                        f"expected {header.strip()}, got {input_header.strip()}"
                    )

                for line in input_file:
                    output_file.write(line)
                    rows_written += 1

    return rows_written


def main() -> None:
    fold_names = _get_common_fold_names(DATASET_FOLD_DIRS)

    for fold_name in fold_names:
        for split_file in SPLIT_FILES:
            input_paths = [dataset_dir / fold_name / split_file for dataset_dir in DATASET_FOLD_DIRS]
            output_path = OUTPUT_DATASET_DIR / fold_name / split_file
            rows_written = _merge_csv_files(input_paths, output_path)
            print(f"Wrote {output_path} ({rows_written} rows)")


if __name__ == "__main__":
    main()
