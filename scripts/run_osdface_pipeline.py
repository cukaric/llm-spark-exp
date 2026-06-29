"""Run the synthetic face pipeline: generate → restore → (upscale)."""

from __future__ import annotations

import logging
import shutil

from llm_spark_exp.paths import DATA_DIR
from llm_spark_exp.synthetic_faces.generation.vec2face_plus import (
    DEFAULT_CENTER_FEATURES_PATH,
    Vec2FacePlusPaths,
    Vec2FacePlusRunner,
    default_vec2face_plus_weights,
    download_center_feature_examples,
    download_huggingface_file,
)
from llm_spark_exp.synthetic_faces.restoration.restore import (
    FaceRestoreConfig,
    FaceRestoreRunner,
    OSDFacePaths,
    clone_osdface_repo,
    download_osdface_weights,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "osdface_restore_50id"
SOURCE_DIR = DATA_DIR / "processed" / EXPERIMENT_NAME
GENERATED_DIR = SOURCE_DIR / "01_generated"
RESTORED_DIR = SOURCE_DIR / "02_restored"

NUM_IDENTITIES = 50
EXAMPLES_PER_IDENTITY = 12


def main() -> None:
    logger.info("=== Pipeline: %s ===", EXPERIMENT_NAME)

    # Step 0: Ensure weights
    logger.info("Step 0: Checking weights...")
    v2f_paths = Vec2FacePlusPaths()
    runner = Vec2FacePlusRunner(paths=v2f_paths)
    weights = default_vec2face_plus_weights()
    for w in weights:
        target = w.local_dir / w.filename
        if not target.exists():
            logger.info("  Downloading %s...", w.filename)
            download_huggingface_file(repo_id=w.repo_id, filename=w.filename, local_dir=w.local_dir)
        else:
            logger.info("  %s: OK", w.filename)

    center_features = DEFAULT_CENTER_FEATURES_PATH
    if not center_features.exists():
        logger.info("  Downloading center_feature_examples.npy...")
        download_center_feature_examples(local_dir=v2f_paths.weights_dir)
    else:
        logger.info("  center_features: OK")

    osd_paths = OSDFacePaths()
    if not osd_paths.repo_dir.exists():
        logger.info("  Cloning OSDFace repo...")
        clone_osdface_repo()
    else:
        logger.info("  OSDFace repo: OK")

    if not osd_paths.img_encoder_weight.exists():
        logger.info("  Downloading OSDFace weights...")
        download_osdface_weights()
    else:
        logger.info("  OSDFace weights: OK")

    # Step 1: Generate identities from center features
    logger.info(
        "Step 1: Generating %d identities x %d examples...",
        NUM_IDENTITIES,
        EXAMPLES_PER_IDENTITY,
    )
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    run = runner.generate_from_center_features(
        center_feature=center_features,
        name=EXPERIMENT_NAME,
        batch_size=12,
        examples=EXAMPLES_PER_IDENTITY,
        start_end=f"0:{NUM_IDENTITIES}",
        device="cuda",
        variation_sigmas="0.08,0.12,0.18",
        variation_weights="0.5,0.35,0.15",
    )
    logger.info("  Generated to: %s", run.output_dir)

    # Copy generated images to our experiment dir
    if run.output_dir != GENERATED_DIR:
        if GENERATED_DIR.exists():
            shutil.rmtree(GENERATED_DIR)
        shutil.copytree(run.output_dir, GENERATED_DIR)
        logger.info("  Copied to: %s", GENERATED_DIR)

    identities = [d for d in GENERATED_DIR.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.jpg"))) for d in identities)
    logger.info("  %d identities, %d images", len(identities), total_images)

    # Step 2: Run OSDFace restoration
    logger.info("Step 2: Running OSDFace face restoration...")
    restore_runner = FaceRestoreRunner(paths=osd_paths)
    restore_runner.restore_dataset(
        source_dir=GENERATED_DIR,
        output_dir=RESTORED_DIR,
        config=FaceRestoreConfig(
            merge_lora=True,
            mixed_precision="fp16",
            jpeg_quality=95,
            upscale=False,
        ),
        device="cuda",
    )
    logger.info("  Restored to: %s", RESTORED_DIR)

    restored_identities = [d for d in RESTORED_DIR.iterdir() if d.is_dir()]
    restored_images = sum(len(list(d.glob("*.jpg"))) for d in restored_identities)
    logger.info("  %d identities, %d images", len(restored_identities), restored_images)

    logger.info("=== Pipeline complete ===")
    logger.info("  Generated: %s", GENERATED_DIR)
    logger.info("  Restored:  %s", RESTORED_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
