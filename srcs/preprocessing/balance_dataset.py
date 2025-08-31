from dataset_balancer import DatasetBalancer


def main():

    manifest_path = "../../datasets/manifest_split.json"
    source_dir = "../../images"
    target_dir = "../../augmented_directory"
    seed = 42

    balancer = DatasetBalancer(
        manifest_path=manifest_path,
        source_dir=source_dir,
        target_dir=target_dir,
        seed=seed,
    )

    balancer.run()


if __name__ == "__main__":
    main()
