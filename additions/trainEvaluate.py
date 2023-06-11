from additon import *

SEED = 1234
images_per_batch = 8
learning_rate = 0.01
batch_size_per_image = 64
iteration = 300
num_classes = 4

if __name__ == "__main__":
    if not os.path.exists('output'):
        os.makedirs('output')

    random.seed(SEED)
    np.random.seed(seed=SEED)
    torch.manual_seed(SEED)

    dataset_name = "glassMetalPapperPlastic"
    metadata = {}
    json_file = "./data/glassMetalPapperPlastic/result.json"
    image_root = "./data/glassMetalPapperPlastic"

    register_coco_instances(dataset_name, metadata, json_file, image_root)
    metadata = MetadataCatalog.get(dataset_name).set(thing_classes=["Glass", "Metal", "Paper", "Plastic"])
    dataset_dicts = DatasetCatalog.get(dataset_name)

    val_dataset_name = "valid25"
    val_metadata = {}
    val_json_file = "./data/valid25/result.json"
    val_image_root = "./data/valid25"

    register_coco_instances(val_dataset_name, val_metadata, val_json_file, val_image_root)
    val_metadata = MetadataCatalog.get(val_dataset_name).set(thing_classes=["Glass", "Metal", "Paper", "Plastic"])
    val_dataset_dicts = DatasetCatalog.get(val_dataset_name)

    init_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    cfg, test_result = train(dataset_name,
                             images_per_batch=images_per_batch,
                             learning_rate=learning_rate,
                             iteration=iteration,
                             batch_size_per_image=batch_size_per_image,
                             num_classes=num_classes,
                             test=True,
                             test_dataset_name=val_dataset_name,
                             init_model=init_model,
                             retinaNet=False)

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model, save_dir="output")
    checkpointer.save("model_mask_rcnn_R_50_FPN_3x")

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    with open("output/mask_rcnn_R_50_FPN_3x.yaml", "w") as f:
        f.write(cfg.dump())

    test_dataset_path = "./data/dataset-resized/test/"
    random_test_sample(test_dataset_path, 10, metadata, predictor)

    os.rename("./output", "./output_mask_rcnn_R_50_FPN_3x")








