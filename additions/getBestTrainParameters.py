from additon import *

SEED = 1234
images_per_batch = [2, 4, 8, 16, 32, 64]
learning_rate = [0.01, 0.0025, 0.001, 0.00025, 0.0001, 0.000025, 0.00001]
batch_size_per_image = [128, 256, 512]
iteration = 1500
num_classes = 4

if __name__ == "__main__":
    if not os.path.exists('result'):
        os.makedirs('result')

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

    result = []
    for ipb in images_per_batch:
        for lr in learning_rate:
            for bspi in batch_size_per_image:
                cfg, test_result = train(dataset_name,
                                         images_per_batch=ipb,
                                         learning_rate=lr,
                                         iteration=iteration,
                                         batch_size_per_image=bspi,
                                         num_classes=num_classes,
                                         test=True,
                                         test_dataset_name=val_dataset_name,
                                         init_model=init_model)

                result.append([ipb, lr, bspi, cfg, test_result])
                save_result(result=result, model_name="mask_rcnn_R_50_FPN_3x", segmentation=True)

    df = pd.DataFrame()
    for i in range(len(result)):
        temp = pd.DataFrame(result[i][4], columns=result[i][4].keys())
        temp = temp.T

        images_per_batch = pd.Series([result[i][0], result[i][0]], index=[0, 1])
        learning_rate = pd.Series([result[i][1], result[i][1]], index=[0, 1])
        batch_size_per_image = pd.Series([result[i][2], result[i][2]], index=[0, 1])

        temp['images_per_batch'] = images_per_batch.values
        temp['learning_rate'] = learning_rate.values
        temp['batch_size_per_image'] = batch_size_per_image.values

        df = pd.concat([df, temp])

    df.to_csv('result/mask_rcnn_R_50_FPN_3x.csv', index=True, index_label="Result-type")

    df2 = pd.read_csv('result/mask_rcnn_R_50_FPN_3x.csv')
    dfbbox, dfsegm = get_best_result(df2, 5)

    print(dfbbox)
    print(dfsegm)
