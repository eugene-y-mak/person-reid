import torchreid
#torchreid.models.show_avai_models()
def main():
    MODEL_NAME = "osnet_x0_5"
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="grid",
        targets="grid",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"]
    )
    print(datamanager.num_train_pids)

    model = torchreid.models.build_model(
        name=MODEL_NAME,
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    torchreid.utils.load_pretrained_weights(model, "pretrained/osnet_x0_5_market.pth")

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir=f"log/{MODEL_NAME}",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=True,
        visrank=True
    )
if __name__ == "__main__":
   main()