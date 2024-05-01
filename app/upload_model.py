from utils import upload_model

upload_model(
    model_path="/Users/y.korobko/Desktop/Applio/juice.pth",
    index_path="/Users/y.korobko/Desktop/Applio/added_IVF750_Flat_nprobe_1_juice_v2.index",
    model_name="JuiceWRLD",
    user_id=341812517,
    pretrain_name="Titan (английский язык)",
    epochs=50,
    batch_size=6,
)
