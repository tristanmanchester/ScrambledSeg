from scrambledSeg.inference.predict import predict_volume

predict_volume(
    checkpoint_path=r"C:\Users\tmanc\CascadeProjects\scrambledSeg\lightning_logs\version_3\checkpoints\epoch=14-step=9239.ckpt",
    input_path=r"C:\Users\tmanc\CascadeProjects\scrambledSeg\raw_data\data\cuo_8.h5",
    output_dir=r"C:\Users\tmanc\CascadeProjects\scrambledSeg\test",
    h5_dataset_path="/data",
    prediction_axis='z',
    model_variant='b4'  # Changed to b4 to match training model
)