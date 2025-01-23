import torch
from face_reaging.model.models import UNet

def export_to_onnx(pth_model_path, onnx_model_path):
    # Initialize the model
    model = UNet()
    model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
    model.eval()

    # Create dummy input based on the input size of the model
    dummy_input = torch.randn(1, 5, 1024, 1024)  # Adjust dimensions as per your model's input
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=["input"], 
        output_names=["output"]
    )
    print(f"Model exported to {onnx_model_path}")

# Export model
export_to_onnx("E:\\WALEE_INTERNSHIP\\Age_Transformation\\Face_Reagging\\best_unet_model.pth", "onnx_model.onnx")
