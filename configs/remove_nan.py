import safetensors
import torch

data_path1 = "/storage2/wangzitongLab/liuxuyin/gene_circuit_design_study/gene_circuit_design/data_new/Feng_2023/dSort-Seq/evo2/merged_data_dSort-Seq_results_rank_0_pca_output.safetensors"
data1 = safetensors.torch.load_file(data_path1)

if "expressions" in data1:
    mask = ~torch.isnan(data1["expressions"]).any(dim=1)
    filtered_data1 = {key: value[mask] for key, value in data1.items()}
    
    new_path1 = data_path1.replace(".safetensors", "_filtered.safetensors")
    safetensors.torch.save_file(filtered_data1, new_path1)
    print(f"Filtered data saved to: {new_path1}")

data_path2 = "/storage2/wangzitongLab/liuxuyin/gene_circuit_design_study/gene_circuit_design/data_new/Feng_2023/dSort-Seq/raw/merged_data_dSort-Seq_results_onehot_encode_PCA.safetensors"
data2 = safetensors.torch.load_file(data_path2)

if "expressions" in data2:
    mask = ~torch.isnan(data2["expressions"]).any(dim=1)
    filtered_data2 = {key: value[mask] for key, value in data2.items()}
    
    new_path2 = data_path2.replace(".safetensors", "_filtered.safetensors")
    safetensors.torch.save_file(filtered_data2, new_path2)
    print(f"Filtered data saved to: {new_path2}")
