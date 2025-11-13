import os
from PyPDF2 import PdfMerger

folder = "plots/evo_pca_512_alpha_range/evo_pca_512_combined_std_exp_alpha_range"

merger = PdfMerger()

for filename in sorted(os.listdir(folder)):
    if filename.endswith("top10_ratio_metrics.pdf"):
        merger.append(os.path.join(folder, filename))

output_path = os.path.join(folder, "merged_top10_ratio_metrics.pdf")
merger.write(output_path)
merger.close()

print("合并完成：", output_path)