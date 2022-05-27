import torch
from captum.attr import visualization

# Attribution Utilities
def aggregate_attributions(attributions):
    # Sum across Embedding Dimension
    attributions = attributions.sum(dim=2).squeeze(0)
    # Frobenius Norm
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    return attributions

def make_viz_record(attributions, tokens, pred, pred_ind, label, delta):
    return visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta)
    

# For Visualization
def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta))   