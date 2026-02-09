# utils/debug_utils.py

import torch
import numpy as np

def debug_data_shape(loader, num_batches=1):
    """
    Debug function to print the shape and type of data from a dataloader.
    Now handles per-horizon transition flags properly.
    
    Args:
        loader: DataLoader to debug
        num_batches: Number of batches to check
    """
    print(f"DataLoader debug - checking {num_batches} batch(es)")
    print(f"Total samples in dataset: {len(loader.dataset)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Number of batches: {len(loader)}")
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
            
        print(f"\nBatch {i+1}:")
        
        if isinstance(batch, (list, tuple)):
            print(f"  Batch type: {type(batch).__name__}, Length: {len(batch)}")
            
            for j, item in enumerate(batch):
                # Check if this is the labels (second to last item)
                if j == len(batch) - 2:
                    if isinstance(item, list):
                        # Multi-horizon labels
                        print(f"  Item {j} (Labels): List of {len(item)} horizon tensors")
                        for h, h_labels in enumerate(item):
                            if isinstance(h_labels, torch.Tensor):
                                print(f"    Horizon {h}: shape={h_labels.shape}, dtype={h_labels.dtype}, device={h_labels.device}")
                                # Print unique values for classification labels
                                if h_labels.dtype in [torch.long, torch.int32, torch.int64]:
                                    unique_vals = torch.unique(h_labels)
                                    if len(unique_vals) <= 10:
                                        print(f"      Unique values: {unique_vals.tolist()}")
                    elif isinstance(item, torch.Tensor):
                        # Single horizon labels
                        print(f"  Item {j} (Labels): shape={item.shape}, dtype={item.dtype}, device={item.device}")
                        if item.dtype in [torch.long, torch.int32, torch.int64]:
                            unique_vals = torch.unique(item)
                            if len(unique_vals) <= 10:
                                print(f"    Unique values: {unique_vals.tolist()}")
                
                # Check if this is the transition flags (last item)
                elif j == len(batch) - 1:
                    if isinstance(item, list):
                        # Per-horizon transition flags
                        print(f"  Item {j} (Transition Flags): List of {len(item)} horizon flag tensors")
                        for h, h_flags in enumerate(item):
                            if isinstance(h_flags, torch.Tensor):
                                true_count = h_flags.sum().item()
                                false_count = len(h_flags) - true_count
                                print(f"    Horizon {h}: shape={h_flags.shape}, dtype={h_flags.dtype}")
                                print(f"      True={true_count}, False={false_count}, Ratio={true_count/len(h_flags):.2%}")
                    elif isinstance(item, torch.Tensor):
                        # Single transition flag (backward compatibility)
                        true_count = item.sum().item()
                        false_count = len(item) - true_count
                        print(f"  Item {j} (Transition Flags): shape={item.shape}, dtype={item.dtype}")
                        print(f"    True={true_count}, False={false_count}, Ratio={true_count/len(item):.2%}")
                    else:
                        print(f"  Item {j} (Transition Flags): Unexpected type {type(item)}")
                
                # Regular modality data
                elif isinstance(item, torch.Tensor):
                    print(f"  Item {j} (Modality): shape={item.shape}, dtype={item.dtype}, device={item.device}")
                    if item.dtype in [torch.float32, torch.float64]:
                        print(f"    Range: [{item.min().item():.4f}, {item.max().item():.4f}]")
                        print(f"    Mean: {item.mean().item():.4f}, Std: {item.std().item():.4f}")
                elif isinstance(item, np.ndarray):
                    print(f"  Item {j}: numpy array, shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"  Item {j}: type={type(item).__name__}")
        else:
            print(f"  Single tensor batch: shape={batch.shape}, dtype={batch.dtype}")
    
    print("\n" + "="*50)


def debug_model_output(model, sample_input, model_name="Model"):
    """
    Debug model output shapes and structure.
    Now supports multi-horizon models.
    
    Args:
        model: PyTorch model
        sample_input: Sample input to test with
        model_name: Name of the model for logging
    """
    print(f"Debugging {model_name} output:")
    
    model.eval()
    with torch.no_grad():
        if isinstance(sample_input, dict):
            # Multi-modal input
            output = model(**sample_input)
        else:
            # Single input
            output = model(sample_input)
    
    if isinstance(output, list):
        print(f"  Output is a list with {len(output)} elements (multi-horizon)")
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                print(f"    Horizon {i}: shape {out.shape}, dtype {out.dtype}")
                # Show class probabilities for first sample
                if len(out.shape) == 2 and out.shape[0] > 0:
                    probs = torch.softmax(out[0], dim=0)
                    top_class = torch.argmax(probs).item()
                    top_prob = probs[top_class].item()
                    print(f"      First sample - Top class: {top_class}, Prob: {top_prob:.4f}")
            else:
                print(f"    Horizon {i}: type {type(out).__name__}")
    elif isinstance(output, torch.Tensor):
        print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
        # Show class probabilities for first sample if classification
        if len(output.shape) == 2 and output.shape[0] > 0:
            probs = torch.softmax(output[0], dim=0)
            top_class = torch.argmax(probs).item()
            top_prob = probs[top_class].item()
            print(f"    First sample - Top class: {top_class}, Prob: {top_prob:.4f}")
    else:
        print(f"  Output type: {type(output).__name__}")
    
    # Model information
    if hasattr(model, 'get_num_prediction_heads'):
        num_heads = model.get_num_prediction_heads()
        print(f"  Model has {num_heads} prediction heads")
        
    if hasattr(model, 'get_prediction_horizons'):
        horizons = model.get_prediction_horizons()
        print(f"  Prediction horizons: {horizons}")
    
    print()


def debug_loss_computation(loss_fn, model_output, labels, loss_name="Loss"):
    """
    Debug loss computation for multi-horizon models.
    
    Args:
        loss_fn: Loss function
        model_output: Model output (list for multi-horizon or tensor for single horizon)
        labels: Labels (list for multi-horizon or tensor for single horizon)
        loss_name: Name of the loss function for logging
    """
    print(f"Debugging {loss_name} computation:")
    
    # Import the MultiHorizonLoss class for type checking
    try:
        from utils.training_utils import MultiHorizonLoss
        is_multi_horizon_loss = isinstance(loss_fn, MultiHorizonLoss)
    except ImportError:
        is_multi_horizon_loss = False
    
    if is_multi_horizon_loss:
        print(f"  Using MultiHorizonLoss with {loss_fn.num_heads} heads")
        print(f"  Loss weights: {loss_fn.loss_weights.tolist()}")
        
        if not isinstance(model_output, list):
            print(f"  ERROR: Expected list output for multi-horizon loss, got {type(model_output)}")
            return
            
        if not isinstance(labels, list):
            print(f"  ERROR: Expected list labels for multi-horizon loss, got {type(labels)}")
            return
        
        print(f"  Model outputs: {len(model_output)} horizons")
        print(f"  Labels: {len(labels)} horizons")
        
        for i, (output, label) in enumerate(zip(model_output, labels)):
            if isinstance(output, torch.Tensor) and isinstance(label, torch.Tensor):
                print(f"    Horizon {i}: output shape {output.shape}, label shape {label.shape}")
                # Show label distribution
                unique_labels, counts = torch.unique(label, return_counts=True)
                label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
                print(f"      Label distribution: {label_dist}")
            else:
                print(f"    Horizon {i}: output type {type(output)}, label type {type(label)}")
        
        # Compute loss
        try:
            loss_result = loss_fn(model_output, labels)
            print(f"  Total loss: {loss_result['total_loss'].item():.6f}")
            for i, individual_loss in enumerate(loss_result['individual_losses']):
                print(f"    Horizon {i} loss: {individual_loss.item():.6f}")
        except Exception as e:
            print(f"  ERROR computing loss: {e}")
            
    else:
        print(f"  Using single-horizon loss")
        
        if isinstance(model_output, torch.Tensor) and isinstance(labels, torch.Tensor):
            print(f"  Output shape: {model_output.shape}, Label shape: {labels.shape}")
            # Show label distribution
            unique_labels, counts = torch.unique(labels, return_counts=True)
            label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
            print(f"  Label distribution: {label_dist}")
            
            # Compute loss
            try:
                loss = loss_fn(model_output, labels)
                print(f"  Loss: {loss.item():.6f}")
            except Exception as e:
                print(f"  ERROR computing loss: {e}")
        else:
            print(f"  Output type: {type(model_output)}, Label type: {type(labels)}")
    
    print()