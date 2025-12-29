import matplotlib.pyplot as plt 
import os

def plot_curves(epochs, train_losses, val_losses, val_accs, val_f1s, save_dir: str):
    """손실/정확도 곡선 PNG로 저장"""
    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Val accuracy curve
    plt.figure()
    plt.plot(epochs, val_accs, label="Val Acc", color="blue")
    plt.plot(epochs, val_f1s, label="Val F1", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy / F1 Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png"))
    plt.close()