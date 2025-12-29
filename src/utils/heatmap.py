import os
import numpy as np
import matplotlib.pyplot as plt

def save_confusion_matrix(cm, class_names, save_path):
    """cm(2D array) + 클래스 이름으로 heatmap PNG 저장"""

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 칸 안에 숫자 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=14,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix heatmap saved to: {save_path}")
