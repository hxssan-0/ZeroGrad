import matplotlib.pyplot as plt
import matplotlib.cm as cm

import _zerograd_backend as zg


def build_cnn():
    cnn = zg.Sequential()
    cnn.add(zg.Conv2d(1, 8, 3, 3, 1, 1))
    cnn.add(zg.ReLU())
    cnn.add(zg.MaxPool2d(2, 2, 2, 0))
    cnn.add(zg.Conv2d(8, 16, 3, 3, 1, 1))
    cnn.add(zg.ReLU())
    cnn.add(zg.MaxPool2d(2, 2, 2, 0))
    cnn.add(zg.Flatten())
    cnn.add(zg.Linear(784, 128))
    cnn.add(zg.ReLU())
    cnn.add(zg.Linear(128, 10))
    return cnn


def build_graph():
    train_data = zg.load_mnist(
        "../../datasets/mnist/train_images.idx3-ubyte",
        "../../datasets/mnist/train_labels.idx1-ubyte"
    )

    batch_size = 64
    loader = zg.DataLoader(train_data, batch_size)
    loader.reset()

    images, labels = loader.next_batch()
    images.shape = [images.shape[0], 1, 28, 28]

    cnn = build_cnn()
    logits = cnn.forward(images)
    loss = zg.ce_loss(logits, labels)
    return loss


def collect_intervals(root):
    analyzer = zg.GraphAnalyzer()
    result = analyzer.dry_forward(root)

    intervals = []
    for tensor, (birth, death, size_bytes) in result.items():
        intervals.append({
            "tensor": tensor,
            "op": tensor.op if tensor.op else "leaf",
            "birth": birth,
            "death": death,
            "size_bytes": size_bytes,
        })

    intervals.sort(key=lambda t: t["birth"])
    return intervals


def plot_gantt(intervals, output_path="tensor_lifetimes.png"):
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(intervals))))

    ops = sorted(set(t["op"] for t in intervals))
    palette = cm.get_cmap("tab20", max(len(ops), 1))
    op_colors = {op: palette(i) for i, op in enumerate(ops)}

    for row, t in enumerate(intervals):
        width = t["death"] - t["birth"]
        ax.barh(
            row,
            width=max(width, 0.3),
            left=t["birth"],
            height=0.6,
            color=op_colors[t["op"]],
            edgecolor="black",
            linewidth=0.5,
        )
        label = f'{t["op"]}  ({t["size_bytes"]}B)'
        ax.text(t["birth"] + max(width, 0.3) + 0.1, row, label,
                 va="center", fontsize=8)

    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels([f'#{i}' for i in range(len(intervals))], fontsize=7)
    ax.set_xlabel("step")
    ax.set_ylabel("tensor (birth order)")
    ax.set_title("Tensor Lifetime Intervals (GraphAnalyzer dry_forward)")
    ax.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=op_colors[op]) for op in ops]
    ax.legend(handles, ops, loc="upper left", bbox_to_anchor=(1.15, 1),
              fontsize=8, title="op")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"saved: {output_path}")

    total_peak_naive = sum(t["size_bytes"] for t in intervals)
    print(f"tensors analyzed: {len(intervals)}")
    print(f"sum of all tensor sizes (naive, no reuse): {total_peak_naive} bytes")


def main():
    root = build_graph()
    intervals = collect_intervals(root)
    plot_gantt(intervals)


if __name__ == "__main__":
    main()