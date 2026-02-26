#!/usr/bin/env python3
"""
plot_data.py - Load 'data.npz' and plot its contents.

Usage:
        python plot_data.py [--file data.npz] [--out fig.png]

The file should contain two fields: 'xvels' and 'torques'.
If the arrays are 1D, they will be plotted vs index. If 2D, each column is plotted as a separate line.
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def plot_field(data, field_name):
    if field_name not in data:
        print(f"Field '{field_name}' not found in the file.", file=sys.stderr)
        return

    arr = data[field_name]
    if arr.ndim == 0:
        print(f"Field '{field_name}' is a scalar; nothing to plot.", file=sys.stderr)
        return

    plt.figure(figsize=(8, 4.5))
    if arr.ndim == 1:
        x = np.arange(arr.shape[0])
        plt.plot(x, arr, lw=1.5, label=field_name)
    else:
        if arr.shape[0] >= arr.shape[1]:
            for i in range(arr.shape[1]):
                plt.plot(
                    np.arange(arr.shape[0]), arr[:, i], label=f"{field_name} col {i}"
                )
        else:
            for i in range(arr.shape[0]):
                plt.plot(
                    np.arange(arr.shape[1]), arr[i, :], label=f"{field_name} row {i}"
                )

        if (arr.ndim == 2) and (max(arr.shape) > 1):
            plt.legend(frameon=False, fontsize="small")

    plt.xlabel("sample index")
    plt.ylabel(field_name)
    plt.title(field_name)
    plt.grid(alpha=0.3)
    plt.tight_layout()


def main():
    p = argparse.ArgumentParser(description="Load and plot data.npz")
    p.add_argument("--file", "-f", default="data.npz", help="NumPy .npz file to load")
    p.add_argument(
        "--out", "-o", help="Optional output image file to save (e.g. fig.png)"
    )
    args = p.parse_args()

    try:
        data = np.load(args.file, allow_pickle=False)
    except Exception as e:
        print(f"Error loading '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    plot_field(data, "xvels")
    if args.out:
        plt.savefig(f"xvels_{args.out}", dpi=200)
        print(f"Saved plot to xvels_{args.out}")
    else:
        plt.show()

    plot_field(data, "torques")
    if args.out:
        plt.savefig(f"torques_{args.out}", dpi=200)
        print(f"Saved plot to torques_{args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
