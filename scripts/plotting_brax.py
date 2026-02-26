#!/usr/bin/env python3
"""
plot_xvel.py - Plot the 1st column (x-velocity) of qvel_trajectory.npy.

Usage:
    python plot_xvel.py [--file qvel_trajectory.npy] [--out fig.png]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot 1st column (x-velocity) of qvel_trajectory.npy")
    parser.add_argument("--file", "-f", default="qvel_trajectory.npy", help="NumPy .npy file to load")
    parser.add_argument("--out", "-o", help="Optional output image file (e.g. fig.png)")
    args = parser.parse_args()

    # Load data
    try:
        qvel = np.load(args.file)
    except Exception as e:
        print(f"Error loading '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Sanity check
    if qvel.ndim != 2 or qvel.shape[1] < 1:
        print(f"Expected shape (N, >=1), but got {qvel.shape}", file=sys.stderr)
        sys.exit(1)

    # Extract first column (x-velocity)
    xvel = qvel[:, 0]
    x = np.arange(len(xvel))

    # Plot
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, xvel, lw=1.5, color="tab:blue")
    plt.xlabel("Sample Index")
    plt.ylabel("x velocity")
    plt.title("MJX xvel")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save or show
    if args.out:
        plt.savefig(args.out, dpi=200)
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
