import mujoco
import numpy as np
import time
from mujoco.viewer import launch_passive
from collections import deque
import matplotlib
import os
import json
from datetime import datetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def export_sensor_history(
    torque_log,
    gyro_log,
    accel_log,
    derived_metrics=None,
    sample_start=0,
    sample_end=None,
    output_dir="data",
):
    """
    Persist raw and derived sensor histories for offline analysis.
    """

    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {key: _to_serialisable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serialisable(value) for value in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    payload = {
        "metadata": {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "sample_start": sample_start,
            "sample_end": sample_end,
        },
        "raw": {
            "torque": _to_serialisable(torque_log),
            "gyro": _to_serialisable(gyro_log),
            "accel": _to_serialisable(accel_log),
        },
    }

    if derived_metrics:
        payload["derived"] = _to_serialisable(derived_metrics)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"sensor_history_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)

    print(f"Sensor history exported to {filepath}")


def plot_sensor_history(torque_log, gyro_log, accel_log):
    """Render history and change plots for torque, gyro, and acceleration sensors."""
    if not torque_log:
        return

    markers = ['o', 's', '^', 'd', 'v', 'P']
    color_cycle = plt.get_cmap("tab10", max(len(torque_log), len(markers)))

    torque_avg, torque_delta = {}, {}
    gyro_mag, gyro_delta = {}, {}
    accel_mag, accel_delta = {}, {}
    gyro_axes = {axis: {} for axis in ("x", "y", "z")}
    accel_axes = {axis: {} for axis in ("x", "y", "z")}
    gyro_axes_delta = {axis: {} for axis in ("x", "y", "z")}
    accel_axes_delta = {axis: {} for axis in ("x", "y", "z")}
    axis_labels = ("x", "y", "z")

    sample_start = 0
    sample_end = 2500

    def _slice(arr):
        return arr[sample_start:sample_end] if sample_end is not None else arr[sample_start:]

    def high_pass(values, alpha=0.05):
        if len(values) <= 1:
            return np.zeros_like(values)
        smooth = np.empty_like(values)
        smooth[0] = values[0]
        for i in range(1, len(values)):
            smooth[i] = alpha * values[i] + (1 - alpha) * smooth[i - 1]
        return np.clip(np.abs(values - smooth), 1e-9, None)

    for name, samples in torque_log.items():
        if not samples:
            continue
        avg_all = 0.5 * (np.array([left for left, _ in samples]) + np.array([right for _, right in samples]))
        avg_vals = np.clip(_slice(avg_all), 1e-9, None)
        torque_avg[name] = avg_vals
        torque_delta[name] = high_pass(avg_vals)

    torque_axes_delta = {axis: {} for axis in ("avg",)}

    for name, samples in gyro_log.items():
        if not samples:
            continue
        comp = np.array(samples, dtype=float)
        comp_slice = _slice(comp)
        mag_vals = np.clip(np.linalg.norm(comp_slice, axis=1), 1e-9, None)
        gyro_mag[name] = mag_vals
        gyro_delta[name] = high_pass(mag_vals)
        if comp_slice.size:
            for idx, axis in enumerate(axis_labels):
                gyro_axes[axis][name] = comp_slice[:, idx]
                gyro_axes_delta[axis][name] = high_pass(comp_slice[:, idx])

    for name, samples in accel_log.items():
        if not samples:
            continue
        comp = np.array(samples, dtype=float)
        comp_slice = _slice(comp)
        mag_vals = np.clip(np.linalg.norm(comp_slice, axis=1), 1e-9, None)
        accel_mag[name] = mag_vals
        accel_delta[name] = high_pass(mag_vals)
        if comp_slice.size:
            for idx, axis in enumerate(axis_labels):
                accel_axes[axis][name] = comp_slice[:, idx]
                accel_axes_delta[axis][name] = high_pass(comp_slice[:, idx])

    def plot_dual_axis_pair(filename, title, left_label, left_data, right_label, right_data, left_linestyle='-', right_linestyle='--'):
        fig, ax_left = plt.subplots(figsize=(12, 7))
        ax_right = ax_left.twinx()
        left_handles, right_handles = [], []
        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]
            left_series = left_data.get(name)
            if left_series is not None and len(left_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(left_series))
                line_left, = ax_left.plot(x_vals, left_series, label=f"{name} {left_label}", color=color, linestyle=left_linestyle, linewidth=1.5)
                left_handles.append(line_left)
            right_series = right_data.get(name)
            if right_series is not None and len(right_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(right_series))
                line_right, = ax_right.plot(x_vals, right_series, label=f"{name} {right_label}", color=color, linestyle=right_linestyle, marker=marker, markevery=[len(right_series) - 1], linewidth=1.5)
                right_handles.append(line_right)
        ax_left.set_title(title)
        ax_left.set_xlabel("Sample")
        ax_left.set_ylabel(f"{left_label} (log scale)")
        ax_right.set_ylabel(f"{right_label} (log scale)")
        ax_left.set_yscale("log")
        ax_right.set_yscale("log")
        ax_left.grid(True, linestyle="--", alpha=0.3)
        handles = left_handles + right_handles
        if handles:
            ax_left.legend(handles, [h.get_label() for h in handles], loc="lower left")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_stacked_pair(filename, top_title, top_ylabel, top_data, bottom_title, bottom_ylabel, bottom_data, top_linestyle='-', bottom_linestyle='--'):
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))
            top_series = top_data.get(name)
            if top_series is not None and len(top_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(top_series))
                ax_top.plot(x_vals, top_series, label=f"{name}", color=color, linestyle=top_linestyle, linewidth=1.6)
            bottom_series = bottom_data.get(name)
            if bottom_series is not None and len(bottom_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(bottom_series))
                ax_bottom.plot(x_vals, bottom_series, label=f"{name}", color=color, linestyle=bottom_linestyle, linewidth=1.4)
        ax_top.set_title(top_title)
        ax_top.set_ylabel(top_ylabel)
        ax_top.set_yscale("log")
        ax_top.grid(True, linestyle="--", alpha=0.3)
        ax_top.legend(loc="lower left")
        ax_bottom.set_title(bottom_title)
        ax_bottom.set_xlabel("Sample")
        ax_bottom.set_ylabel(bottom_ylabel)
        ax_bottom.set_yscale("log")
        ax_bottom.grid(True, linestyle="--", alpha=0.3)
        ax_bottom.legend(loc="lower left")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_stacked_triple(filename, titles, ylabels, data_series, linestyles):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        for ax, title, ylabel, data_dict, linestyle in zip(axes, titles, ylabels, data_series, linestyles):
            for idx, name in enumerate(sorted(torque_log.keys())):
                color = color_cycle(idx % len(markers))
                series = data_dict.get(name)
                if series is not None and len(series) > 0:
                    x_vals = np.arange(sample_start, sample_start + len(series))
                    ax.plot(x_vals, series, label=f"{name}", color=color, linestyle=linestyle, linewidth=1.5)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="lower left")
        axes[-1].set_xlabel("Sample")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_single(filename, title, ylabel, data_dict, linestyle='-'):
        fig, ax = plt.subplots(figsize=(12, 7))
        for idx, name in enumerate(sorted(data_dict.keys())):
            series = data_dict[name]
            if series is None or len(series) == 0:
                continue
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]
            x_vals = np.arange(sample_start, sample_start + len(series))
            ax.plot(x_vals, series, label=name, color=color, linestyle=linestyle, marker=marker, markevery=[len(series) - 1], linewidth=1.75)
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="lower left")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_single(filename, title, ylabel, data_dict, linestyle='-'):
        fig, ax = plt.subplots(figsize=(12, 7))
        for idx, name in enumerate(sorted(data_dict.keys())):
            series = data_dict[name]
            if series is None or len(series) == 0:
                continue
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]
            x_vals = np.arange(sample_start, sample_start + len(series))
            ax.plot(
                x_vals,
                series,
                label=name,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=[len(series) - 1],
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        if data_dict:
            ax.legend(loc="best")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_change_single(filename, title, ylabel, data_dict, linestyle='-'):
        fig, ax = plt.subplots(figsize=(12, 7))
        for idx, name in enumerate(sorted(data_dict.keys())):
            series = data_dict[name]
            if series is None or len(series) == 0:
                continue
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]
            x_vals = np.arange(sample_start, sample_start + len(series))
            ax.plot(
                x_vals,
                series,
                label=name,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=[len(series) - 1],
                linewidth=1.6,
            )
        ax.set_title(title)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.3)
        if data_dict:
            ax.legend(loc="best")
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_with_torque(
        filename,
        axis_name,
        sensor_label,
        sensor_data,
        torque_data,
        sensor_linestyle='-',
        torque_label="Torque",
    ):
        fig, ax_sensor = plt.subplots(figsize=(12, 7))
        ax_torque = ax_sensor.twinx()

        sensor_handles = []
        torque_handles = []
        axis_suffix = axis_name.upper()

        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]

            sensor_series = sensor_data.get(name)
            if sensor_series is not None and len(sensor_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(sensor_series))
                line_sensor, = ax_sensor.plot(
                    x_vals,
                    sensor_series,
                    label=f"{name} {sensor_label} {axis_suffix}",
                    color=color,
                    linestyle=sensor_linestyle,
                    linewidth=1.6,
                )
                sensor_handles.append(line_sensor)

            torque_series = torque_data.get(name)
            if torque_series is not None and len(torque_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(torque_series))
                line_torque, = ax_torque.plot(
                    x_vals,
                    torque_series,
                    label=f"{name} {torque_label}",
                    color=color,
                    linestyle=":",
                    linewidth=1.6,
                )
                torque_handles.append(line_torque)

        ax_sensor.set_title(f"{axis_suffix}-Axis {sensor_label} with {torque_label} Overlay")
        ax_sensor.set_xlabel("Sample")
        ax_sensor.set_ylabel(f"{sensor_label} {axis_suffix}-Axis (log scale)")
        ax_sensor.set_yscale("log")
        ax_sensor.grid(True, linestyle="--", alpha=0.3)

        ax_torque.set_ylabel(f"{torque_label} (log scale)")
        ax_torque.set_yscale("log")

        legend_handles = sensor_handles + torque_handles
        if legend_handles:
            ax_sensor.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_with_torque_change(
        filename,
        axis_name,
        sensor_label,
        sensor_change,
        torque_change,
        sensor_linestyle='-',
        torque_label="Torque Change",
    ):
        fig, ax_sensor = plt.subplots(figsize=(12, 7))
        ax_torque = ax_sensor.twinx()

        sensor_handles = []
        torque_handles = []
        axis_suffix = axis_name.upper()

        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))

            sensor_series = sensor_change.get(name)
            if sensor_series is not None and len(sensor_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(sensor_series))
                line_sensor, = ax_sensor.plot(
                    x_vals,
                    sensor_series,
                    label=f"{name} {sensor_label} Δ{axis_suffix}",
                    color=color,
                    linestyle=sensor_linestyle,
                    linewidth=1.6,
                )
                sensor_handles.append(line_sensor)

            torque_series = torque_change.get(name)
            if torque_series is not None and len(torque_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(torque_series))
                line_torque, = ax_torque.plot(
                    x_vals,
                    torque_series,
                    label=f"{name} {torque_label}",
                    color=color,
                    linestyle=":",
                    linewidth=1.6,
                )
                torque_handles.append(line_torque)

        ax_sensor.set_title(f"{axis_suffix}-Axis {sensor_label} Change with {torque_label} Overlay")
        ax_sensor.set_xlabel("Sample")
        ax_sensor.set_ylabel(f"{sensor_label} Δ{axis_suffix} (log scale)")
        ax_sensor.set_yscale("log")
        ax_sensor.grid(True, linestyle="--", alpha=0.3)

        ax_torque.set_ylabel(f"{torque_label} (log scale)")
        ax_torque.set_yscale("log")

        legend_handles = sensor_handles + torque_handles
        if legend_handles:
            ax_sensor.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_combined(filename, axis_name, gyro_data, accel_data, torque_data):
        fig, ax_sensor = plt.subplots(figsize=(12, 7))
        ax_torque = ax_sensor.twinx()

        sensor_handles = []
        torque_handles = []
        axis_suffix = axis_name.upper()

        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))
            marker = markers[idx % len(markers)]

            gyro_series = gyro_data.get(name)
            if gyro_series is not None and len(gyro_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(gyro_series))
                line_gyro, = ax_sensor.plot(
                    x_vals,
                    gyro_series,
                    label=f"{name} Gyro {axis_suffix}",
                    color=color,
                    linestyle="-",
                    linewidth=1.6,
                )
                sensor_handles.append(line_gyro)

            accel_series = accel_data.get(name)
            if accel_series is not None and len(accel_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(accel_series))
                line_accel, = ax_sensor.plot(
                    x_vals,
                    accel_series,
                    label=f"{name} Accel {axis_suffix}",
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )
                sensor_handles.append(line_accel)

            torque_series = torque_data.get(name)
            if torque_series is not None and len(torque_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(torque_series))
                line_torque, = ax_torque.plot(
                    x_vals,
                    torque_series,
                    label=f"{name} Torque",
                    color=color,
                    linestyle=":",
                    linewidth=1.6,
                )
                torque_handles.append(line_torque)

        ax_sensor.set_title(f"{axis_suffix}-Axis Gyro & Accel with Torque Overlay")
        ax_sensor.set_xlabel("Sample")
        ax_sensor.set_ylabel(f"{axis_suffix}-Axis Sensor Value (log scale)")
        ax_sensor.set_yscale("log")
        ax_sensor.grid(True, linestyle="--", alpha=0.3)

        ax_torque.set_ylabel("Torque (Avg, log scale)")
        ax_torque.set_yscale("log")

        legend_handles = sensor_handles + torque_handles
        if legend_handles:
            ax_sensor.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_axis_combined_change(filename, axis_name, gyro_change, accel_change, torque_change):
        fig, ax_sensor = plt.subplots(figsize=(12, 7))
        ax_torque = ax_sensor.twinx()

        sensor_handles = []
        torque_handles = []
        axis_suffix = axis_name.upper()

        for idx, name in enumerate(sorted(torque_log.keys())):
            color = color_cycle(idx % len(markers))

            gyro_series = gyro_change.get(name)
            if gyro_series is not None and len(gyro_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(gyro_series))
                line_gyro, = ax_sensor.plot(
                    x_vals,
                    gyro_series,
                    label=f"{name} Gyro Δ{axis_suffix}",
                    color=color,
                    linestyle="-",
                    linewidth=1.6,
                )
                sensor_handles.append(line_gyro)

            accel_series = accel_change.get(name)
            if accel_series is not None and len(accel_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(accel_series))
                line_accel, = ax_sensor.plot(
                    x_vals,
                    accel_series,
                    label=f"{name} Accel Δ{axis_suffix}",
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )
                sensor_handles.append(line_accel)

            torque_series = torque_change.get(name)
            if torque_series is not None and len(torque_series) > 0:
                x_vals = np.arange(sample_start, sample_start + len(torque_series))
                line_torque, = ax_torque.plot(
                    x_vals,
                    torque_series,
                    label=f"{name} Torque Δ",
                    color=color,
                    linestyle=":",
                    linewidth=1.6,
                )
                torque_handles.append(line_torque)

        ax_sensor.set_title(f"{axis_suffix}-Axis Change with Torque Overlay")
        ax_sensor.set_xlabel("Sample")
        ax_sensor.set_ylabel(f"{axis_suffix}-Axis Change (High-Pass, log scale)")
        ax_sensor.grid(True, linestyle="--", alpha=0.3)
        ax_sensor.set_yscale("log")

        ax_torque.set_ylabel("Torque Change (High-Pass, log scale)")
        ax_torque.set_yscale("log")

        legend_handles = sensor_handles + torque_handles
        if legend_handles:
            ax_sensor.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    # Pairwise dual-axis history plots
    plot_dual_axis_pair("graphics/torque_gyro_curve", "Torque vs Gyro History", "Gyro Magnitude", gyro_mag, "Torque", torque_avg)
    plot_dual_axis_pair("graphics/torque_accel_curve", "Torque vs Accel History", "Accel Magnitude", accel_mag, "Torque", torque_avg, left_linestyle=':', right_linestyle='--')
    plot_dual_axis_pair("graphics/gyro_accel_curve", "Gyro vs Accel History", "Gyro Magnitude", gyro_mag, "Accel Magnitude", accel_mag, left_linestyle='-', right_linestyle=':')

    # Pairwise dual-axis change plots
    plot_dual_axis_pair("graphics/torque_gyro_change", "Torque vs Gyro Change", "Gyro Change", gyro_delta, "Torque Change", torque_delta)
    plot_dual_axis_pair("graphics/torque_accel_change", "Torque vs Accel Change", "Accel Change", accel_delta, "Torque Change", torque_delta, left_linestyle=':', right_linestyle='--')
    plot_dual_axis_pair("graphics/gyro_accel_change", "Gyro vs Accel Change", "Gyro Change", gyro_delta, "Accel Change", accel_delta, left_linestyle='-', right_linestyle=':')

    # Pairwise stacked history plots
    plot_stacked_pair("graphics/torque_gyro_stacked_curve", "Gyro Magnitude (Log Scale)", "Gyro Magnitude", gyro_mag, "Torque (Log Scale)", "Torque", torque_avg)
    plot_stacked_pair("graphics/torque_accel_stacked_curve", "Acceleration Magnitude (Log Scale)", "Accel Magnitude", accel_mag, "Torque (Log Scale)", "Torque", torque_avg, top_linestyle=':', bottom_linestyle='--')
    plot_stacked_pair("graphics/gyro_accel_stacked_curve", "Gyro Magnitude (Log Scale)", "Gyro Magnitude", gyro_mag, "Acceleration Magnitude (Log Scale)", "Accel Magnitude", accel_mag, top_linestyle='-', bottom_linestyle=':')

    # Pairwise stacked change plots
    plot_stacked_pair("graphics/torque_gyro_stacked_change", "Gyro Change (High-Pass, Log Scale)", "Gyro Change", gyro_delta, "Torque Change (High-Pass, Log Scale)", "Torque Change", torque_delta)
    plot_stacked_pair("graphics/torque_accel_stacked_change", "Accel Change (High-Pass, Log Scale)", "Accel Change", accel_delta, "Torque Change (High-Pass, Log Scale)", "Torque Change", torque_delta, top_linestyle=':', bottom_linestyle='--')
    plot_stacked_pair("graphics/gyro_accel_stacked_change", "Gyro Change (High-Pass, Log Scale)", "Gyro Change", gyro_delta, "Acceleration Change (High-Pass, Log Scale)", "Accel Change", accel_delta, top_linestyle='-', bottom_linestyle=':')

    # Triple stacked plots (all sensors)
    plot_stacked_triple(
        "graphics/torque_gyro_accel_stacked_curve",
        ["Gyro Magnitude (Log Scale)", "Torque (Log Scale)", "Acceleration Magnitude (Log Scale)"],
        ["Gyro Magnitude", "Torque", "Accel Magnitude"],
        [gyro_mag, torque_avg, accel_mag],
        ['-', '--', ':']
    )
    plot_stacked_triple(
        "graphics/torque_gyro_accel_stacked_change",
        ["Gyro Change (High-Pass, Log Scale)", "Torque Change (High-Pass, Log Scale)", "Acceleration Change (High-Pass, Log Scale)"],
        ["Gyro Change", "Torque Change", "Accel Change"],
        [gyro_delta, torque_delta, accel_delta],
        ['-', '--', ':']
    )

    # Single-sensor history plots
    plot_single("graphics/torque_curve_only", "Roller Torque History (Average Left/Right)", "Torque (log scale)", torque_avg, linestyle='--')
    plot_single("graphics/gyro_curve_only", "Roller Gyro History (Magnitude)", "Gyro Magnitude (log scale)", gyro_mag)
    plot_single("graphics/accel_curve_only", "Roller Acceleration History (Magnitude)", "Acceleration (log scale)", accel_mag, linestyle=':')

    # Single-sensor change plots
    plot_single("graphics/torque_change_only", "Change in Roller Torque (High-Pass)", "Torque Change (log scale)", torque_delta, linestyle='--')
    plot_single("graphics/gyro_change_only", "Change in Roller Gyro Magnitude", "Gyro Change (log scale)", gyro_delta)
    plot_single("graphics/accel_change_only", "Change in Roller Acceleration Magnitude", "Acceleration Change (log scale)", accel_delta, linestyle=':')

    axis_title_map = {"x": "X", "y": "Y", "z": "Z"}
    for axis_key in axis_labels:
        axis_suffix = axis_title_map[axis_key]
        plot_axis_single(
            f"graphics/gyro_axis_{axis_key}_curve",
            f"Gyro {axis_suffix}-Axis History",
            f"Gyro {axis_suffix}-Axis",
            gyro_axes[axis_key],
            linestyle='-',
        )
        plot_axis_single(
            f"graphics/accel_axis_{axis_key}_curve",
            f"Acceleration {axis_suffix}-Axis History",
            f"Acceleration {axis_suffix}-Axis",
            accel_axes[axis_key],
            linestyle='--',
        )
        plot_axis_change_single(
            f"graphics/gyro_axis_{axis_key}_change",
            f"Gyro {axis_suffix}-Axis Change (High-Pass)",
            f"Gyro Δ{axis_suffix}",
            gyro_axes_delta[axis_key],
            linestyle='-',
        )
        plot_axis_change_single(
            f"graphics/accel_axis_{axis_key}_change",
            f"Acceleration {axis_suffix}-Axis Change (High-Pass)",
            f"Accel Δ{axis_suffix}",
            accel_axes_delta[axis_key],
            linestyle='--',
        )
        plot_axis_combined(
            f"graphics/gyro_accel_torque_axis_{axis_key}_curve",
            axis_suffix,
            gyro_axes[axis_key],
            accel_axes[axis_key],
            torque_avg,
        )
        plot_axis_combined_change(
            f"graphics/gyro_accel_torque_axis_{axis_key}_change",
            axis_suffix,
            gyro_axes_delta[axis_key],
            accel_axes_delta[axis_key],
            torque_delta,
        )
        plot_axis_with_torque(
            f"graphics/gyro_axis_{axis_key}_torque_curve",
            axis_suffix,
            "Gyro",
            gyro_axes[axis_key],
            torque_delta,
            sensor_linestyle='-',
            torque_label="Torque Change",
        )
        plot_axis_with_torque(
            f"graphics/accel_axis_{axis_key}_torque_curve",
            axis_suffix,
            "Accel",
            accel_axes[axis_key],
            torque_delta,
            sensor_linestyle='--',
            torque_label="Torque Change",
        )
        plot_axis_with_torque_change(
            f"graphics/gyro_axis_{axis_key}_torque_change",
            axis_suffix,
            "Gyro",
            gyro_axes_delta[axis_key],
            torque_delta,
            sensor_linestyle='-',
            torque_label="Torque Change",
        )
        plot_axis_with_torque_change(
            f"graphics/accel_axis_{axis_key}_torque_change",
            axis_suffix,
            "Accel",
            accel_axes_delta[axis_key],
            torque_delta,
            sensor_linestyle='--',
            torque_label="Torque Change",
        )

    derived_bundle = {
        "torque_avg": torque_avg,
        "torque_delta": torque_delta,
        "gyro_mag": gyro_mag,
        "gyro_delta": gyro_delta,
        "accel_mag": accel_mag,
        "accel_delta": accel_delta,
        "gyro_axes": gyro_axes,
        "gyro_axes_delta": gyro_axes_delta,
        "accel_axes": accel_axes,
        "accel_axes_delta": accel_axes_delta,
    }

    export_sensor_history(
        torque_log,
        gyro_log,
        accel_log,
        derived_metrics=derived_bundle,
        sample_start=sample_start,
        sample_end=sample_end,
    )

def roller_actuator_rotation():
    """
    Loads a MuJoCo model from XML, sets target velocities for actuators,
    and runs a roller with visualization.
    """
    try:
        # Load the MuJoCo model from the XML string
        model = mujoco.MjModel.from_xml_path("roller/scene.xml")
        data = mujoco.MjData(model)

        # Ensure the model and data are valid
        if model is None:
            print("Error: Could not load MuJoCo model from XML string.")
            return
        
        target_body_name = 'control'
        target_body_id = model.body(target_body_name).id


        if target_body_id == -1:
            print(f"Error: Body '{target_body_name}' not found in the model.")
            exit()

        command_history_length = 100
        speeds = np.concatenate((np.ones((command_history_length//2))*5, np.ones((command_history_length//2))*5))
        measured_torques = np.zeros(command_history_length)

        bugs = {
            "roller1": [model.actuator("L1").id, model.actuator("R1").id, model.site("roller1").id, "roller1_accel", "roller1_gyro", "L1_w", "R1_w", "0.5", "L1_torque", "R1_torque", speeds, measured_torques],
            "roller2": [model.actuator("L2").id, model.actuator("R2").id, model.site("roller2").id, "roller2_accel", "roller2_gyro", "L2_w", "R2_w", "0.5", "L2_torque", "R2_torque", speeds, measured_torques],
            "roller3": [model.actuator("L3").id, model.actuator("R3").id, model.site("roller3").id, "roller3_accel", "roller3_gyro", "L3_w", "R3_w", "0.5", "L3_torque", "R3_torque", speeds, measured_torques],
            "roller4": [model.actuator("L4").id, model.actuator("R4").id, model.site("roller4").id, "roller4_accel", "roller4_gyro", "L4_w", "R4_w", "0.5", "L4_torque", "R4_torque", speeds, measured_torques],
            "roller5": [model.actuator("L5").id, model.actuator("R5").id, model.site("roller5").id, "roller5_accel", "roller5_gyro", "L5_w", "R5_w", "0.5", "L5_torque", "R5_torque", speeds, measured_torques],
            "roller6": [model.actuator("L6").id, model.actuator("R6").id, model.site("roller6").id, "roller6_accel", "roller6_gyro", "L6_w", "R6_w", "0.5", "L6_torque", "R6_torque", speeds, measured_torques],

        }
        torque_log = {name: [] for name in bugs}
        gyro_log = {name: [] for name in bugs}
        accel_log = {name: [] for name in bugs}

        grav = np.array([0,0,1])


        labels = [
            "ax", "ay", "az", 
            "vx", "vy", "vz", 
             "grav%", "command", "L_w", "R_w", "L_tau", "R_tau"
                ]
        label_line = " ".join(f"{label:>7}" for label in labels)



        do_print = False
        current_pos = 0
        # Initialize the viewer
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:

                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = target_body_id

                        # Set the camera to tracking mode
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = target_body_id

                # Optional: Adjust initial camera distance, azrollerth, elevation
                # These values are relative to the tracked body and can still be adjusted by the user in the viewer
                viewer.cam.distance = 3.0  # Distance from the target
                viewer.cam.azimuth = 90    # Horizontal angle
                viewer.cam.elevation = -10 # Vertical angle

                print("\nMuJoCo roller Started. Press ESC or close window to exit.")
                print("Actuator Target Velocities (rad/s):")

                # roller loop
                
                boot_time = data.time
                last_print_time = data.time
                prev_time = data.time
                print(boot_time)
                while viewer.is_running():

                    current_time = data.time
                    dt = current_time - prev_time if prev_time is not None else model.opt.timestep
                    if dt <= 0:
                        dt = model.opt.timestep
                    prev_time = current_time

                    if (current_time - last_print_time) >= 0.01:
                        do_print = True
                        last_print_time = current_time
                        print("--------------------------------------------------------------------------------")
                        print(label_line)
                    else:
                        do_print = False

                    

                    for name, [L_id, R_id,sens_id, accel_name,vel_name, left_name, right_name, velocity, left_torque_name, right_torque_name, commands, torques] in bugs.items():

                        accel_vec = np.array(data.sensor(accel_name).data, dtype=float)
                        gyro_vec = np.array(data.sensor(vel_name).data, dtype=float)
                        left_w = float(data.sensor(left_name).data[0])
                        right_w = float(data.sensor(right_name).data[0])
                        left_tau = float(data.sensor(left_torque_name).data[0])
                        right_tau = float(data.sensor(right_torque_name).data[0])
                        avg_tau = 0.5 * (left_tau + right_tau)
                        torque_log[name].append((left_tau, right_tau))
                        gyro_log[name].append(gyro_vec)
                        accel_log[name].append(accel_vec)
                        command = commands[current_pos]

                        accel_norm = np.linalg.norm(accel_vec)
                        if accel_norm > 0:
                            accel = accel_vec / accel_norm
                        else:
                            accel = accel_vec
                        vel = gyro_vec
                        angle = np.dot(accel, grav)

                        base_speed = 250/5
                        

                        speed = base_speed * command
                        
                        data.ctrl[L_id] = speed
                        data.ctrl[R_id] = speed

                        #look for torque spike

                        # print(speed)
                        if do_print:
                            temp0 = np.concatenate((accel, vel))
                            
                            temp1 = [angle, speed, left_w, right_w, left_tau, right_tau]

                            # temp = np.concatenate((temp0,temp1))
                            # printable = np.round(np.concatenate((temp, [speed])),3)

                            # printable = np.round(np.concatenate((temp0, [speed])),3)
                            printable = np.round(np.concatenate((temp0, temp1)),3)

                            value_line = " ".join(f"{val:>7.3g}" for val in printable)
                            print(value_line)

                    if do_print:
                        do_print = False

                    current_pos = (current_pos + 1)%command_history_length #wrap around code
                    
                    # Step the roller forward
                    mujoco.mj_step(model, data)

                    
                    viewer.sync()

                
                    time.sleep(0.005)
                    pass
        finally:
            print("done")
            # plot_sensor_history(torque_log, gyro_log, accel_log)
            
    except Exception as e:
        print(f"An error occurred during roller: {e}")

if __name__ == "__main__":
    # Call the roller function when the script is executed
    roller_actuator_rotation()
