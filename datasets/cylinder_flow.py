import pickle
import zipfile
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader, Dataset


ZIP_FILENAME = "cylinder_flow_captioned.zip"
NORMALIZATION_FILENAME = "train_normal_stat.pkl"
SPLIT_TO_FILENAME = {
    "train": "train_downsampled_labeled.h5",
    "valid": "valid_downsampled_labeled.h5",
}
NODE_TYPE_TO_INDEX = {
    0: 0,
    4: 1,
    5: 2,
    6: 3,
}


def _read_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def extract_file(data_dir, filename):
    data_dir = Path(data_dir)
    output_path = data_dir / filename
    if output_path.exists():
        return output_path

    zip_path = data_dir / ZIP_FILENAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing dataset archive: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extract(filename, path=data_dir)

    return output_path


def load_state_normalization(data_dir):
    stats_path = extract_file(data_dir, NORMALIZATION_FILENAME)
    with open(stats_path, "rb") as file_handle:
        stats = pickle.load(file_handle)

    mean = torch.tensor(stats[:3], dtype=torch.float32)
    std = torch.tensor(stats[3:], dtype=torch.float32)
    return mean, std


def encode_node_type(node_type):
    node_type = node_type.view(-1).tolist()
    encoded = torch.zeros(len(node_type), len(NODE_TYPE_TO_INDEX), dtype=torch.float32)
    for row_index, value in enumerate(node_type):
        encoded[row_index, NODE_TYPE_TO_INDEX[int(value)]] = 1.0
    return encoded


def build_edge_index(cells):
    edge_pairs = torch.cat(
        [
            cells[:, [0, 1]],
            cells[:, [1, 2]],
            cells[:, [2, 0]],
        ],
        dim=0,
    )
    edge_pairs = torch.sort(edge_pairs, dim=1).values
    edge_pairs = torch.unique(edge_pairs, dim=0)
    reverse_pairs = edge_pairs[:, [1, 0]]
    full_edge_pairs = torch.cat([edge_pairs, reverse_pairs], dim=0)
    return full_edge_pairs.t().contiguous().long()


class CylinderFlowDataset(Dataset):
    def __init__(
        self,
        data_dir="data",
        split="train",
        input_steps=1,
        output_steps=1,
        max_samples=None,
        window_stride=1,
    ):
        if split not in SPLIT_TO_FILENAME:
            raise ValueError(f"Unknown split: {split}")
        if input_steps < 1:
            raise ValueError("input_steps must be at least 1")
        if output_steps < 1:
            raise ValueError("output_steps must be at least 1")
        if window_stride < 1:
            raise ValueError("window_stride must be at least 1")

        self.data_dir = Path(data_dir)
        self.split = split
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.window_stride = window_stride
        self.file_path = extract_file(self.data_dir, SPLIT_TO_FILENAME[split])
        self.file_handle = None
        self.state_mean, self.state_std = load_state_normalization(self.data_dir)
        self.graph_cache = {}
        self.node_feature_dim = input_steps * 3 + 2 + 4 + 6
        self.samples = self._build_sample_index(max_samples)

    def _build_sample_index(self, max_samples):
        sample_index = []
        with h5py.File(self.file_path, "r") as file_handle:
            for sample_key in sorted(file_handle.keys(), key=int):
                num_steps = file_handle[sample_key]["pressure"].shape[0]
                max_start = num_steps - self.input_steps - self.output_steps + 1
                for start_step in range(0, max_start, self.window_stride):
                    sample_index.append((sample_key, start_step))
                    if max_samples is not None and len(sample_index) >= max_samples:
                        return sample_index
        return sample_index

    def _get_file(self):
        if self.file_handle is None:
            self.file_handle = h5py.File(self.file_path, "r")
        return self.file_handle

    def _get_graph_data(self, sample_key):
        if sample_key in self.graph_cache:
            return self.graph_cache[sample_key]

        group = self._get_file()[sample_key]
        mesh_pos = torch.from_numpy(group["mesh_pos"][:]).float()
        node_type_raw = torch.from_numpy(group["node_type"][:]).long()
        node_type = encode_node_type(node_type_raw)
        edge_index = build_edge_index(torch.from_numpy(group["cells"][:]).long())

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        edge_attr = mesh_pos[target_nodes] - mesh_pos[source_nodes]

        metadata = group["metadata"]
        center = torch.from_numpy(metadata["center"][:]).float()
        global_features = torch.tensor(
            [
                center[0].item() / 1.6,
                center[1].item() / 0.4,
                float(_read_scalar(metadata["radius"][()])),
                float(_read_scalar(metadata["reynolds_number"][()])) / 1500.0,
                float(_read_scalar(metadata["u_inlet"][()])),
                float(_read_scalar(metadata["v_inlet"][()])),
            ],
            dtype=torch.float32,
        )

        self.graph_cache[sample_key] = {
            "mesh_pos": mesh_pos,
            "node_type": node_type,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "global_features": global_features,
            "prompt": _read_scalar(metadata["prompt"][()]),
        }
        return self.graph_cache[sample_key]

    def normalize_states(self, states):
        mean = self.state_mean.view(1, 1, -1)
        std = self.state_std.view(1, 1, -1)
        return (states - mean) / std

    def denormalize_states(self, states):
        mean = self.state_mean.to(states.device).view(1, 1, -1)
        std = self.state_std.to(states.device).view(1, 1, -1)
        return states * std + mean

    def build_node_features(self, inputs, mesh_pos, node_type, global_features):
        num_nodes = mesh_pos.shape[0]
        state_features = inputs.permute(1, 0, 2).reshape(num_nodes, -1)
        repeated_globals = global_features.view(1, -1).repeat(num_nodes, 1)
        return torch.cat([state_features, mesh_pos, node_type, repeated_globals], dim=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_key, start_step = self.samples[index]
        group = self._get_file()[sample_key]
        graph_data = self._get_graph_data(sample_key)

        pressure = torch.from_numpy(group["pressure"][:]).float()
        u = torch.from_numpy(group["u"][:]).float().unsqueeze(-1)
        v = torch.from_numpy(group["v"][:]).float().unsqueeze(-1)
        states = torch.cat([u, v, pressure], dim=-1)

        input_start = start_step
        input_end = start_step + self.input_steps
        target_end = input_end + self.output_steps

        input_states = states[input_start:input_end]
        target_states = states[input_end:target_end]
        normalized_inputs = self.normalize_states(input_states)
        normalized_targets = self.normalize_states(target_states)
        node_features = self.build_node_features(
            inputs=normalized_inputs,
            mesh_pos=graph_data["mesh_pos"],
            node_type=graph_data["node_type"],
            global_features=graph_data["global_features"],
        )

        return {
            "sample_id": sample_key,
            "node_features": node_features,
            "inputs": normalized_inputs,
            "targets": normalized_targets,
            "targets_raw": target_states,
            "mesh_pos": graph_data["mesh_pos"],
            "node_type": graph_data["node_type"],
            "global_features": graph_data["global_features"],
            "edge_index": graph_data["edge_index"],
            "edge_attr": graph_data["edge_attr"],
            "prompt": graph_data["prompt"],
        }

    def close(self):
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except (TypeError, ValueError, OSError):
                pass
            self.file_handle = None

    def __del__(self):
        self.close()


def collate_single_sample(batch):
    if len(batch) != 1:
        raise ValueError("This code path expects batch_size=1 for variable-size meshes.")
    return batch[0]


def create_dataloader(
    data_dir="data",
    split="train",
    input_steps=1,
    output_steps=1,
    shuffle=False,
    max_samples=None,
    window_stride=1,
):
    dataset = CylinderFlowDataset(
        data_dir=data_dir,
        split=split,
        input_steps=input_steps,
        output_steps=output_steps,
        max_samples=max_samples,
        window_stride=window_stride,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_single_sample,
    )
