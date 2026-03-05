#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI-based SGLang deployment tool.

Usage:
    python3 deploy.py configs/kimi_k2_5.yaml
"""

import os
import sys
import json
import time
import datetime
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# Config container
# ---------------------------------------------------------------------------

class Configs:
    """Lightweight key-value container with shell serialisation helpers."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", dict(data))

    # --- mapping protocol ---------------------------------------------------

    def __getattr__(self, key: str):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value):
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Configs({self._data!r})"

    # --- serialisation ------------------------------------------------------

    def to_env(self) -> str:
        """Serialise to a VAR=value string suitable for prefixing a shell command."""
        parts = []
        for k, v in self._data.items():
            parts.append(f"{k.replace('-', '_')}={v}")
        return " ".join(parts)

    def to_args(self) -> str:
        """Serialise to --key=value CLI arguments (keys use hyphens)."""
        parts = []
        for k, v in self._data.items():
            flag = k.replace("_", "-")
            if isinstance(v, bool):
                parts.append(f"--{flag}")
            else:
                parts.append(f"--{flag}={v}")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# YAML loading & config building
# ---------------------------------------------------------------------------

class YamlLoader:
    """Reads a YAML file and returns the raw dict."""

    def __init__(self, path: str):
        self._path = path

    def load(self) -> dict:
        with open(self._path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


class ConfigBuilder:
    """Converts raw YAML sections into typed Configs objects."""

    @staticmethod
    def env(raw: dict) -> Configs:
        return Configs(raw)

    @staticmethod
    def args(raw: dict) -> Configs:
        """Convert raw args dict, JSON-encoding any nested dict values."""
        data = {}
        for k, v in raw.items():
            data[k] = "'" + json.dumps(v) + "'" if isinstance(v, dict) else v
        return Configs(data)


# ---------------------------------------------------------------------------
# Host-file parser
# ---------------------------------------------------------------------------

class HostfileParser:
    """Parses the MPI host-file format: <name> <ip> slots=<n>."""

    @staticmethod
    def parse(path: str) -> Tuple[List[str], List[int], List[str]]:
        ips, slots, names = [], [], []
        with open(path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                name, ip, slot_str = parts
                names.append(name)
                ips.append(ip)
                slots.append(int(slot_str.split("=")[1]))
        return ips, slots, names


# ---------------------------------------------------------------------------
# MPI process launcher
# ---------------------------------------------------------------------------

class MPILauncher:
    """Creates run-scripts and launches sglang nodes via mpirun."""

    _MPI_CMD = "mpirun --allow-run-as-root {mpi_args} bash {bash_file}"
    _RANK_CMD = (
        "{env} python3 -m sglang.launch_server {args} "
        "1>{stdout} 2>{stderr}"
    )

    def __init__(self, rundir: str):
        self._rundir = rundir

    def launch(
        self,
        role: str,
        ips: List[str],
        node_names: List[str],
        args: Configs,
        env: Configs,
    ) -> subprocess.Popen:
        n = len(node_names)
        print(f"[{role}] Launching {n} node(s)")

        # Overwrite distributed-init fields with runtime MPI env vars.
        port = args.dist_init_addr.split(":")[1]
        args.nnodes = "$OMPI_COMM_WORLD_SIZE"
        args.node_rank = "$OMPI_COMM_WORLD_RANK"
        args.dist_init_addr = f"{ips[0]}:{port}"

        role_dir = os.path.join(self._rundir, role)
        rank_log_dir = os.path.join(role_dir, "ranks", "$OMPI_COMM_WORLD_RANK")

        self._mkdirs(role_dir, n)

        bash_cmd = self._RANK_CMD.format(
            env=env.to_env(),
            args=args.to_args(),
            stdout=f"{rank_log_dir}/stdout.log",
            stderr=f"{rank_log_dir}/stderr.log",
        )
        bash_file = os.path.join(role_dir, "run.sh")
        with open(bash_file, "w") as fh:
            fh.write(bash_cmd)

        mpi_args = f"-x PATH -npernode 1 -host {','.join(node_names)}"
        mpi_cmd = self._MPI_CMD.format(mpi_args=mpi_args, bash_file=bash_file)

        print(f"[{role}] mpi  cmd: {mpi_cmd}")
        print(f"[{role}] bash cmd: {bash_cmd}")
        return subprocess.Popen(mpi_cmd, shell=True)

    def _mkdirs(self, role_dir: str, n: int):
        subprocess.call(f"mkdir -p {role_dir}/ranks", shell=True)
        for i in range(n):
            subprocess.call(f"mkdir -p {role_dir}/ranks/{i}", shell=True)


# ---------------------------------------------------------------------------
# Process watcher
# ---------------------------------------------------------------------------

class ProcessWatcher:
    """Polls a set of named processes; kills all if any one exits."""

    _POLL_INTERVAL = 10  # seconds

    def watch(self, procs: Dict[str, subprocess.Popen]):
        while True:
            for name, proc in procs.items():
                if proc.poll() is not None:
                    print(f"[watcher] '{name}' exited — stopping all processes")
                    for p in procs.values():
                        p.kill()
                        p.wait(self._POLL_INTERVAL)
                    print("[watcher] All processes stopped.")
                    return
            time.sleep(self._POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Deployers
# ---------------------------------------------------------------------------

class BaseDeployer(ABC):
    """Common interface for deployment strategies."""

    @abstractmethod
    def deploy(self, **kwargs): ...


class AggregatedDeployer(BaseDeployer):
    """Runs unified prefill+decode nodes via a single mpirun launch."""

    def __init__(self, launcher: MPILauncher, watcher: ProcessWatcher):
        self._launcher = launcher
        self._watcher = watcher

    def deploy(
        self,
        ips: List[str],
        node_names: List[str],
        args: Configs,
        env: Configs,
        nnode: int,
    ):
        proc = self._launcher.launch(
            "prefill_decode", ips[:nnode], node_names[:nnode], args, env
        )
        self._watcher.watch({"prefill_decode": proc})


class DisaggregatedDeployer(BaseDeployer):
    """Runs separate prefill / decode clusters plus a mini load-balancer."""

    def __init__(self, launcher: MPILauncher, watcher: ProcessWatcher, rundir: str):
        self._launcher = launcher
        self._watcher = watcher
        self._rundir = rundir

    def deploy(
        self,
        ips: List[str],
        node_names: List[str],
        prefill_args: Configs,
        prefill_env: Configs,
        decode_args: Configs,
        decode_env: Configs,
        prefill_nnode: int,
        decode_nnode: int,
    ):
        prefill_ips = ips[:prefill_nnode]
        decode_ips = ips[prefill_nnode: prefill_nnode + decode_nnode]
        prefill_names = node_names[:prefill_nnode]
        decode_names = node_names[prefill_nnode: prefill_nnode + decode_nnode]

        prefill_proc = self._launcher.launch(
            "prefill", prefill_ips, prefill_names, prefill_args, prefill_env
        )
        decode_proc = self._launcher.launch(
            "decode", decode_ips, decode_names, decode_args, decode_env
        )
        lb_proc = self._launch_lb(prefill_ips[0], decode_ips[0])

        self._watcher.watch(
            {"prefill": prefill_proc, "decode": decode_proc, "load_balancer": lb_proc}
        )

    def _launch_lb(self, prefill_ip: str, decode_ip: str) -> subprocess.Popen:
        print("[lb] Starting load balancer")
        lb_log = os.path.join(self._rundir, "lb.log")
        cmd = (
            f"python3 -m sglang.srt.disaggregation.mini_lb "
            f"--prefill http://{prefill_ip}:30000 "
            f"--decode http://{decode_ip}:30000 "
            f"2>&1 1>{lb_log}"
        )
        return subprocess.Popen(cmd, shell=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DeploymentOrchestrator:
    """
    Top-level coordinator.

    Reads a YAML config, wires dependencies together, and calls the
    appropriate deployment strategy.
    """

    def __init__(self, yaml_path: str):
        self._yaml_path = yaml_path

    def run(self):
        raw = YamlLoader(self._yaml_path).load()
        deploy_cfg = raw.get("deploy", {})

        hostfile = deploy_cfg.get("hostfile", "/etc/mpi/mpi-hostfile")
        ips, _, node_names = HostfileParser.parse(hostfile)

        rundir = self._make_rundir()
        launcher = MPILauncher(rundir)
        watcher = ProcessWatcher()

        deploy_type = deploy_cfg.get("type", "aggregated")
        if deploy_type == "disaggregated":
            self._run_disaggregated(raw, deploy_cfg, ips, node_names, launcher, watcher, rundir)
        else:
            self._run_aggregated(raw, deploy_cfg, ips, node_names, launcher, watcher)

    # --- private helpers ---------------------------------------------------

    @staticmethod
    def _make_rundir() -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cwd = os.getcwd()
        rundir = os.path.join(cwd, "jobs", ts)
        subprocess.call(
            f"mkdir -p {rundir} && "
            f"unlink {cwd}/jobs/latest 2>/dev/null; "
            f"ln -sf {rundir} {cwd}/jobs/latest",
            shell=True,
        )
        return rundir

    @staticmethod
    def _run_aggregated(raw, deploy_cfg, ips, node_names, launcher, watcher):
        AggregatedDeployer(launcher, watcher).deploy(
            ips=ips,
            node_names=node_names,
            args=ConfigBuilder.args(raw.get("sglang_args", {})),
            env=ConfigBuilder.env(raw.get("env", {})),
            nnode=deploy_cfg["nnode"],
        )

    @staticmethod
    def _run_disaggregated(raw, deploy_cfg, ips, node_names, launcher, watcher, rundir):
        DisaggregatedDeployer(launcher, watcher, rundir).deploy(
            ips=ips,
            node_names=node_names,
            prefill_args=ConfigBuilder.args(raw.get("prefill_args", {})),
            prefill_env=ConfigBuilder.env(raw.get("prefill_env", {})),
            decode_args=ConfigBuilder.args(raw.get("decode_args", {})),
            decode_env=ConfigBuilder.env(raw.get("decode_env", {})),
            prefill_nnode=deploy_cfg["prefill_nnode"],
            decode_nnode=deploy_cfg["decode_nnode"],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    DeploymentOrchestrator(sys.argv[1]).run()
