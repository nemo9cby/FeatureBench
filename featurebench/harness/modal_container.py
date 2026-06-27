"""
Modal Sandbox backend for evaluation (drop-in alternative to EvalContainerManager).

The harness eval (`run_instance_level1` / `run_instance_level2`) drives the container
through just two methods -- `exec_run(...)` and `put_archive(...)` -- plus `kill()` /
`remove()` for cleanup. `ModalEvalContainer` implements that docker-py surface on top of
a `modal.Sandbox`, so the existing eval logic runs unchanged on Modal.

Sandbox lifecycle (image pull, GPU mapping, env, terminate) is delegated to the infer
backend's `ModalContainerManager` to avoid duplicating that logic.
"""

import io
import logging
import tarfile
from collections import namedtuple
from typing import Optional

from featurebench.infer.modal_container import ModalContainerManager as _InfraManager

# docker-py's exec_run returns an object with .exit_code and .output (bytes).
ExecResult = namedtuple("ExecResult", ["exit_code", "output"])


class ModalEvalContainer:
    """Adapts a modal.Sandbox to the docker-py container surface harness eval uses."""

    def __init__(self, modal_container):
        # modal_container is an infer ModalContainer (has .sandbox, .id, .short_id).
        self.sandbox = modal_container.sandbox
        self.id = modal_container.id
        self.short_id = modal_container.short_id
        self.status = "running"

    def exec_run(self, cmd, user=None, workdir=None, stream=False, demux=False, **kwargs):
        # harness passes cmd as ["/bin/bash", "-lc", inner_cmd]; the sandbox workdir is
        # already /testbed and inner commands cd explicitly, so workdir is advisory.
        if isinstance(cmd, str):
            cmd = ["/bin/bash", "-lc", cmd]
        proc = self.sandbox.exec(*cmd)
        out = proc.stdout.read()
        err = proc.stderr.read()
        proc.wait()
        combined = (out or "") + (err or "")
        # demux=False in all harness call sites -> single combined byte stream.
        return ExecResult(exit_code=proc.returncode, output=combined.encode("utf-8", errors="replace"))

    def put_archive(self, dst_dir, tar_stream):
        data = tar_stream.read() if hasattr(tar_stream, "read") else tar_stream
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                self.sandbox.exec("bash", "-c", f"mkdir -p {dst_dir}").wait()
                target = f"{dst_dir.rstrip('/')}/{member.name}"
                self.sandbox.filesystem.write_bytes(fobj.read(), target)
        return True

    def _terminate(self):
        self.status = "exited"
        try:
            self.sandbox.terminate()
        except Exception:
            pass

    def kill(self):
        self._terminate()

    def stop(self, timeout: int = 10):
        self._terminate()

    def remove(self, force: bool = False):
        self._terminate()

    def reload(self):
        return None


class ModalEvalContainerManager:
    """Manages Modal sandboxes for evaluation (drop-in for EvalContainerManager)."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        # Parity with EvalContainerManager; no docker client exists.
        self.client = None
        self._infra = _InfraManager(logger=logger)

    def pull_image_if_needed(self, image_name: str) -> None:
        self._infra.pull_image(image_name)

    def create_container(
        self,
        image_name: str,
        instance_id: str,
        n_attempt: int = 1,
        gpu_ids: Optional[str] = None,
        proxy_port: Optional[int] = None,
        docker_runtime_config: Optional[dict] = None,
        labels: Optional[dict] = None,
    ) -> ModalEvalContainer:
        # GPU is provided by Modal via docker_runtime_config["need_gpu"]; local gpu_ids
        # (host GPU pool indices) are irrelevant on Modal and are ignored.
        modal_container = self._infra.create_container(
            image_name,
            working_dir="/testbed",
            docker_runtime_config=docker_runtime_config or {},
        )
        return ModalEvalContainer(modal_container)

    def cleanup_container(self, container) -> None:
        try:
            container.kill()
        except Exception as e:
            self.logger.warning(f"Error terminating Modal sandbox: {e}")
