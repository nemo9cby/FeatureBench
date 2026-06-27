"""
Modal Sandbox backend for inference (drop-in alternative to ContainerManager).

This mirrors the public surface of `featurebench.infer.container.ContainerManager`
but runs each task in a `modal.Sandbox` instead of a local Docker container, so
benchmark images run natively on amd64 (and on real GPUs) without a local Docker
daemon.

Only the methods the inference pipeline actually calls are implemented:
    pull_image, create_container, exec_command, exec_command_stream,
    copy_to_container, copy_from_container, stop_container, get_container_logs

The returned "container" is a thin `ModalContainer` shim exposing the attributes
the runner touches directly (`id`, `short_id`, `status`, `reload`, `kill`,
`remove`) so registration/cleanup code in run_infer.py works unchanged.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal

from .container import strip_ansi_codes

# Default Modal app that owns all inference sandboxes.
_MODAL_APP_NAME = "featurebench-infer"

# Max lifetime of a sandbox (seconds). Generous: agent runs can take >30 min.
# The per-command timeout passed to exec_command_stream is the real guardrail.
_DEFAULT_SANDBOX_TIMEOUT = 4 * 60 * 60

# Default GPU type used when a task requires a GPU. Overridable via env_vars
# key "FB_MODAL_GPU" (e.g. "A100", "A100-80GB", "H100", "L40S").
_DEFAULT_GPU_TYPE = "A100"


class ModalContainer:
    """Shim wrapping a modal.Sandbox to look enough like a docker-py Container.

    run_infer.py registers/cleans up containers by touching `.id`, `.short_id`,
    `.status`, `.reload()`, `.kill()`, `.remove()` directly; everything else
    goes through the manager.
    """

    def __init__(self, sandbox: "modal.Sandbox"):
        self.sandbox = sandbox
        self.id = sandbox.object_id
        self.short_id = sandbox.object_id[:22]
        # docker-py exposes `.status`; cleanup only checks for "running".
        self.status = "running"

    def reload(self) -> None:
        # Modal has no per-call refresh we need; cleanup tolerates this being a no-op.
        return None

    def stop(self, timeout: int = 10) -> None:
        self._terminate()

    def kill(self) -> None:
        self._terminate()

    def remove(self, force: bool = False) -> None:
        self._terminate()

    def _terminate(self) -> None:
        self.status = "exited"
        try:
            self.sandbox.terminate()
        except Exception:
            pass


class ModalContainerManager:
    """Manages Modal sandboxes for inference (drop-in for ContainerManager)."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.env_vars = env_vars or {}
        # Present for parity with ContainerManager; no docker client exists.
        self.client = None

        self._app = modal.App.lookup(_MODAL_APP_NAME, create_if_missing=True)
        # Cache from_registry image handles per image name.
        self._image_cache: Dict[str, "modal.Image"] = {}
        self.sandbox_timeout = _DEFAULT_SANDBOX_TIMEOUT

    # ------------------------------------------------------------------ images
    def _get_image(self, image_name: str) -> "modal.Image":
        if image_name not in self._image_cache:
            # add_python=None: use the image's own interpreter; we only exec bash.
            self._image_cache[image_name] = modal.Image.from_registry(
                image_name, add_python=None
            )
        return self._image_cache[image_name]

    def pull_image(self, image_name: str) -> bool:
        """No-op pre-pull: the real fetch happens lazily on Sandbox.create.

        We construct (and cache) the image handle here so failures surface early
        and the API matches ContainerManager.
        """
        self._get_image(image_name)
        self.logger.info(f"Prepared Modal image handle for {image_name}")
        return True

    # -------------------------------------------------------------- containers
    def create_container(
        self,
        image_name: str,
        container_name: Optional[str] = None,
        working_dir: str = "/testbed",
        extra_env: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, Dict]] = None,
        use_host_network: bool = False,
        proxy_port: Optional[int] = None,
        gpu_ids: Optional[str] = None,
        docker_runtime_config: Optional[Dict[str, Any]] = None,
    ) -> ModalContainer:
        docker_runtime_config = docker_runtime_config or {}

        # Merge env: global agent env < repo runtime env < task-specific extra_env.
        env: Dict[str, str] = dict(self.env_vars)
        env_vars_from_config = docker_runtime_config.get("env_vars", {})
        if env_vars_from_config:
            env.update(env_vars_from_config)
            self.logger.info(
                f"Added environment variables from config: {list(env_vars_from_config.keys())}"
            )
        if extra_env:
            env.update(extra_env)

        # Modal has no docker0 host gateway; the localhost-rewrite from the docker
        # backend is intentionally dropped. Public APIs (e.g. OpenRouter) work as-is.
        str_env = {str(k): str(v) for k, v in env.items() if v is not None and v != ""}

        # GPU mapping.
        need_gpu = bool(docker_runtime_config.get("need_gpu"))
        number_once = docker_runtime_config.get("number_once", 1)
        if not isinstance(number_once, int) or number_once <= 0:
            number_once = 1
        gpu_spec: Optional[str] = None
        if need_gpu:
            gpu_type = str(self.env_vars.get("FB_MODAL_GPU") or _DEFAULT_GPU_TYPE)
            gpu_spec = gpu_type if number_once == 1 else f"{gpu_type}:{number_once}"
            self.logger.info(f"Requesting Modal GPU: {gpu_spec}")
        else:
            self.logger.info("This task does not require GPU; creating CPU sandbox")

        image = self._get_image(image_name)

        create_kwargs: Dict[str, Any] = {
            "image": image,
            "app": self._app,
            "timeout": self.sandbox_timeout,
            "workdir": working_dir,
        }
        if str_env:
            create_kwargs["secrets"] = [modal.Secret.from_dict(str_env)]
        if gpu_spec:
            create_kwargs["gpu"] = gpu_spec

        self.logger.info(f"Creating Modal sandbox from {image_name} ...")
        t0 = time.time()
        sandbox = modal.Sandbox.create(**create_kwargs)
        container = ModalContainer(sandbox)
        self.logger.info(
            f"Created Modal sandbox {container.short_id} in {time.time() - t0:.1f}s"
        )

        # Apply -ee environment exports (write to .bashrc), mirroring docker backend.
        env_exports = docker_runtime_config.get("env_exports", [])
        if env_exports:
            self._apply_env_exports(container, env_exports)

        return container

    def _apply_env_exports(self, container: ModalContainer, env_exports: List[str]) -> None:
        if not env_exports:
            return
        try:
            cmds = [
                'echo "" >> ~/.bashrc',
                'echo "# Custom environment variables from repo_settings" >> ~/.bashrc',
            ]
            for export_stmt in env_exports:
                escaped = export_stmt.replace("'", "'\\''")
                cmds.append(f"echo '{escaped}' >> ~/.bashrc")
            exit_code, output = self.exec_command(container, " && ".join(cmds))
            if exit_code == 0:
                self.logger.info(f"Applied {len(env_exports)} environment exports to .bashrc")
            else:
                self.logger.warning(f"Failed to apply env exports: {output}")
        except Exception as e:
            self.logger.warning(f"Error applying env exports: {e}")

    # --------------------------------------------------------------- execution
    def _wrap(self, command: str, workdir: Optional[str], skip_bashrc: bool) -> str:
        if skip_bashrc:
            full = command
        else:
            full = f"source ~/.bashrc && conda activate testbed 2>/dev/null || true && {command}"
        if workdir:
            full = f"cd {workdir} && {full}"
        return full

    def exec_command(
        self,
        container: ModalContainer,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        log_file: Optional[Path] = None,
    ) -> Tuple[int, str]:
        full_command = self._wrap(command, workdir, skip_bashrc=False)
        try:
            exec_kwargs: Dict[str, Any] = {}
            if timeout:
                exec_kwargs["timeout"] = timeout
            proc = container.sandbox.exec("bash", "-c", full_command, **exec_kwargs)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            proc.wait()
            exit_code = proc.returncode

            output = stdout or ""
            if stderr:
                output += "\n" + stderr

            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(output)
                    f.write(f"\n[Exit code: {exit_code}]\n\n")
            return exit_code, output
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            self.logger.error(error_msg)
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"$ {command}\n")
                    f.write(f"ERROR: {error_msg}\n\n")
            return -1, error_msg

    def exec_command_stream(
        self,
        container: ModalContainer,
        command: str,
        log_file: Path,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        skip_bashrc: bool = False,
    ) -> int:
        full_command = self._wrap(command, workdir, skip_bashrc=skip_bashrc)
        proc = None
        try:
            exec_kwargs: Dict[str, Any] = {}
            if timeout:
                exec_kwargs["timeout"] = timeout
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"$ {command}\n")
                f.flush()

                proc = container.sandbox.exec("bash", "-c", full_command, **exec_kwargs)
                start_time = time.time()
                timed_out = False

                # Modal stdout is line-iterable; stderr is merged in afterwards.
                try:
                    for line in proc.stdout:
                        f.write(strip_ansi_codes(line))
                        f.flush()
                        if timeout and (time.time() - start_time) > timeout:
                            timed_out = True
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            f.write(f"\n[TIMEOUT after {timeout} seconds]\n\n")
                            f.flush()
                            return -1
                except Exception as stream_err:
                    f.write(f"\n[stream error: {stream_err}]\n")
                    f.flush()

                if timed_out:
                    return -1

                proc.wait()
                # Drain any stderr the merged stream missed.
                try:
                    err = proc.stderr.read()
                    if err and err.strip():
                        f.write(strip_ansi_codes(err))
                except Exception:
                    pass
                f.write(f"\n[Exit code: {proc.returncode}]\n\n")
                f.flush()
                return proc.returncode
        except Exception as e:
            import traceback
            self.logger.error(f"Stream execution failed: {e}")
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: {e}\n")
                f.write(f"Traceback: {traceback.format_exc()}\n")
            return -1

    # ------------------------------------------------------------- file copies
    def copy_to_container(
        self, container: ModalContainer, src_path: Path, dest_path: str
    ) -> None:
        src_path = Path(src_path)
        fs = container.sandbox.filesystem
        # Ensure parent dir exists.
        parent = str(Path(dest_path).parent)
        container.sandbox.exec("bash", "-c", f"mkdir -p {parent}").wait()

        if src_path.is_file():
            fs.write_bytes(src_path.read_bytes(), dest_path)
            return

        # Directory: write each file under dest_path preserving structure.
        for item in src_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_path)
                target = f"{dest_path.rstrip('/')}/{rel.as_posix()}"
                container.sandbox.exec(
                    "bash", "-c", f"mkdir -p {str(Path(target).parent)}"
                ).wait()
                fs.write_bytes(item.read_bytes(), target)

    def copy_from_container(
        self, container: ModalContainer, src_path: str, dest_path: Path
    ) -> bool:
        dest_path = Path(dest_path)
        try:
            data = container.sandbox.filesystem.read_bytes(src_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            self.logger.info(f"Copied file {src_path} to {dest_path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to copy {src_path} from sandbox: {e}")
            return False

    # ----------------------------------------------------------------- cleanup
    def stop_container(self, container: ModalContainer, force: bool = False) -> None:
        try:
            container.sandbox.terminate()
            container.status = "exited"
            self.logger.info(f"Terminated Modal sandbox {container.short_id}")
        except Exception as e:
            self.logger.warning(f"Error terminating sandbox: {e}")

    def get_container_logs(self, container: ModalContainer) -> str:
        try:
            return container.sandbox.stdout.read()
        except Exception as e:
            return f"Failed to get logs: {e}"
