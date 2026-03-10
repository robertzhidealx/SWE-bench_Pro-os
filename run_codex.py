#!/usr/bin/env python3
"""
Enhanced Codex wrapper for SWE-Bench Pro with Daytona support and concurrent execution.

Features:
1. Docker mode (original) - Uses local Docker containers
2. Daytona mode - Uses Daytona cloud sandboxes (similar to Harbor's DaytonaEnvironment)
3. Concurrent execution - Run multiple instances in parallel using asyncio
"""

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tenacity import retry, stop_after_attempt, wait_exponential

# Add parent directory to path to import helper_code
sys.path.insert(0, str(Path(__file__).parent))
from helper_code.image_uri import get_dockerhub_image_uri
from helper_code.create_problem_statement import create_problem_statement

from datasets import load_dataset
from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    Image,
    Resources,
    SessionExecuteRequest,
)

def run_command(cmd: list[str], timeout: Optional[int] = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    try:
        return subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            **kwargs
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}") from e


def format_problem_description(instance: dict, repo_full_name: str, base_commit: str, instance_id: str) -> str:
    """Format problem description with metadata."""
    return create_problem_statement(instance)


class DockerExecutor:
    """Execute Codex agent using local Docker containers (Harbor's pattern)."""
    
    def __init__(self, dockerhub_username: str = "jefzda"):
        self.dockerhub_username = dockerhub_username
    
    async def run_instance(
        self,
        instance_id: str,
        instance: dict,
        model: str,
        output_dir: Path,
        timeout: int = 3000,
        dockerhub_username: str = "",
        status_callback=None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Run Codex on a single instance using Docker."""
        def update_status(status: str):
            if status_callback:
                status_callback(status)
        
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract repo information
        repo_base = instance.get("repo_base", "").strip()
        repo_name = instance.get("repo_name", "").strip()
        
        if not repo_base or not repo_name:
            repo = instance.get("repo", "").strip()
            if "/" in repo:
                repo_base, repo_name = repo.split("/", 1)
            else:
                return False, None, f"Missing repository information"
        
        repo_full_name = f"{repo_base}/{repo_name}"
        docker_image = get_dockerhub_image_uri(instance_id, self.dockerhub_username, repo_full_name)
        base_commit = instance.get("base_commit", "")

        # Format problem description with metadata
        problem_description = format_problem_description(instance, repo_full_name, base_commit, instance_id)
        
        work_dir = instance_dir / "work"
        work_dir.mkdir(exist_ok=True)
        
        update_status("pulling image...")
        try:
            pull_result = await asyncio.to_thread(
                run_command,
                ["docker", "pull", docker_image],
                timeout=300
            )
            if pull_result.returncode != 0:
                return False, None, f"Failed to pull Docker image: {pull_result.stderr}"
        except TimeoutError:
            return False, None, "Timeout while pulling Docker image"
        except Exception as e:
            return False, None, f"Error pulling Docker image: {str(e)}"
        
        # Generate unique container name
        container_name = f"codex-{instance_id.replace('/', '-').replace('_', '-')}-{uuid4().hex[:8]}"
        
        # Start persistent container
        print(f"  Starting container...")
        start_cmd = [
            "docker", "run",
            "--name", container_name,
            "--rm",
            "-d",
            "--entrypoint", "",
            "-v", f"{work_dir.absolute()}:/workspace",
            "-w", "/workspace",
            "-e", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}",
            "-e", f"OPENAI_BASE_URL={os.environ.get('OPENAI_BASE_URL', '')}",
            docker_image,
            "sleep", "infinity"
        ]
        
        try:
            start_result = await asyncio.to_thread(run_command, start_cmd, timeout=30)
            if start_result.returncode != 0:
                return False, None, f"Failed to start container: {start_result.stderr}"
            
            await asyncio.sleep(2)
            
            # Check if container is running
            check_result = await asyncio.to_thread(
                run_command,
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                timeout=5
            )
            if check_result.returncode != 0 or check_result.stdout.strip() != "true":
                logs_result = await asyncio.to_thread(
                    run_command,
                    ["docker", "logs", container_name],
                    timeout=5
                )
                return False, None, f"Container not running. Logs: {logs_result.stdout[:500]}"
            
            async def docker_exec(command: str, timeout_sec: int) -> subprocess.CompletedProcess:
                """Execute command in the running container."""
                return await asyncio.to_thread(
                    run_command,
                    ["docker", "exec", container_name, "bash", "-c", command],
                    timeout=timeout_sec
                )
            
            update_status("verifying repository...")
            verify_result = await docker_exec(
                """
                set -euo pipefail
                cd /app
                if [ ! -d ".git" ]; then
                    echo "ERROR: No git repo found at /app"
                    exit 1
                fi
                git status
                """,
                timeout_sec=10
            )
            if verify_result.returncode != 0:
                return False, None, f"Failed to verify repo: {verify_result.stderr}"
            
            # Only reset to base commit at runtime. Do NOT run
            # before_repo_set_cmd here — its last line checks out gold test files
            # from the solution commit, leaking test information to the agent.
            # The full before_repo_set_cmd runs at verification time in test.sh.
            update_status("resetting to base commit...")
            reset_result = await docker_exec(
                f"set -e && cd /app && git reset --hard {base_commit} && git clean -fd && git checkout {base_commit}",
                timeout_sec=30
            )
            if reset_result.returncode != 0:
                return False, None, f"Failed to reset to base commit: {reset_result.stderr}"

            update_status("installing codex...")
            install_result = await docker_exec(
                """
                set -euo pipefail

                if command -v codex &> /dev/null; then
                    echo "Codex already installed"
                    exit 0
                fi

                apt-get update -qq && apt-get install -y -qq curl 2>&1

                export NVM_DIR="$HOME/.nvm"
                curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
                [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

                nvm install 22 2>&1
                npm install -g @openai/codex@latest 2>&1

                # Symlink to /usr/local/bin so codex is always on PATH
                ln -sf "$(which codex)" /usr/local/bin/codex

                codex --version
                """,
                timeout_sec=300
            )
            if install_result.returncode != 0:
                return False, None, f"Failed to install Codex: {install_result.stderr}"
            
            update_status("running codex...")

            model_name = model.split("/")[-1]
            escaped_instruction = shlex.quote(problem_description)
            codex_home = "/workspace/.codex"

            # Setup: write auth.json (Harbor pattern)
            await docker_exec(
                f"""
                set -e
                mkdir -p /tmp/codex-secrets
                cat >/tmp/codex-secrets/auth.json <<EOF
{{
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}}
EOF
                mkdir -p {codex_home}
                ln -sf /tmp/codex-secrets/auth.json {codex_home}/auth.json
                """,
                timeout_sec=10
            )

            # Run codex (Harbor pattern)
            agent_result = await docker_exec(
                f"""
                cd /app
                export CODEX_HOME={codex_home}
                [ -s ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh
                codex exec \
                  --dangerously-bypass-approvals-and-sandbox \
                  --skip-git-repo-check \
                  --model {model_name} \
                  --json \
                  --enable unified_exec \
                  -c model_reasoning_effort=high \
                  -- \
                  {escaped_instruction} \
                  2>&1 </dev/null | tee /workspace/agent.log
                """,
                timeout_sec=3060  # 3000s for codex + 60s buffer
            )
            
            (work_dir / "agent_stdout.txt").write_text(agent_result.stdout)
            (work_dir / "agent_stderr.txt").write_text(agent_result.stderr)
            
            update_status("generating patch...")
            patch_result = await docker_exec(
                """
                set -euo pipefail
                cd /app
                git diff HEAD > /workspace/patch.diff
                """,
                timeout_sec=30
            )
            if patch_result.returncode != 0:
                return False, None, f"Failed to generate patch: {patch_result.stderr}"
            
            # Check for patch
            patch_path = work_dir / "patch.diff"
            if patch_path.exists() and patch_path.stat().st_size > 0:
                patch = patch_path.read_text()
                
                pred_path = instance_dir / f"{instance_id}.pred"
                pred_path.write_text(patch)
                
                return True, patch, None
            else:
                error_msg = "No patch generated"
                if agent_result.stderr:
                    error_msg += f": {agent_result.stderr[:500]}"
                return False, None, error_msg
                
        except TimeoutError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Error: {str(e)}"
        finally:
            # Always clean up container
            try:

                await asyncio.to_thread(
                    run_command,
                    ["docker", "stop", container_name],
                    timeout=10
                )
            except:
                try:
                    await asyncio.to_thread(
                        run_command,
                        ["docker", "rm", "-f", container_name],
                        timeout=10
                    )
                except:
                    pass


class DaytonaExecutor:
    """Execute Codex agent using Daytona cloud sandboxes (like Harbor's DaytonaEnvironment)."""
    
    def __init__(self):
        pass
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _create_sandbox(self, client: AsyncDaytona, image: str, cpus: int = 1, memory_gb: int = 4):
        """Create a Daytona sandbox."""
        resources = Resources(
            cpu=cpus,
            memory=memory_gb,
            disk=10,  # GB
        )
        
        params = CreateSandboxFromImageParams(
            image=Image.base(image),
            auto_delete_interval=0,
            resources=resources,
        )
        
        sandbox = await client.create(params=params, timeout=600)
        return sandbox
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get_session_command_with_retry(self, sandbox, session_id: str, command_id: str):
        """Get session command with retry logic."""
        return await sandbox.process.get_session_command(session_id, command_id)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get_session_command_logs_with_retry(self, sandbox, session_id: str, command_id: str):
        """Get session command logs with retry logic."""
        return await sandbox.process.get_session_command_logs(session_id, command_id)
    
    async def _poll_response(self, sandbox, session_id: str, command_id: str):
        """Poll for command completion."""
        response = await self._get_session_command_with_retry(sandbox, session_id, command_id)
        
        while response.exit_code is None:
            await asyncio.sleep(1)
            response = await self._get_session_command_with_retry(sandbox, session_id, response.id)
        
        logs = await self._get_session_command_logs_with_retry(sandbox, session_id, command_id)
        
        return {
            "stdout": logs.stdout,
            "stderr": logs.stderr,
            "return_code": int(response.exit_code),
        }
    
    async def exec(
        self,
        sandbox,
        command: str,
        cwd: str = "/app",
        timeout_sec: int = 3000,
        env: dict[str, str] | None = None,
    ):
        """Execute command in sandbox (similar to Harbor's DaytonaEnvironment.exec)."""
        session_id = str(uuid4())
        try:
            await sandbox.process.create_session(session_id)

            # Build env prefix to inject env vars at the process level (like Harbor)
            env_prefix = ""
            if env:
                env_parts = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
                env_prefix = " && ".join(env_parts) + " && "

            full_command = f"bash -lc {shlex.quote(command)}"

            if timeout_sec:
                full_command = f"timeout {timeout_sec} {full_command}"

            response = await sandbox.process.execute_session_command(
                session_id,
                SessionExecuteRequest(
                    command=f"{env_prefix}cd {cwd} && {full_command}",
                    run_async=True,
                ),
                timeout=timeout_sec,
            )
            
            if response.cmd_id is None:
                raise RuntimeError("Cannot find command ID.")
            
            result = await self._poll_response(sandbox, session_id, response.cmd_id)
            
        finally:
            try:
                # Clean up session - commented out like Harbor does
                # await sandbox.process.delete_session(session_id)
                pass
            except Exception:
                pass
        
        return result
    
    async def run_instance(
        self,
        instance_id: str,
        instance: dict,
        model: str,
        output_dir: Path,
        timeout: int = 3000,
        dockerhub_username: str = "jefzda",
        status_callback=None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Run Codex on a single instance using Daytona."""
        def update_status(status: str):
            if status_callback:
                status_callback(status)
        
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract repo information
        repo_base = instance.get("repo_base", "").strip()
        repo_name = instance.get("repo_name", "").strip()
        
        if not repo_base or not repo_name:
            repo = instance.get("repo", "").strip()
            if "/" in repo:
                repo_base, repo_name = repo.split("/", 1)
            else:
                return False, None, f"Missing repository information"
        
        repo_full_name = f"{repo_base}/{repo_name}"
        docker_image = get_dockerhub_image_uri(instance_id, dockerhub_username, repo_full_name)
        base_commit = instance.get("base_commit", "")

        # Format problem description with metadata
        problem_description = format_problem_description(instance, repo_full_name, base_commit, instance_id)

        work_dir = instance_dir / "work"
        work_dir.mkdir(exist_ok=True)

        # Create dedicated client and sandbox for this instance
        client = None
        sandbox = None
        try:
            update_status("creating client...")
            client = AsyncDaytona()
            
            update_status("creating sandbox...")
            sandbox = await self._create_sandbox(client, docker_image)
            
            update_status("verifying repository...")
            verify_result = await self.exec(
                sandbox,
                "if [ ! -d '.git' ]; then echo 'ERROR: No git repo'; exit 1; fi; git status",
                cwd="/app",
                timeout_sec=10
            )
            if verify_result["return_code"] != 0:
                return False, None, f"Failed to verify repo: {verify_result['stderr']}"
            
            # Only reset to base commit at runtime. Do NOT run
            # before_repo_set_cmd here — its last line checks out gold test files
            # from the solution commit, leaking test information to the agent.
            # The full before_repo_set_cmd runs at verification time in test.sh.
            update_status("resetting to base commit...")
            reset_result = await self.exec(
                sandbox,
                f"set -e && git reset --hard {base_commit} && git clean -fd && git checkout {base_commit}",
                cwd="/app",
                timeout_sec=30
            )
            if reset_result["return_code"] != 0:
                return False, None, f"Failed to reset to base commit: {reset_result['stderr']}"

            update_status("installing codex...")
            install_script = """
            set -e

            if command -v codex &> /dev/null; then
                echo "Codex already installed"
                exit 0
            fi

            apt-get update -qq && apt-get install -y -qq curl 2>&1

            export NVM_DIR="$HOME/.nvm"
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
            [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

            nvm install 22 2>&1
            npm install -g @openai/codex@latest 2>&1

            # Symlink to /usr/local/bin so codex is always on PATH
            ln -sf "$(which codex)" /usr/local/bin/codex

            codex --version
            """
            
            install_result = await self.exec(
                sandbox,
                install_script,
                cwd="/app",
                timeout_sec=300
            )
            if install_result["return_code"] != 0:
                return False, None, f"Failed to install Codex.\nstdout: {install_result['stdout'][-1000:]}\nstderr: {install_result['stderr'][-1000:]}"

            update_status("running codex...")
            model_name = model.split("/")[-1]
            
            # Build env dict matching Harbor's codex agent pattern
            codex_home = "/logs/agent"
            agent_env = {
                "OPENAI_API_KEY": os.environ.get('OPENAI_API_KEY', ''),
                "CODEX_HOME": codex_home,
            }
            openai_base_url = os.environ.get('OPENAI_BASE_URL', '')
            if openai_base_url:
                agent_env["OPENAI_BASE_URL"] = openai_base_url

            # Setup command: write auth.json (Harbor pattern)
            setup_script = f"""
mkdir -p /tmp/codex-secrets
cat >/tmp/codex-secrets/auth.json <<EOF
{{
  "OPENAI_API_KEY": "$OPENAI_API_KEY"
}}
EOF
mkdir -p {codex_home}
ln -sf /tmp/codex-secrets/auth.json {codex_home}/auth.json
            """
            await self.exec(
                sandbox,
                setup_script,
                cwd="/app",
                timeout_sec=10,
                env=agent_env,
            )

            # Run codex (Harbor pattern)
            escaped_instruction = shlex.quote(problem_description)
            agent_script = (
                "[ -s ~/.nvm/nvm.sh ] && . ~/.nvm/nvm.sh; "
                "codex exec "
                "--dangerously-bypass-approvals-and-sandbox "
                "--skip-git-repo-check "
                f"--model {model_name} "
                "--json "
                "--enable unified_exec "
                "-c model_reasoning_effort=high "
                "-- "
                f"{escaped_instruction} "
                "2>&1 </dev/null"
            )

            agent_result = await self.exec(
                sandbox,
                agent_script,
                cwd="/app",
                timeout_sec=3060,  # 3000s for codex + 60s buffer
                env=agent_env,
            )
            
            (work_dir / "agent_stdout.txt").write_text(agent_result["stdout"])
            (work_dir / "agent_stderr.txt").write_text(agent_result["stderr"])
            
            update_status("generating patch...")
            patch_result = await self.exec(
                sandbox,
                "git diff HEAD",
                cwd="/app",
                timeout_sec=30
            )
            
            if patch_result["return_code"] == 0 and patch_result["stdout"].strip():
                patch = patch_result["stdout"]
                
                pred_path = instance_dir / f"{instance_id}.pred"
                pred_path.write_text(patch)
                
                patch_path = work_dir / "patch.diff"
                patch_path.write_text(patch)
                
                return True, patch, None
            else:
                error_msg = "No patch generated"
                if agent_result["stderr"]:
                    error_msg += f": {agent_result['stderr'][:500]}"
                return False, None, error_msg
                
        except Exception as e:
            return False, None, f"Error: {str(e)}"
        finally:
            # Clean up sandbox first
            if sandbox:
                try:
                    await sandbox.delete()
                except Exception:
                    pass
            
            # Don't close client here - let GC handle it to avoid race conditions
            # in concurrent execution where one client closing affects another's sandbox


async def run_instances_concurrent(
    executor,
    dataset,
    model: str,
    output_dir: Path,
    timeout: int,
    skip_existing: bool,
    max_concurrent: int,
    dockerhub_username: str = "jefzda",
    quiet: bool = False,    num_retries: int = 0,):
    """Run multiple instances concurrently using asyncio semaphore with rich progress UI."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results_list = []
    
    # Create progress bars
    loading_progress = Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    
    running_progress = Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}"),
    )
    
    async def run_with_semaphore(
        instance: dict,
        loading_task: TaskID,
        num_retries: int = 0,
    ):
        """Run a single instance with semaphore control, progress tracking, and retry logic."""
        async with semaphore:
            instance_id = instance["instance_id"]
            
            # Check if already exists
            if skip_existing:
                pred_path = output_dir / instance_id / f"{instance_id}.pred"
                if pred_path.exists():
                    loading_progress.advance(loading_task)
                    return {
                        "instance_id": instance_id,
                        "success": None,
                        "skipped": True,
                        "error": None,
                    }
            
            # Truncate instance_id for display (keep first 40 chars)
            display_id = instance_id[:40] + "..." if len(instance_id) > 40 else instance_id
            
            # Add running task if not quiet
            if not quiet:
                running_task = running_progress.add_task(
                    f"{display_id}: preparing...", total=None
                )
                
                def status_updater(status: str):
                    running_progress.update(
                        running_task,
                        description=f"{display_id}: {status}"
                    )
            else:
                status_updater = None
            
            result = None
            
            # Retry loop (similar to Harbor's _execute_trial_with_retries)
            for attempt in range(num_retries + 1):
                if attempt > 0:
                    # Exponential backoff: min(2^attempt, 60) seconds
                    delay = min(2 ** attempt, 60)
                    if status_updater:
                        status_updater(f"retrying (attempt {attempt + 1}/{num_retries + 1}) in {delay}s...")
                    await asyncio.sleep(delay)
                    if status_updater:
                        status_updater(f"retry attempt {attempt + 1}/{num_retries + 1}")
                
                try:
                    success, patch, error = await executor.run_instance(
                        instance_id=instance_id,
                        instance=instance,
                        model=model,
                        output_dir=output_dir,
                        timeout=timeout,
                        dockerhub_username=dockerhub_username,
                        status_callback=status_updater,
                    )
                    
                    result = {
                        "instance_id": instance_id,
                        "success": success,
                        "skipped": False,
                        "error": error,
                        "attempts": attempt + 1,
                    }
                    
                    # If successful, append to incremental copy script immediately
                    if success:
                        with open(output_dir / "copy_tasks_incremental.sh", "a") as f:
                            f.write(f"cp -r datasets/swebenchpro4/{instance_id} datasets/swebenchpro5/{instance_id}\n")
                        break
                    
                    # If this was the last attempt, keep the result
                    if attempt == num_retries:
                        break
                    
                    # Otherwise, clean up and retry
                    if status_updater:
                        status_updater(f"failed: {error[:50] if error else 'unknown'}")
                    
                    # Clean up failed attempt artifacts
                    instance_dir = output_dir / instance_id
                    if instance_dir.exists():
                        import shutil
                        shutil.rmtree(instance_dir)
                        
                except asyncio.CancelledError:
                    # Task was cancelled (e.g., due to timeout or shutdown)
                    result = {
                        "instance_id": instance_id,
                        "success": False,
                        "skipped": False,
                        "error": "Task was cancelled",
                        "attempts": attempt + 1,
                    }
                    raise  # Re-raise to propagate cancellation
                    
                except Exception as e:
                    result = {
                        "instance_id": instance_id,
                        "success": False,
                        "skipped": False,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
                    # If this was the last attempt, break; otherwise continue retrying
                    if attempt == num_retries:
                        break
            
            # Update progress
            if not quiet:
                running_progress.remove_task(running_task)
            
            if result:
                results_list.append(result)
            loading_progress.advance(loading_task)
            
            # Update loading progress description with current stats
            success_count = sum(1 for r in results_list if r["success"] is True)
            failed_count = sum(1 for r in results_list if r["success"] is False)
            loading_progress.update(
                loading_task,
                description=f"Rate: {success_count/len(results_list):0.2%}",
            )
            
            return result
    
    # Run with progress UI
    if quiet:
        with loading_progress:
            progress_task = loading_progress.add_task(
                "Running instances...", total=len(dataset)
            )
            
            tasks = [
                run_with_semaphore(instance, progress_task, num_retries=num_retries)
                for instance in dataset
            ]
            
            results = await asyncio.gather(*tasks)
    else:
        with Live(Group(loading_progress, running_progress), refresh_per_second=10):
            progress_task = loading_progress.add_task(
                "Running instances...", total=len(dataset)
            )
            
            tasks = [
                run_with_semaphore(instance, progress_task, num_retries=num_retries)
                for instance in dataset
            ]
            
            results = await asyncio.gather(*tasks)
    
    # Calculate stats
    stats = {
        "success": sum(1 for r in results if r["success"] is True),
        "failed": sum(1 for r in results if r["success"] is False),
        "skipped": sum(1 for r in results if r["skipped"]),
    }
    
    return results, stats


async def main_async():
    parser = argparse.ArgumentParser(
        description="Run Codex agent on SWE-Bench Pro with Docker or Daytona execution"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["docker", "daytona"],
        default="docker",
        help="Execution mode: docker (local) or daytona (cloud)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o, gpt-5-mini-2025-08-07)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/codex"),
        help="Output directory",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=None,
        help="Number of instances to run",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample instances instead of taking first N",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for deterministic sampling (only with --random_sample)",
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        default=None,
        help="Comma-separated instance IDs",
    )
    parser.add_argument(
        "--instance_ids_file",
        type=str,
        default=None,
        help="Path to file containing instance IDs (one per line)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3000,
        help="Timeout per instance (seconds)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip instances with existing results",
    )
    parser.add_argument(
        "--dockerhub_username",
        type=str,
        default="jefzda",
        help="DockerHub username for image URIs (default: jefzda)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="Maximum number of concurrent instances (default: 1)",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=0,
        help="Number of retry attempts for failed tasks (default: 0)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress individual instance progress displays",
    )
    
    args = parser.parse_args()
    
    load_dotenv(override=True)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    
    # Filter instances
    if args.instance_ids_file:
        # Load instance IDs from file
        with open(args.instance_ids_file, 'r') as f:
            instance_ids = set(line.strip() for line in f if line.strip())
        dataset = [item for item in dataset if item["instance_id"] in instance_ids]
    elif args.instance_ids:
        instance_ids = set(args.instance_ids.split(","))
        dataset = [item for item in dataset if item["instance_id"] in instance_ids]
    elif args.subset_size:
        if args.random_sample:
            # Random sampling with optional seed
            import random
            if args.seed is not None:
                random.seed(args.seed)
            indices = random.sample(range(len(dataset)), min(args.subset_size, len(dataset)))
            dataset = dataset.select(indices)
        else:
            # Sequential sampling (first N)
            dataset = dataset.select(range(min(args.subset_size, len(dataset))))
    
    # Generate full shell script to copy all task directories upfront
    task_script_path = args.output_dir / "copy_tasks.sh"
    with open(task_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Copy script for {len(dataset)} tasks\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Mode: {args.mode}\n")
        if args.random_sample:
            f.write(f"# Sampling: random (seed={args.seed})\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("mkdir -p datasets/swebenchpro5\n")
        f.write("\n")
        for item in dataset:
            instance_id = item['instance_id']
            f.write(f"cp -r datasets/swebenchpro4/{instance_id} datasets/swebenchpro5/{instance_id}\n")
    
    # Initialize incremental shell script for successful tasks only
    incremental_script_path = args.output_dir / "copy_tasks_incremental.sh"
    with open(incremental_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Incremental copy script for successful tasks only\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Mode: {args.mode}\n")
        if args.random_sample:
            f.write(f"# Sampling: random (seed={args.seed})\n")
        f.write(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("mkdir -p datasets/swebenchpro5\n")
        f.write("\n")
    
    # Make both scripts executable
    import os
    os.chmod(task_script_path, 0o755)
    os.chmod(incremental_script_path, 0o755)
    
    # Create executor
    if args.mode == "docker":
        executor = DockerExecutor(dockerhub_username=args.dockerhub_username)
    elif args.mode == "daytona":
        executor = DaytonaExecutor()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Run instances
    start_time = time.time()
    
    results, stats = await run_instances_concurrent(
        executor=executor,
        dataset=dataset,
        model=args.model,
        output_dir=args.output_dir,
        timeout=args.timeout,
        skip_existing=args.skip_existing,
        max_concurrent=args.max_concurrent,
        dockerhub_username=args.dockerhub_username,
        quiet=args.quiet,
        num_retries=args.num_retries,
    )
    
    elapsed_time = time.time() - start_time
    
    # Save final summary
    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "model": args.model,
            "max_concurrent": args.max_concurrent,
            "elapsed_time_sec": elapsed_time,
            "stats": stats,
            "results": results,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Elapsed time: {elapsed_time:.1f}s")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*70}")


def main():
    """Entry point that runs the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
