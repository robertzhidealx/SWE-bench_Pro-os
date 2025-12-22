"""
The script is used to evaluate the performance of the SWEAP Pro agent with Modal.

This evaluation script:
1. Takes a CSV file containing test cases and a JSON file containing patches
2. Runs each patch in a Modal sandbox environment using Docker Hub images
3. Executes the tests using local run scripts and collects results
4. Calculates overall accuracy based on test pass/fail status

Usage:
python sweap_pro_eval_modal.py \
    --raw_sample_path=data.csv \
    --patch_path={OUTPUT}/gold_patches.json \
    --output_dir={OUTPUT}/ \
    --scripts_dir=run_scripts \
    --num_workers=100 \
    --dockerhub_username=your-username

It expects:
- Local run scripts in run_scripts/{instance_id}/run_script.sh
- Local parser scripts in run_scripts/{instance_id}/parser.py
- CSV file with columns: instance_id, before_repo_set_cmd, selected_test_files_to_run, 
  base_commit, base_dockerfile, instance_dockerfile, FAIL_TO_PASS, PASS_TO_PASS

And the generated patch file (gold_patches.json) should have the following format:
[
    {
        "instance_id": "unique_id",
        "patch": "git patch content",
        "prefix": "optional_prefix"
    },
    ...
]
"""

import argparse
import asyncio
import concurrent.futures
import json
import os
import platform as py_platform
import shlex
from uuid import uuid4

try:
    import modal  # Lazy/optional: only required when not using --use_local_docker
except Exception:
    modal = None
try:
    import docker  # Optional: used when --use_local_docker is set
except Exception:
    docker = None
try:
    from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources, SessionExecuteRequest
    from tenacity import retry, stop_after_attempt, wait_exponential
except Exception:
    AsyncDaytona = None
import pandas as pd

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from helper_code.image_uri import get_dockerhub_image_uri

# Credit: prabhuteja12
def load_base_docker(iid):
    with open(f"dockerfiles/base_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def instance_docker(iid):
    with open(f"dockerfiles/instance_dockerfile/{iid}/Dockerfile") as fp:
        return fp.read()

def load_local_script(scripts_dir, instance_id, script_name):
    """Load a script file from local scripts directory."""
    script_path = os.path.join(scripts_dir, instance_id, script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    with open(script_path, 'r') as f:
        return f.read()


def create_entryscript(sample):
    before_repo_set_cmd = sample["before_repo_set_cmd"].strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(eval(sample["selected_test_files_to_run"]))
    base_commit = sample["base_commit"]
    base_dockerfile = load_base_docker(sample["instance_id"])
    instance_dockerfile = instance_docker(sample["instance_id"])
    
    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                # Convert ENV commands to export statements
                env_cmd = line.replace("ENV", "export", 1)
                env_cmds.append(env_cmd)
    
    env_cmds = "\n".join(env_cmds)

    entry_script = f"""
{env_cmds}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def create_dockerhub_tag(uid, repo_name=""):
    """
    Convert instance_id and repo name to Docker Hub compatible tag format.
    This must match the format used in the upload script.

    Args:
        uid (str): The instance_id (e.g., "django__django-12345")
        repo_name (str): The repository name from ECR (e.g., "sweap-images/nodebb.nodebb")

    Returns:
        str: Docker Hub compatible tag (e.g., "nodebb-nodebb-12345")
    """
    if repo_name:
        # For "NodeBB/NodeBB" -> repo_base="nodebb", repo_name="nodebb" 
        # Format: {repo_base}.{repo_name}-{OriginalCase}__{OriginalCase}-{hash}-{version}
        # Example: nodebb.nodebb-NodeBB__NodeBB-7b8bffd763e2155cf88f3ebc258fa68ebe18188d-vf2cf3cbd463b7ad942381f1c6d077626485a1e9e
        repo_base, repo_name_only = repo_name.lower().split("/")
        # Keep original case for the instance_id part (after removing "instance_" prefix)
        hsh = uid.replace("instance_", "")
        return f"{repo_base}.{repo_name_only}-{hsh}"
    else:
        image_name = "default"

    # Extract the tag part from the instance ID
    # For UIDs that start with a pattern like "django__django-", extract everything after position 9
    if "__" in uid and len(uid) > 9:
        tag_part = uid[9:]  # Skip the first 9 characters (e.g., "django__")
    else:
        tag_part = uid

    return f"{image_name}-{tag_part}"




def prepare_run(uid, output_dir, prefix, redo):
    uid_dir = os.path.join(output_dir, uid)
    os.makedirs(uid_dir, exist_ok=True)
    output_path = os.path.join(uid_dir, f"{prefix}_output.json")
    if not redo and os.path.exists(output_path):
        print(f"Skipping {uid} - output already exists")
        with open(output_path, "r") as f:
            return json.load(f), output_path, os.path.join(uid_dir, "workspace")
    workspace_dir = os.path.join(uid_dir, "workspace")
    os.makedirs(workspace_dir, exist_ok=True)
    return None, output_path, workspace_dir


def write_patch_snapshot(output_dir, uid, prefix, patch):
    with open(os.path.join(output_dir, uid, f"{prefix}_patch.diff"), "w") as f:
        f.write(patch)


def assemble_workspace_files(uid, scripts_dir, patch, sample):
    run_script = load_local_script(scripts_dir, uid, "run_script.sh")
    parser_script = load_local_script(scripts_dir, uid, "parser.py")
    entryscript_content = create_entryscript(sample)

    files = {
        "patch.diff": patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript_content,
    }
    return files, entryscript_content


def write_files_modal(sandbox, files):
    for rel_path, content in files.items():
        with sandbox.open(f"/workspace/{rel_path}", "w") as f:
            f.write(content)


def write_files_local(workspace_dir, files):
    for rel_path, content in files.items():
        dst = os.path.join(workspace_dir, rel_path)
        with open(dst, "w") as f:
            f.write(content)


async def write_files_daytona(sandbox, files, workspace_dir):
    """Write files to Daytona sandbox."""
    import tempfile
    # Write files to local temp location first, then upload
    for rel_path, content in files.items():
        temp_file = os.path.join(workspace_dir, rel_path)
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        with open(temp_file, "w") as f:
            f.write(content)
        await sandbox.fs.upload_file(temp_file, f"/workspace/{rel_path}")


def save_entryscript_copy(output_dir, uid, prefix, entryscript_content):
    with open(os.path.join(output_dir, uid, f"{prefix}_entryscript.sh"), "w") as f:
        f.write(entryscript_content if entryscript_content is not None else "")


def collect_outputs_modal(sandbox, output_dir, uid, prefix):
    # Save logs first (best-effort)
    try:
        with sandbox.open("/workspace/stdout.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stdout.log"), "w") as f:
                stdout_content = f_in.read()
                f.write(stdout_content if stdout_content is not None else "")
    except FileNotFoundError:
        pass
    try:
        with sandbox.open("/workspace/stderr.log", "r") as f_in:
            with open(os.path.join(output_dir, uid, f"{prefix}_stderr.log"), "w") as f:
                stderr_content = f_in.read()
                f.write(stderr_content if stderr_content is not None else "")
    except FileNotFoundError:
        pass

    # Then try to read output.json
    try:
        with sandbox.open("/workspace/output.json", "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


def collect_outputs_local(workspace_dir, output_dir, uid, prefix):
    def _copy_safe(src_name, dest_name):
        src_path = os.path.join(workspace_dir, src_name)
        dest_path = os.path.join(output_dir, uid, dest_name)
        try:
            with open(src_path, "r") as f_in:
                content = f_in.read()
        except FileNotFoundError:
            content = ""
        with open(dest_path, "w") as f_out:
            f_out.write(content if content is not None else "")

    _copy_safe("stdout.log", f"{prefix}_stdout.log")
    _copy_safe("stderr.log", f"{prefix}_stderr.log")

    # Then try to read output.json
    try:
        with open(os.path.join(workspace_dir, "output.json"), "r") as f_in:
            output = json.load(f_in)
            with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
                json.dump(output, f)
            return output
    except FileNotFoundError:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


async def collect_outputs_daytona(sandbox, output_dir, uid, prefix, workspace_dir):
    """Collect outputs from Daytona sandbox."""
    # Save logs first (best-effort)
    try:
        temp_stdout = os.path.join(workspace_dir, "stdout.log")
        await sandbox.fs.download_file("/workspace/stdout.log", temp_stdout)
        with open(temp_stdout, "r") as f:
            stdout_content = f.read()
        with open(os.path.join(output_dir, uid, f"{prefix}_stdout.log"), "w") as f:
            f.write(stdout_content if stdout_content is not None else "")
    except Exception:
        pass
    
    try:
        temp_stderr = os.path.join(workspace_dir, "stderr.log")
        await sandbox.fs.download_file("/workspace/stderr.log", temp_stderr)
        with open(temp_stderr, "r") as f:
            stderr_content = f.read()
        with open(os.path.join(output_dir, uid, f"{prefix}_stderr.log"), "w") as f:
            f.write(stderr_content if stderr_content is not None else "")
    except Exception:
        pass

    # Then try to read output.json
    try:
        temp_output = os.path.join(workspace_dir, "output.json")
        await sandbox.fs.download_file("/workspace/output.json", temp_output)
        with open(temp_output, "r") as f:
            output = json.load(f)
        with open(os.path.join(output_dir, uid, f"{prefix}_output.json"), "w") as f:
            json.dump(output, f)
        return output
    except Exception:
        print(
            f"Warning: output.json not found for {uid}. Check {prefix}_stdout.log and {prefix}_stderr.log for details"
        )
        return None


async def eval_with_daytona(patch, sample, output_dir, dockerhub_username, scripts_dir, prefix="", redo=False, block_network=False, docker_platform=None, status_callback=None):
    """Evaluate using Daytona cloud sandboxes."""
    if AsyncDaytona is None:
        raise RuntimeError("daytona SDK is not installed. Install via 'pip install daytona-sdk' or use --use_local_docker or default Modal mode")
    
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    client = None
    sandbox = None
    
    def update_status(msg):
        if status_callback:
            status_callback(msg)
    
    update_status("preparing...")
    try:
        write_patch_snapshot(output_dir, uid, prefix, patch)

        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            update_status(f"error loading scripts: {str(e)[:50]}")
            return None

        # Create Daytona client and sandbox
        client = AsyncDaytona()
        
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        
        # Create sandbox with retry logic
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async def create_sandbox():
            resources = Resources(
                cpu=1,
                memory=4,
                disk=10,
            )
            params = CreateSandboxFromImageParams(
                image=Image.base(dockerhub_image_uri),
                auto_delete_interval=0,
                resources=resources,
            )
            return await client.create(params=params, timeout=600)
            params = CreateSandboxFromImageParams(
                image=Image.base(dockerhub_image_uri),
                auto_delete_interval=0,
                resources=resources,
            )
            return await client.create(params=params, timeout=600)
        
        update_status("creating sandbox...")
        sandbox = await create_sandbox()
        update_status("sandbox created")
        
        # Create workspace directory
        session_id = str(uuid4())
        update_status("creating session...")
        await sandbox.process.create_session(session_id)
        update_status("creating workspace...")
        
        mkdir_cmd = f"bash -ic {shlex.quote('mkdir -p /workspace')}"
        cmd_req = await sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(
                command=mkdir_cmd,
                run_async=True,
            ),
            timeout=60,
        )
        await _poll_daytona_command(sandbox, session_id, cmd_req.cmd_id, max_polls=120, status_callback=update_status)
        update_status("workspace ready")
        
        # Write files
        update_status("uploading files...")
        await write_files_daytona(sandbox, files, workspace_dir)
        update_status("files uploaded")
        
        # Execute entryscript
        update_status("running tests...")
        # Wrap command like run_codex_enhanced.py does
        entryscript_cmd = f"bash -ic {shlex.quote('bash /workspace/entryscript.sh')}"
        entryscript_cmd = f"timeout 3000 {entryscript_cmd}"
        
        cmd_req = await sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(
                command=entryscript_cmd,
                run_async=True,
            ),
            timeout=3060,  # 3000s for tests + 60s buffer
        )
        result = await _poll_daytona_command(sandbox, session_id, cmd_req.cmd_id, max_polls=3120, status_callback=update_status)  # 3060s + 60s buffer
        
        if result["return_code"] != 0:
            update_status(f"tests failed (code {result['return_code']})")
        else:
            update_status("tests complete")
        
        # Collect outputs
        update_status("collecting results...")
        output = await collect_outputs_daytona(sandbox, output_dir, uid, prefix, workspace_dir)
        if output is None:
            update_status("error collecting output")
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)
        
        return output
    except Exception as e:
        update_status(f"error: {str(e)[:50]}")
        return None
    finally:
        # Clean up sandbox then client
        if sandbox:
            try:
                await sandbox.delete()
            except Exception:
                pass
        
        # Don't close client to avoid race conditions in concurrent execution
        # Let GC handle cleanup


async def _poll_daytona_command(sandbox, session_id: str, command_id: str, max_polls: int = 3600, status_callback=None):
    """Poll for Daytona command completion with timeout."""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_command():
        return await sandbox.process.get_session_command(session_id, command_id)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_logs():
        return await sandbox.process.get_session_command_logs(session_id, command_id)
    
    response = await get_command()
    poll_count = 0
    
    while response.exit_code is None:
        if poll_count >= max_polls:
            raise TimeoutError(f"Command {command_id} timed out after {max_polls} seconds")
        await asyncio.sleep(1)
        response = await get_command()
        poll_count += 1
        # Removed frequent "waiting..." status updates - they clutter the display with 10 concurrent tasks
    
    logs = await get_logs()
    
    return {
        "stdout": logs.stdout,
        "stderr": logs.stderr,
        "return_code": int(response.exit_code),
    }


def eval_with_modal(patch, sample, output_dir, dockerhub_username, scripts_dir, prefix="", redo=False, block_network=False, docker_platform=None):
    if modal is None:
        raise RuntimeError("modal is not installed. Install it or run with --use_local_docker")
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    sandbox = None
    
    print(f"Running evaluation for {uid}")
    try:
        write_patch_snapshot(output_dir, uid, prefix, patch)

        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None

        app = modal.App.lookup(name="swe-bench-pro-eval", create_if_missing=True)
        
        # Use Docker Hub image instead of ECR
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        print(f"Using Docker Hub image: {dockerhub_image_uri}")
        
        image = modal.Image.from_registry(
            dockerhub_image_uri
        )

        sandbox = modal.Sandbox.create(
            image=image,
            app=app,
            timeout=3060,  # 3000s for tests + 60s buffer
            cpu=(1, 4),
            memory=(5 * 1024, 30 * 1024),
            block_network=block_network,
        )
        
        process = sandbox.exec("mkdir", "-p", "/workspace")
        process.wait()
        
        write_files_modal(sandbox, files)
            
        process = sandbox.exec("bash", "/workspace/entryscript.sh")
        process.wait()
        
        # Check if the process was successful
        if process.returncode != 0:
            print(f"Entryscript failed for {uid} with return code: {process.returncode}")
            # Get stderr from the process directly (note: this may not work with all Modal versions)
            try:
                stderr_content = getattr(process, 'stderr', None)
                if stderr_content and hasattr(stderr_content, 'read'):
                    error_details = stderr_content.read()
                    if error_details:
                        print(f"Error details for {uid}:")
                        print(error_details[:1000])  # Print first 1000 chars
            except Exception as e:
                print(f"Failed to read stderr for {uid}: {e}")
            
        output = collect_outputs_modal(sandbox, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)
            
        return output
    except Exception as e:
        print(f"Error in eval_with_modal for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None
    finally:
        if sandbox:
            try:
                sandbox.terminate()
            except Exception:
                pass


def eval_with_docker(patch, sample, output_dir, dockerhub_username, scripts_dir, prefix="", redo=False, block_network=False, docker_platform=None):
    if docker is None:
        raise RuntimeError("docker SDK is not installed. Install via 'pip install docker' or run without --use_local_docker")
    uid = sample["instance_id"]
    existing_output, output_path, workspace_dir = prepare_run(uid, output_dir, prefix, redo)
    if existing_output is not None:
        return existing_output

    print(f"Running local-docker evaluation for {uid}")

    try:
        try:
            files, entryscript_content = assemble_workspace_files(uid, scripts_dir, patch, sample)
        except FileNotFoundError as e:
            print(f"Error loading scripts for {uid}: {e}")
            return None
        write_files_local(workspace_dir, files)
        write_patch_snapshot(output_dir, uid, prefix, patch)

        # Run container via Docker SDK
        dockerhub_image_uri = get_dockerhub_image_uri(uid, dockerhub_username, sample.get("repo", ""))
        print(f"Using Docker Hub image: {dockerhub_image_uri}")

        client = docker.from_env()
        try:
            if docker_platform:
                client.images.pull(dockerhub_image_uri, platform=docker_platform)
            else:
                client.images.pull(dockerhub_image_uri)
        except Exception as pull_err:
            # If pull fails, fall back to a local image if present; otherwise, fail this run
            try:
                client.images.get(dockerhub_image_uri)
                print(f"Using locally available image: {dockerhub_image_uri}")
            except Exception:
                print(f"Failed to pull or find image locally for {uid}: {pull_err}")
                return None

        abs_workspace_dir = os.path.abspath(workspace_dir)
        volumes = {abs_workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        run_kwargs = {
            "volumes": volumes,
            "detach": True,
            "remove": True,
            "entrypoint": "/bin/bash",  # Override image entrypoint
            "command": ["-c", "timeout 3000 bash /workspace/entryscript.sh"],
        }
        if block_network:
            run_kwargs["network_mode"] = "none"
        # Optional platform override (useful on Apple Silicon)
        if docker_platform:
            run_kwargs["platform"] = docker_platform

        container = client.containers.run(
            dockerhub_image_uri,
            **run_kwargs,
        )

        result = container.wait()
        status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
        if status_code != 0:
            print(f"Entryscript failed for {uid} with return code: {status_code}")
        # Collect outputs and logs, and save entryscript for reference
        output = collect_outputs_local(workspace_dir, output_dir, uid, prefix)
        if output is None:
            return None
        save_entryscript_copy(output_dir, uid, prefix, entryscript_content)

        return output
    except Exception as e:
        print(f"Error in eval_with_docker for {uid}: {repr(e)}")
        print(f"Error type: {type(e)}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run SWEAP Pro evaluations using Modal or local Docker with Docker Hub images and local scripts")
    parser.add_argument("--raw_sample_path", required=True, help="Path to the raw sample CSV file")
    parser.add_argument(
        "--patch_path", required=True, help="Path to the JSON file containing patches"
    )
    parser.add_argument("--output_dir", required=True, help="Directory to store evaluation outputs")
    parser.add_argument(
        "--dockerhub_username", required=True, help="Docker Hub username where sweap-images repository is located"
    )
    parser.add_argument(
        "--scripts_dir", required=True, help="Directory containing local run scripts (e.g., scripts/run_scripts)"
    )
    parser.add_argument(
        "--use_local_docker", action="store_true", help="Run locally with Docker instead of Modal"
    )
    parser.add_argument(
        "--use_daytona", action="store_true", help="Run on Daytona cloud sandboxes instead of Modal or local Docker"
    )
    parser.add_argument(
        "--docker_platform",
        default=None,
        help="Docker platform override, e.g., linux/amd64; defaults to auto-detect",
    )
    parser.add_argument(
        "--redo", action="store_true", help="Redo evaluations even if output exists"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=50,
        help="Number of workers to run evaluations in parallel",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=0,
        help="Number of retry attempts for failed evaluations (default: 0)",
    )
    parser.add_argument(
        "--block_network", action="store_true", help="Block network access inside container"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Support both JSONL and CSV input files
    if args.raw_sample_path.endswith(".jsonl"):
        raw_sample_df = pd.read_json(args.raw_sample_path, lines=True)
    else:
        raw_sample_df = pd.read_csv(args.raw_sample_path)
    
    # Replace nulls with empty strings
    raw_sample_df = raw_sample_df.fillna("")
    
    # use instance_id as index
    raw_sample_df = raw_sample_df.set_index("instance_id", drop=False)

    # each patch sample is a dict with keys: instance_id, patch, prefix
    with open(args.patch_path, "r") as f:
        patches_to_run = json.load(f)
    eval_results = {}

    # Filter patches to only include those with matching instance_ids in the raw sample data
    valid_patches = []
    missing_instances = []
    for patch_sample in patches_to_run:
        instance_id = patch_sample["instance_id"]
        if instance_id in raw_sample_df.index:
            valid_patches.append(patch_sample)
        else:
            missing_instances.append(instance_id)
    
    if missing_instances:
        print(f"Warning: Found {len(missing_instances)} patch instances not in raw sample data:")
        for missing_id in missing_instances[:5]:  # Show first 5
            print(f"  - {missing_id}")
        if len(missing_instances) > 5:
            print(f"  ... and {len(missing_instances) - 5} more")
        print(f"Proceeding with {len(valid_patches)} valid patches out of {len(patches_to_run)} total patches")

    # Select runtime
    # Auto-detect default platform if not provided: prefer linux/amd64 on Apple Silicon
    detected_platform = None
    if args.use_local_docker and args.docker_platform is None:
        try:
            if py_platform.machine().lower() in {"arm64", "aarch64"}:
                detected_platform = "linux/amd64"
        except Exception:
            detected_platform = None

    if args.use_daytona:
        eval_fn = eval_with_daytona
        use_async = True
    elif args.use_local_docker:
        eval_fn = eval_with_docker
        use_async = False
    else:
        eval_fn = eval_with_modal
        use_async = False

    # Track progress with rich progress bar and task status display
    loading_progress = Progress(
        SpinnerColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    running_progress = Progress(
        TimeElapsedColumn(),
        TextColumn("[progress.description]{task.description}")
    )
    progress_group = Group(loading_progress, running_progress)
    
    # Track running tasks for status display
    running_tasks = {}  # future -> task_id mapping
    
    with Live(progress_group, refresh_per_second=10):
        progress_task = loading_progress.add_task(
            "Evaluating patches...",
            total=len(valid_patches)
        )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Create wrapper function to handle retries
            def run_eval_with_retries(patch_sample, num_retries, progress_refs):
                """Run evaluation with retry logic."""
                import time
                instance_id = patch_sample["instance_id"]
                output_dir = args.output_dir
                display_id = instance_id[:50] + "..." if len(instance_id) > 50 else instance_id
                
                # Add task to running progress when actually starting
                task_id = running_progress.add_task(
                    f"{display_id}: starting...",
                    total=None
                )
                progress_refs['task_id'] = task_id
                
                def status_callback(status):
                    running_progress.update(task_id, description=f"{display_id}: {status}")
                
                try:
                    for attempt in range(num_retries + 1):
                        if attempt > 0:
                            # Exponential backoff: min(2^attempt, 60) seconds
                            delay = min(2 ** attempt, 60)
                            status_callback(f"retrying (attempt {attempt + 1}/{num_retries + 1}) in {delay}s...")
                            import time
                            time.sleep(delay)
                            status_callback(f"retry attempt {attempt + 1}/{num_retries + 1}")
                        
                        try:
                            # Run evaluation
                            if use_async:
                                result = asyncio.run(eval_fn(
                                    patch_sample.get("model_patch", patch_sample.get("patch", "")),
                                    raw_sample_df.loc[instance_id],
                                    args.output_dir,
                                    args.dockerhub_username,
                                    args.scripts_dir,
                                    prefix=patch_sample.get("prefix", ""),
                                    redo=args.redo,
                                    block_network=args.block_network,
                                    docker_platform=(args.docker_platform or detected_platform) if args.use_local_docker else None,
                                    status_callback=status_callback,
                                ))
                            else:
                                result = eval_fn(
                                    patch_sample.get("model_patch", patch_sample.get("patch", "")),
                                    raw_sample_df.loc[instance_id],
                                    args.output_dir,
                                    args.dockerhub_username,
                                    args.scripts_dir,
                                    prefix=patch_sample.get("prefix", ""),
                                    redo=args.redo,
                                    block_network=args.block_network,
                                    docker_platform=(args.docker_platform or detected_platform) if args.use_local_docker else None,
                                )
                            
                            # If we got a valid result, return it
                            if result is not None:
                                return result, attempt + 1
                            
                            # If result is None and this is not the last attempt, retry
                            if attempt < num_retries:
                                status_callback(f"failed: no output")
                                # Clean up failed attempt
                                import shutil
                                instance_output_dir = os.path.join(output_dir, instance_id)
                                if os.path.exists(instance_output_dir):
                                    shutil.rmtree(instance_output_dir)
                                continue
                            
                            return result, attempt + 1
                            
                        except Exception as e:
                            # If this is the last attempt, return None
                            if attempt == num_retries:
                                status_callback(f"error: {str(e)[:30]}")
                                return None, attempt + 1
                            
                            # Otherwise, clean up and retry
                            status_callback(f"error: {str(e)[:30]}")
                            import shutil
                            instance_output_dir = os.path.join(output_dir, instance_id)
                            if os.path.exists(instance_output_dir):
                                shutil.rmtree(instance_output_dir)
                    
                    return None, num_retries + 1
                finally:
                    # Always remove task from running progress when done
                    if 'task_id' in progress_refs:
                        running_progress.remove_task(progress_refs['task_id'])
            
            # Submit all tasks
            future_to_patch = {}
            future_to_progress = {}  # Track progress refs for each future
            for patch_sample in valid_patches:
                progress_refs = {}  # Will store task_id when task starts
                
                # Submit with retry wrapper
                future = executor.submit(
                    run_eval_with_retries,
                    patch_sample,
                    args.num_retries,
                    progress_refs,
                )
                
                future_to_patch[future] = patch_sample
                future_to_progress[future] = progress_refs
            
            for future in concurrent.futures.as_completed(future_to_patch):
                patch_sample = future_to_patch[future]
                instance_id = patch_sample["instance_id"]
                
                try:
                    # Get the result (if any error occurred, it will be raised here)
                    output, attempts = future.result()
                    if output is None:
                        eval_results[instance_id] = False
                    else:
                        if instance_id not in raw_sample_df.index:
                            eval_results[instance_id] = False
                        else:
                            raw_sample = raw_sample_df.loc[instance_id]
                            passed_tests = {x["name"] for x in output["tests"] if x["status"] == "PASSED"}
                            f2p = set(eval(raw_sample["fail_to_pass"]))
                            p2p = set(eval(raw_sample["pass_to_pass"]))
                            result = (f2p | p2p) <= passed_tests
                            eval_results[instance_id] = result

                    # Update progress
                    loading_progress.advance(progress_task)
                    success_count = sum(eval_results.values())
                    total_evaluated = len(eval_results)
                    current_accuracy = success_count / total_evaluated if total_evaluated > 0 else 0
                    loading_progress.update(
                        progress_task,
                        description=f"Passed: {success_count}/{total_evaluated} | Accuracy: {current_accuracy:.2%}",
                    )
                except Exception as exc:
                    eval_results[instance_id] = False
                    # Update progress
                    loading_progress.advance(progress_task)
                    success_count = sum(eval_results.values())
                    total_evaluated = len(eval_results)
                    current_accuracy = success_count / total_evaluated if total_evaluated > 0 else 0
                    loading_progress.update(
                        progress_task,
                        description=f"Passed: {success_count}/{total_evaluated} | Accuracy: {current_accuracy:.2%}",
                    )
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_results, f)
    print("Overall accuracy: ", sum(eval_results.values()) / len(eval_results))


if __name__ == "__main__":
    main()
