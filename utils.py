import os
import subprocess

from importlib import reload


def ensure_hypll_repo(ref: str, repo_dir: str = "hyperbolic_learning_library"):
    repo_url = "git@github.com:tiberiuiancu/hyperbolic_learning_library.git"
    cwd = os.getcwd()
    target_path = os.path.join(cwd, repo_dir)

    # Clone if not present
    if not os.path.isdir(target_path):
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

        try:
            subprocess.run(["python", "-m", "pip", "install", "-e", repo_dir])
        except subprocess.CalledProcessError:
            raise RuntimeError(f"Failed to install")

    # Fetch and checkout the specific commit
    try:
        subprocess.run(["git", "fetch"], cwd=target_path, check=True)
        subprocess.run(["git", "checkout", ref], cwd=target_path, check=True)
        try:
            subprocess.run(["git", "pull"], cwd=target_path, check=True)
        except subprocess.CalledProcessError:
            print("Could not git pull; ref is likely not a branch")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to checkout ref '{ref}' in {target_path}")

    try:
        import hypll

        reload(hypll)
    except ImportError:
        raise ImportError(f"Could not import hypll")
